"""MeshTarget — everything the pipeline needs about the source mesh.

Wraps a `trimesh.Trimesh` with cached surface samples, palette-quantised
colours, and voxel-grid representation. Immutable after construction;
Detectors and Builders read from it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh

from .. import bricks, colors
from ..voxelize import VoxelGrid, load_mesh, voxelize, _fit_transform


@dataclass
class MeshTarget:
    """Sampled representation of the mesh used by RegionDetector + Builders.

    Attributes
    ----------
    mesh : trimesh.Trimesh
        Original mesh, world coordinates in millimetres.
    max_studs : int
        Nominal grid extent; the mesh is normalised so its longer horizontal
        side fits in this many LEGO studs.
    surface_points : (N, 3) float32
        Uniformly-sampled points on the mesh surface, in mesh-mm coords.
    surface_normals : (N, 3) float32
        Mesh face-normal at each sample.
    surface_colors : (N, 3) uint8
        Original RGB at each sample (from vertex colours or texture).
    voxel_grid : VoxelGrid
        Cached voxelisation of the mesh at LEGO stud/plate resolution.
    """

    mesh: trimesh.Trimesh
    max_studs: int
    surface_points: np.ndarray
    surface_normals: np.ndarray
    surface_colors: np.ndarray
    voxel_grid: VoxelGrid
    # 4x4 transform mapping mesh-mm → voxel-index space (x=stud, y=plate,
    # z=stud). Builders MUST apply this before treating coordinates as
    # LEGO grid indices — the voxelizer scales the mesh to fit max_studs
    # anisotropically, so 1 mesh mm ≠ 1 stud.
    mesh_to_voxel: np.ndarray
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_path(cls, path: str | Path, *,
                   max_studs: int = 32,
                   n_samples: int = 100_000,
                   color_samples: int = 120_000,
                   subvoxels: int = 1) -> "MeshTarget":
        """Load a mesh from disk and build a MeshTarget."""
        mesh = load_mesh(path)
        # `voxelize.load_mesh` intentionally skips trimesh processing (for
        # reproducible voxelization). But the facet-based detector requires
        # face adjacency info, which trimesh only computes when the mesh is
        # processed. Trigger it here explicitly.
        try:
            mesh.process(validate=True)
        except Exception:
            pass
        return cls.from_mesh(mesh, max_studs=max_studs,
                              n_samples=n_samples,
                              color_samples=color_samples,
                              subvoxels=subvoxels)

    @classmethod
    def from_mesh(cls, mesh: trimesh.Trimesh, *,
                   max_studs: int = 32,
                   n_samples: int = 100_000,
                   color_samples: int = 120_000,
                   subvoxels: int = 1) -> "MeshTarget":
        pts, face_ids = trimesh.sample.sample_surface(mesh, n_samples)
        normals = np.asarray(mesh.face_normals[face_ids], dtype=np.float32)
        rgb = _sample_mesh_colors(mesh, pts, face_ids)
        # `subvoxels > 1` runs the voxeliser at N× finer occupancy grid
        # then aggregates by majority vote at the LEGO scale — recovers
        # thin walls, sharp edges, small features (window frames,
        # trim) that the coarse grid alone would alias away.
        grid = voxelize(mesh, max_studs=max_studs,
                          color_samples=color_samples,
                          subvoxels=subvoxels)
        # _fit_transform's T maps mesh_mm to voxel space with mesh CENTRED,
        # but voxelize() shifts the grid so cell (0,0,0) is at the mesh's
        # min-corner (all indices ≥ 0). Rebuild the transform so surface
        # builders' coords align with the voxel grid's indices.
        T_raw = _fit_transform(mesh, max_studs=max_studs)
        bounds_min_h = np.array([mesh.bounds[0, 0], mesh.bounds[0, 1],
                                   mesh.bounds[0, 2], 1.0], dtype=np.float64)
        min_voxel = T_raw @ bounds_min_h
        T = T_raw.copy()
        # Shift so mesh_min maps to (0, 0, 0) in voxel index space.
        T[0, 3] -= min_voxel[0]
        T[1, 3] -= min_voxel[1]
        T[2, 3] -= min_voxel[2]
        return cls(
            mesh=mesh,
            max_studs=max_studs,
            surface_points=np.asarray(pts, dtype=np.float32),
            surface_normals=normals,
            surface_colors=rgb,
            voxel_grid=grid,
            mesh_to_voxel=np.asarray(T, dtype=np.float32),
            metadata={"n_samples": n_samples},
        )

    # -- coord-frame helpers --------------------------------------------

    def mm_point_to_voxel(self, p_mm: np.ndarray) -> np.ndarray:
        """Map a mesh-mm point (x, y, z) → voxel-index (stud, plate, stud)."""
        p = np.asarray(p_mm, dtype=np.float32)
        homo = np.array([p[0], p[1], p[2], 1.0], dtype=np.float32)
        return (self.mesh_to_voxel @ homo)[:3]

    def mm_length_to_stud(self, length_mm: float) -> float:
        """Convert a horizontal mm distance to stud count."""
        return float(length_mm * self.mesh_to_voxel[0, 0])

    def mm_length_to_plate(self, length_mm: float) -> float:
        """Convert a vertical mm distance to plate count."""
        return float(length_mm * self.mesh_to_voxel[1, 1])

    # -- convenience -----------------------------------------------------

    @property
    def bounds_mm(self) -> np.ndarray:
        return np.asarray(self.mesh.bounds, dtype=np.float32)

    def n_faces(self) -> int:
        return int(len(self.mesh.faces))


def _sample_mesh_colors(mesh: trimesh.Trimesh,
                          points: np.ndarray,
                          face_ids: np.ndarray) -> np.ndarray:
    """RGB per sample point. Uses vertex colours or texture UVs if present,
    otherwise falls back to a light-gray default."""
    try:
        from ..voxelize import _face_color_lookup
        rgb = _face_color_lookup(mesh, np.asarray(points),
                                   np.asarray(face_ids))
        return np.clip(rgb, 0, 255).astype(np.uint8)
    except Exception:
        return np.full((len(points), 3), 200, dtype=np.uint8)
