"""MeshTarget — everything the pipeline needs about the source mesh.

Wraps a `trimesh.Trimesh` with cached surface samples, palette-quantised
colours, and voxel-grid representation. Immutable after construction;
Detectors and Builders read from it.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh

from .. import bricks, colors
from ..voxelize import VoxelGrid, grid_phase, load_mesh, voxelize


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
                   subvoxels: int = 1,
                   min_island_cells: int = 0, align: bool = True) -> "MeshTarget":
        """Load a mesh from disk and build a MeshTarget (align=False: the mesh is already square to the grid)."""
        mesh = load_mesh(path)
        # `voxelize.load_mesh` intentionally skips trimesh processing (for
        # reproducible voxelization). But the facet-based detector requires
        # face adjacency info, which trimesh only computes when the mesh is
        # processed. Trigger it here explicitly.
        try:
            mesh.process(validate=True)
        except Exception as exc:
            warnings.warn(f"mesh cleanup failed ({exc}); facet detection may be poorer",
                          RuntimeWarning, stacklevel=2)
        return cls.from_mesh(mesh, max_studs=max_studs,
                              n_samples=n_samples,
                              color_samples=color_samples,
                              subvoxels=subvoxels, align=align,
                              min_island_cells=min_island_cells)

    @classmethod
    def from_mesh(cls, mesh: trimesh.Trimesh, *,
                   max_studs: int = 32,
                   n_samples: int = 100_000,
                   color_samples: int = 120_000,
                   subvoxels: int = 1, align: bool = True,
                   seed: int | None = None,
                   min_island_cells: int = 0) -> "MeshTarget":
        if align:
            from ..voxelize import align_to_axes
            mesh = align_to_axes(mesh)
        pts, face_ids = trimesh.sample.sample_surface(mesh, n_samples, seed=seed)
        normals = np.asarray(mesh.face_normals[face_ids], dtype=np.float32)
        rgb = _sample_mesh_colors(mesh, pts, face_ids)
        # `subvoxels > 1` runs the voxeliser at N× finer occupancy grid
        # then aggregates by majority vote at the LEGO scale — recovers
        # thin walls, sharp edges, small features (window frames,
        # trim) that the coarse grid alone would alias away.
        # The mesh keeps its true scale and is shifted so its walls fall on cell faces (fewest part-filled cells);
        # +0.5: trimesh's cell faces sit at half-integers.  Watermill: perfect-cell build's stray surface 24 % -> 9 %.
        # shortcut: with subvoxels > 1 the shift is not kept exactly (fine cells group from the mesh's min corner).
        grid = voxelize(mesh, max_studs=max_studs,
                          color_samples=color_samples,
                          subvoxels=subvoxels, align=False,
                          offset=grid_phase(mesh, max_studs) + 0.5)
        # Drop voxel islands smaller than `min_island_cells`.  A marching-
        # cubes mesh sheds specks (a 2-cell "chimney" floating above the
        # roof) that no builder can attach to; removing them here keeps
        # them out of every downstream stage.  Off by default (0): a clean
        # authored mesh can legitimately have small detached parts — a
        # window pane modelled as its own box — that a builder will place.
        # The photo CLI path turns it on.
        if min_island_cells > 1:
            from scipy.ndimage import label as _label
            lab, n_comp = _label(grid.occupied())
            if n_comp > 1:
                sizes = np.bincount(lab.ravel())
                sizes[0] = 0
                small_ids = np.flatnonzero(sizes < min_island_cells)
                if small_ids.size:
                    grid.palette_idx[np.isin(lab, small_ids)] = -1
        # the voxeliser's own mesh -> cell map, at true scale (fitting the mesh's bounds to the occupied cells
        # instead stretched it 1.5-3 % per axis and put every wall half a stud out)
        T = grid.metadata["mesh_to_voxel"]
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
    except Exception as exc:
        warnings.warn(f"colour sampling failed ({exc}); the model will be uniform grey",
                      RuntimeWarning, stacklevel=2)
        return np.full((len(points), 3), 200, dtype=np.uint8)
