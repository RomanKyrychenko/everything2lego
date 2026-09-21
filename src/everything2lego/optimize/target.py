"""Precomputed optimization target — everything the loss needs about the mesh.

Voxelizes the mesh at LEGO plate resolution (reusing `voxelize.voxelize`),
then derives:
- `occupied[y,x,z]` — is this cell inside the mesh
- `color_idx[y,x,z]` — palette color at cell (from voxelizer's own sampling)
- `surface[y,x,z]` — cell is on the outside of the shape (has ≥1 empty face)
- `normal[y,x,z,3]` — surface normal from mesh triangles (0 on interior)

Also emits `rgb_lab[y,x,z,3]` so the loss can compute Lab-space color error
directly without redoing palette lookups.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

from .. import bricks, colors
from ..voxelize import VoxelGrid, load_mesh, voxelize


def _lab_from_rgb(rgb: np.ndarray) -> np.ndarray:
    """Fast approximate sRGB → Lab (D65). Good enough for a loss."""
    # Normalize to [0,1]
    r = rgb[..., 0] / 255.0
    g = rgb[..., 1] / 255.0
    b = rgb[..., 2] / 255.0
    # Approximate linearization (skip full gamma — L2 in this space is fine).
    x = 0.4124 * r + 0.3576 * g + 0.1805 * b
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    z_ = 0.0193 * r + 0.1192 * g + 0.9505 * b
    # Divide by D65 reference white.
    xn, yn, zn = 0.9505, 1.0000, 1.0890
    fx = np.cbrt(x / xn + 1e-12)
    fy = np.cbrt(y / yn + 1e-12)
    fz = np.cbrt(z_ / zn + 1e-12)
    L = 116 * fy - 16
    A = 500 * (fx - fy)
    B = 200 * (fy - fz)
    return np.stack([L, A, B], axis=-1).astype(np.float32)


@dataclass
class Target:
    """Everything the optimizer needs to know about the target mesh."""

    occupied: np.ndarray       # (H, W, D) bool
    color_idx: np.ndarray      # (H, W, D) int16 — palette index, -1 elsewhere
    surface: np.ndarray        # (H, W, D) bool — occupied cell with ≥1 empty face
    rgb: np.ndarray            # (H, W, D, 3) uint8 — palette-quantized sample
    lab: np.ndarray            # (H, W, D, 3) float32 — same in Lab
    normal: np.ndarray         # (H, W, D, 3) float32 — surface normal (0 in interior)
    # Distance (in millimetres) from each cell's centre to the nearest point
    # on the *original* mesh surface — computed once from the source mesh, not
    # the voxelized target. Non-zero at cells the voxelization pushed away
    # from the true surface; the optimizer uses it to measure how blocky the
    # assembly is relative to the source, not just relative to the voxel target.
    surface_residual_mm: np.ndarray  # (H, W, D) float32
    # The LEGO cell type whose oriented face best matches the mesh's normal
    # at this cell. PLAIN elsewhere. The shape loss penalises the assembly
    # for using the wrong type at surface cells so slopes/tiles are chosen
    # over plain bricks/plates where the mesh surface is angled or smooth.
    ideal_type: np.ndarray  # (H, W, D) int8
    metadata: dict

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self.occupied.shape)  # type: ignore[return-value]


def _surface_cells(occupied: np.ndarray) -> np.ndarray:
    """A cell is on the surface if it's occupied and has ≥1 empty face-neighbor."""
    surf = np.zeros_like(occupied)
    H, W, D = occupied.shape
    # Any face-neighbor is empty ⇒ surface.
    surf[:-1, :, :] |= occupied[:-1, :, :] & ~occupied[1:, :, :]
    surf[1:, :, :]  |= occupied[1:, :, :]  & ~occupied[:-1, :, :]
    surf[:, :-1, :] |= occupied[:, :-1, :] & ~occupied[:, 1:, :]
    surf[:, 1:, :]  |= occupied[:, 1:, :]  & ~occupied[:, :-1, :]
    surf[:, :, :-1] |= occupied[:, :, :-1] & ~occupied[:, :, 1:]
    surf[:, :, 1:]  |= occupied[:, :, 1:]  & ~occupied[:, :, :-1]
    # Boundary cells (touching the grid edge) are also surface.
    surf[0, :, :]  |= occupied[0, :, :]
    surf[-1, :, :] |= occupied[-1, :, :]
    surf[:, 0, :]  |= occupied[:, 0, :]
    surf[:, -1, :] |= occupied[:, -1, :]
    surf[:, :, 0]  |= occupied[:, :, 0]
    surf[:, :, -1] |= occupied[:, :, -1]
    return surf


def _finite_diff_normals(occupied: np.ndarray, surface: np.ndarray) -> np.ndarray:
    """Approximate surface normal per surface cell as the gradient of the
    occupancy indicator, pointing outward (from occupied → empty).
    Cheap and good enough for slope orientation."""
    occ = occupied.astype(np.float32)
    # Central differences on padded array.
    gx = np.zeros_like(occ)
    gy = np.zeros_like(occ)
    gz = np.zeros_like(occ)
    gy[1:-1, :, :] = occ[:-2, :, :] - occ[2:, :, :]
    gx[:, 1:-1, :] = occ[:, :-2, :] - occ[:, 2:, :]
    gz[:, :, 1:-1] = occ[:, :, :-2] - occ[:, :, 2:]
    n = np.stack([gy, gx, gz], axis=-1)  # (Y,X,Z,3) — outward
    norms = np.linalg.norm(n, axis=-1, keepdims=True)
    n = np.where(norms > 1e-6, n / (norms + 1e-6), 0.0)
    # Only keep on surface cells.
    n = n * surface[..., None]
    return n.astype(np.float32)


def build_target_from_voxels(grid: VoxelGrid,
                              mesh: trimesh.Trimesh | None = None) -> Target:
    """Turn an already-voxelized grid into a Target.

    If `mesh` is provided, precomputes both:
      - `surface_residual_mm` — distance from each cell's centre to the
        nearest point on the *original* mesh surface (voxelization loss).
      - `lab` from the *original* mesh sample, not from palette quantization,
        so `L_color` sees palette-choice error rather than trivial equality.
    """
    occupied = grid.occupied()
    color_idx = grid.palette_idx.copy()
    surface = _surface_cells(occupied) & occupied
    pal = colors.palette()
    normal = _finite_diff_normals(occupied, surface)

    if mesh is not None:
        surface_residual_mm, true_rgb, mesh_normal = _compute_surface_residual_and_color(
            surface, grid, mesh)
        # Use the *mesh* normal (accurate, from trimesh face_normals) for
        # picking ideal cell types, not the occupancy-gradient finite-diff
        # normal we computed above.
        normal = mesh_normal
        # Palette-quantized fill for interior cells; true colour on surface.
        rgb = np.zeros((*occupied.shape, 3), dtype=np.uint8)
        for i, c in enumerate(pal):
            mask = (color_idx == i) & ~surface
            if mask.any():
                rgb[mask] = np.array(c.rgb, dtype=np.uint8)
        rgb[surface] = true_rgb[surface]
    else:
        surface_residual_mm = np.zeros(occupied.shape, dtype=np.float32)
        rgb = np.zeros((*occupied.shape, 3), dtype=np.uint8)
        for i, c in enumerate(pal):
            mask = color_idx == i
            if mask.any():
                rgb[mask] = np.array(c.rgb, dtype=np.uint8)

    lab = _lab_from_rgb(rgb.astype(np.float32))

    curvature = _compute_local_curvature(surface, normal)
    ideal_type = _compute_ideal_type_field(surface, normal, occupied, curvature)

    return Target(
        occupied=occupied,
        color_idx=color_idx.astype(np.int16),
        surface=surface,
        rgb=rgb,
        lab=lab,
        normal=normal,
        surface_residual_mm=surface_residual_mm,
        ideal_type=ideal_type,
        metadata=dict(grid.metadata),
    )


def _compute_local_curvature(surface: np.ndarray, normal: np.ndarray
                                ) -> np.ndarray:
    """For each surface cell, the mean angle (radians) between its normal
    and its face-neighbours' normals. Zero for flat regions; large for
    strongly curved regions like a sphere. Interior cells get 0."""
    H, W, D = surface.shape
    curvature = np.zeros(surface.shape, dtype=np.float32)
    surf_cells = np.argwhere(surface)
    offsets = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
                (0, 0, 1), (0, 0, -1))
    for y, x, z in surf_cells:
        n = normal[y, x, z]
        if float((n * n).sum()) < 0.09:
            continue
        angles = []
        for dy, dx, dz in offsets:
            ny_, nx_, nz_ = int(y + dy), int(x + dx), int(z + dz)
            if not (0 <= ny_ < H and 0 <= nx_ < W and 0 <= nz_ < D):
                continue
            if not surface[ny_, nx_, nz_]:
                continue
            n2 = normal[ny_, nx_, nz_]
            if float((n2 * n2).sum()) < 0.09:
                continue
            dot = float(np.clip(np.dot(n, n2), -1.0, 1.0))
            angles.append(float(np.arccos(dot)))
        if angles:
            curvature[y, x, z] = float(np.mean(angles))
    return curvature


def _compute_ideal_type_field(surface: np.ndarray,
                                normal: np.ndarray,
                                occupied: np.ndarray,
                                curvature: np.ndarray) -> np.ndarray:
    """For each surface cell, the LEGO cell type whose face orientation best
    matches the mesh normal there.

    Uses local curvature to distinguish flat walls from curved surfaces of
    the same normal orientation:  a sphere's equator has purely horizontal
    normals but high curvature ⇒ SLOPE; a house wall has horizontal normal
    but zero curvature ⇒ PLAIN.

    Returns an int8 grid the same shape as `surface`. Interior cells (and
    empty cells) hold PLAIN as a placeholder — the loss ignores them.
    """
    from .loss import (PLAIN, TILE,
                        SLOPE_XP, SLOPE_XM, SLOPE_ZP, SLOPE_ZM,
                        SLOPE_INV_XP, SLOPE_INV_XM, SLOPE_INV_ZP, SLOPE_INV_ZM,
                        SLOPE_CORNER_PP, SLOPE_CORNER_PN,
                        SLOPE_CORNER_NP, SLOPE_CORNER_NN)
    ideal = np.full(surface.shape, PLAIN, dtype=np.int8)
    surf_cells = np.argwhere(surface)
    # ~5.7° per neighbour = "meaningfully curved" — anything below that reads
    # as a flat wall or roof.
    CURVATURE_TOL = 0.1
    for y, x, z in surf_cells:
        ny = float(normal[y, x, z, 0])
        nx_ = float(normal[y, x, z, 1])
        nz = float(normal[y, x, z, 2])
        mag = (ny * ny + nx_ * nx_ + nz * nz) ** 0.5
        if mag < 0.3:
            continue
        k = float(curvature[y, x, z])

        # Flat top (very vertical +Y normal AND low curvature): tile.
        if ny > 0.9 and k < CURVATURE_TOL:
            ideal[y, x, z] = TILE
            continue
        # Flat bottom (very vertical -Y normal AND low curvature): plain
        # (usually invisible from above).
        if ny < -0.85 and k < CURVATURE_TOL:
            continue

        # Tilted-up (roof pitch or upper curved surface).
        if ny > 0.15 or (ny > 0.05 and k > CURVATURE_TOL):
            if abs(nx_) > 0.35 and abs(nz) > 0.35:
                if nx_ > 0 and nz > 0:
                    ideal[y, x, z] = SLOPE_CORNER_PP
                elif nx_ > 0 and nz < 0:
                    ideal[y, x, z] = SLOPE_CORNER_PN
                elif nx_ < 0 and nz > 0:
                    ideal[y, x, z] = SLOPE_CORNER_NP
                else:
                    ideal[y, x, z] = SLOPE_CORNER_NN
            elif abs(nx_) >= abs(nz):
                ideal[y, x, z] = SLOPE_XP if nx_ > 0 else SLOPE_XM
            else:
                ideal[y, x, z] = SLOPE_ZP if nz > 0 else SLOPE_ZM
            continue

        # Tilted-down (overhang / lower curved surface).
        if ny < -0.15 or (ny < -0.05 and k > CURVATURE_TOL):
            if abs(nx_) >= abs(nz):
                ideal[y, x, z] = SLOPE_INV_XP if nx_ > 0 else SLOPE_INV_XM
            else:
                ideal[y, x, z] = SLOPE_INV_ZP if nz > 0 else SLOPE_INV_ZM
            continue

        # Near-horizontal normal (wall-facing surface). If the local mesh is
        # CURVED here (a sphere's equator, a cylinder's side), we still want
        # a slope pointing outward — curved 15068 will substitute in as the
        # SA fallback chain. If NOT curved (a real flat wall), stay PLAIN.
        if k > CURVATURE_TOL:
            if abs(nx_) >= abs(nz):
                ideal[y, x, z] = SLOPE_XP if nx_ > 0 else SLOPE_XM
            else:
                ideal[y, x, z] = SLOPE_ZP if nz > 0 else SLOPE_ZM
    return ideal


def _compute_surface_residual_and_color(
        surface: np.ndarray, grid: VoxelGrid, mesh: trimesh.Trimesh
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each surface cell, compute:
      (a) mm distance from its centre to the nearest point on the mesh
      (b) the *original* mesh RGB at that nearest point (before palette lookup)
      (c) the mesh face normal at that closest point — this is the true
          surface direction, needed to pick slope orientations.

    Returns (residual_mm, true_rgb, mesh_normal). Interior cells are 0."""
    H, W, D = surface.shape
    residual = np.zeros(surface.shape, dtype=np.float32)
    true_rgb = np.zeros((H, W, D, 3), dtype=np.uint8)
    mesh_normal = np.zeros((H, W, D, 3), dtype=np.float32)
    if not surface.any():
        return residual, true_rgb, mesh_normal

    from ..voxelize import _fit_transform, _face_color_lookup
    T = _fit_transform(mesh, max_studs=max(W, D))
    Tinv = np.linalg.inv(T)

    cells = np.argwhere(surface)
    n = len(cells)
    centres_index = cells.astype(np.float32) + 0.5
    homo = np.ones((n, 4), dtype=np.float32)
    homo[:, 0] = centres_index[:, 1]
    homo[:, 1] = centres_index[:, 0]
    homo[:, 2] = centres_index[:, 2]
    world_mm = (Tinv @ homo.T).T[:, :3]

    try:
        from trimesh.proximity import ProximityQuery
        pq = ProximityQuery(mesh)
        closest_pts, distance, face_ids = pq.on_surface(world_mm)
    except Exception:
        pts, face_ids = trimesh.sample.sample_surface(mesh, 100_000)
        from scipy.spatial import cKDTree
        tree = cKDTree(pts)
        distance, nearest_idx = tree.query(world_mm)
        closest_pts = pts[nearest_idx]
        face_ids = face_ids[nearest_idx]

    # Sample mesh colour at the closest points.
    try:
        sampled_rgb = _face_color_lookup(mesh, closest_pts, np.asarray(face_ids))
        sampled_rgb = np.clip(sampled_rgb, 0, 255).astype(np.uint8)
    except Exception:
        sampled_rgb = np.full((n, 3), 200, dtype=np.uint8)

    # Mesh face normals stay in *physical* orientation. Only axis reorder
    # from mesh (x, y, z) to loss.py's (ny, nx, nz).  Do NOT rescale by the
    # anisotropic voxel-space transform — the ideal-type picker asks
    # "which face of a physical LEGO part matches this surface?" and LEGO
    # parts live in mm space, not stud/plate-index space.
    face_normals_world = np.asarray(mesh.face_normals[np.asarray(face_ids)])
    norms = np.linalg.norm(face_normals_world, axis=1, keepdims=True)
    face_normals_world = np.where(norms > 1e-6,
                                    face_normals_world / (norms + 1e-9), 0.0)
    n_reordered = np.stack([face_normals_world[:, 1],
                             face_normals_world[:, 0],
                             face_normals_world[:, 2]], axis=1)

    for (y, x, z), d, rgb, nrm in zip(cells, distance, sampled_rgb, n_reordered):
        residual[y, x, z] = float(d)
        true_rgb[y, x, z] = rgb
        mesh_normal[y, x, z] = nrm.astype(np.float32)
    return residual, true_rgb, mesh_normal


def build_target(mesh_or_path, *,
                  max_studs: int = 32,
                  color_samples: int = 200_000,
                  **voxelize_kwargs) -> Target:
    """Build a Target from a mesh path or Trimesh. Extra kwargs are forwarded
    to `voxelize()`."""
    if isinstance(mesh_or_path, (str, Path)):
        mesh = load_mesh(mesh_or_path)
    else:
        mesh = mesh_or_path
    grid = voxelize(mesh, max_studs=max_studs, color_samples=color_samples,
                    **voxelize_kwargs)
    return build_target_from_voxels(grid, mesh=mesh)
