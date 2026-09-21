"""Hybrid backend: TripoSR neural mesh + multi-view silhouette hull.

TripoSR reconstructs concavities and back-side details that classical shape-
from-silhouette cannot, but it also hallucinates when unsure. Silhouettes are
the opposite - conservative but geometrically hard-bounded. Intersecting the
two grids gives you what both methods agree on, which is usually a closer
approximation than either alone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from .. import carve as _carve_mod
from ..voxelize import VoxelGrid, voxelize
from . import triposr as _tri


def _align_and_intersect(a: VoxelGrid, b: VoxelGrid) -> VoxelGrid:
    """Pad the two grids to a common (H, W, D), then keep only voxels that are
    occupied in both. Colors come from `a` (neural) which is usually richer."""
    H = max(a.palette_idx.shape[0], b.palette_idx.shape[0])
    W = max(a.palette_idx.shape[1], b.palette_idx.shape[1])
    D = max(a.palette_idx.shape[2], b.palette_idx.shape[2])

    def _center_pad(grid: np.ndarray) -> np.ndarray:
        out = np.full((H, W, D), -1, dtype=grid.dtype)
        pad_y = 0  # anchor to the ground (y=0) for both
        pad_x = (W - grid.shape[1]) // 2
        pad_z = (D - grid.shape[2]) // 2
        out[pad_y:pad_y + grid.shape[0],
            pad_x:pad_x + grid.shape[1],
            pad_z:pad_z + grid.shape[2]] = grid
        return out

    a_padded = _center_pad(a.palette_idx)
    b_padded = _center_pad(b.palette_idx)

    keep = (a_padded >= 0) & (b_padded >= 0)
    out = np.where(keep, a_padded, -1).astype(np.int16)

    metadata = {
        "method": "hybrid",
        "neural": a.metadata,
        "silhouette": b.metadata,
        "kept_voxels": int(keep.sum()),
        "neural_only_voxels": int(((a_padded >= 0) & (b_padded < 0)).sum()),
        "silhouette_only_voxels": int(((b_padded >= 0) & (a_padded < 0)).sum()),
    }
    return VoxelGrid(palette_idx=out, metadata=metadata)


def build(image_paths: Sequence[str | Path], *, max_studs: int = 32,
          mc_resolution: int = 128, front_view: int = 0,
          angles=None, background_hex=None, tolerance: int = 28,
          min_views_fraction: float = 1.0, masks=None,
          **_kwargs) -> VoxelGrid:
    """Neural (TripoSR) mesh AND multi-view silhouette hull.

    All extra kwargs beyond the TripoSR ones are forwarded to space carving.
    """
    if not image_paths:
        raise ValueError("No images provided")
    paths = [Path(p) for p in image_paths]

    # TripoSR from one view.
    mesh = _tri.infer_mesh(paths[front_view], mc_resolution=mc_resolution)
    grid_neural = voxelize(mesh, max_studs=max_studs)

    # Silhouette hull from all views (or a single view, if only one supplied).
    grid_silhouette = _carve_mod.carve(
        paths, max_studs=max_studs, angles=angles,
        background_hex=background_hex, tolerance=tolerance,
        min_views_fraction=min_views_fraction, masks=masks,
    )
    return _align_and_intersect(grid_neural, grid_silhouette)
