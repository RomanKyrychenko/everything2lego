"""Voxel grid -> semantic regions.

Geometry-only heuristics for now; a follow-up patch can add opening
detection from the source mesh's face normals (where the wall surface
flips inward), and detail detection from protrusion analysis.
"""

from __future__ import annotations

import numpy as np

from ..voxelize import VoxelGrid
from .regions import Region, RegionLabel


def segment(grid: VoxelGrid, *,
             foundation_plates: int = 3,
             roof_area_ratio: float = 0.8) -> dict[RegionLabel, Region]:
    """Split a voxel grid into semantic regions.

    Foundation = bottom N plates.
    Roof       = plates from the roof line (sharpest area drop) upward.
    Wall       = everything between foundation and roof.
    Opening / detail = not yet detected (empty regions returned).
    """
    voxels = grid.palette_idx
    occ = voxels >= 0
    if not occ.any():
        raise ValueError("Empty voxel grid")

    H, W, D = voxels.shape
    # Cross-section area per Y-plate (in x-z).
    area_by_y = occ.reshape(H, -1).sum(axis=1)

    # Skip the foundation for roof-line detection.
    non_foundation = area_by_y[foundation_plates:] if H > foundation_plates else area_by_y
    if len(non_foundation) < 4:
        roof_line = H
    else:
        max_wall = non_foundation.max() if non_foundation.size else 0
        threshold = max_wall * roof_area_ratio
        # Sharpest drop (max absolute delta) below `threshold`.
        roof_line = H
        for i in range(2, len(non_foundation)):
            if non_foundation[i] < threshold:
                roof_line = foundation_plates + i
                break

    # Build masks.
    foundation_mask = np.zeros_like(occ)
    foundation_mask[:min(foundation_plates, H)] = occ[:min(foundation_plates, H)]

    roof_mask = np.zeros_like(occ)
    roof_mask[roof_line:] = occ[roof_line:]

    wall_mask = occ & ~foundation_mask & ~roof_mask

    def _region(label: RegionLabel, mask: np.ndarray) -> Region:
        pi = np.where(mask, voxels, np.int16(-1))
        return Region(label=label, mask=mask, palette_idx=pi)

    regions: dict[RegionLabel, Region] = {
        RegionLabel.FOUNDATION: _region(RegionLabel.FOUNDATION, foundation_mask),
        RegionLabel.WALL: _region(RegionLabel.WALL, wall_mask),
        RegionLabel.ROOF: _region(RegionLabel.ROOF, roof_mask),
    }
    # TODO: Opening detection needs mesh-face-normal analysis, not just
    # voxel geometry. Detail detection needs protrusion analysis (Z-scores
    # on cross-section area).  Left as empty regions so downstream code can
    # already iterate the full label set.
    regions[RegionLabel.OPENING] = _region(RegionLabel.OPENING, np.zeros_like(occ))
    regions[RegionLabel.DETAIL] = _region(RegionLabel.DETAIL, np.zeros_like(occ))
    return regions
