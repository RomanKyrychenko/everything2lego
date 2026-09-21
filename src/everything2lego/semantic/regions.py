"""Semantic region dataclass.

A Region is a contiguous voxel subset with a semantic label ('foundation',
'wall', 'roof', 'opening', 'detail'). Each region carries the color info
inherited from the voxel grid plus builder-facing metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class RegionLabel(str, Enum):
    FOUNDATION = "foundation"
    WALL = "wall"
    ROOF = "roof"
    OPENING = "opening"
    DETAIL = "detail"


@dataclass
class Region:
    label: RegionLabel
    mask: np.ndarray                    # bool (H, W, D) — cells in this region
    palette_idx: np.ndarray             # int16 (H, W, D) — colors, -1 elsewhere
    metadata: dict = field(default_factory=dict)

    @property
    def bounds_yxz(self) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
        """(y_min, y_max), (x_min, x_max), (z_min, z_max) inclusive-exclusive."""
        occ = np.argwhere(self.mask)
        y = (int(occ[:, 0].min()), int(occ[:, 0].max()) + 1)
        x = (int(occ[:, 1].min()), int(occ[:, 1].max()) + 1)
        z = (int(occ[:, 2].min()), int(occ[:, 2].max()) + 1)
        return y, x, z

    def voxel_count(self) -> int:
        return int(self.mask.sum())
