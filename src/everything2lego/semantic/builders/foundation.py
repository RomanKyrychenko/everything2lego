"""Foundation builder.

Takes a bottom-slab Region and emits a placement list that covers it with
the largest available plates (Plate 8x8, 6x10, etc.), colored by the voxel
grid's own colors — same treatment the template's `_foundation` uses.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from ...pack import Placement
from ...templates.base import color_index_by_code, tile_area_with_plates
from ..regions import Region


def build_foundation(region: Region) -> list[Placement]:
    """Cover the foundation Region with large plates. Colors are taken
    plate-by-plate from the majority color in each Y layer."""
    if not region.mask.any():
        return []

    (y0, y1), (x0, x1), (z0, z1) = region.bounds_yxz
    palette_idx = region.palette_idx
    placements: list[Placement] = []

    for y in range(y0, y1):
        # Dominant color in this plate layer.
        layer_colors = palette_idx[y, x0:x1, z0:z1]
        occ = layer_colors >= 0
        if not occ.any():
            continue
        counts = Counter(int(c) for c in layer_colors[occ].ravel())
        color_idx, _ = counts.most_common(1)[0]
        placements.extend(tile_area_with_plates(
            x0, z0, x1 - x0, z1 - z0,
            y=y, color_idx=color_idx,
        ))
    return placements
