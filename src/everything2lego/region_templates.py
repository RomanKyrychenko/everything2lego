"""Post-sampler region-template pass: swap each placement for a same-
footprint one in a shape family preferred by its region label
(classify_regions.LABEL).  First real step beyond the region classifier.

Not an architectural template system (no multi-cell patterns like
masonry-offset or slope-tile-cap).  It's a per-placement preference
rule: for each placement, consult the dominant region of its cells
and swap for a same-footprint-same-height part in a preferred shape
family when one exists in the vocabulary.

Rust-neutral: operates on `Placement` objects after sampling, so the
native engine's output passes through unchanged.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from .pack import Placement
from .sampler import VPart, vocabulary, _box_kind
from .colors import palette


# per region, which shape families to PREFER (in order).  The post-process
# picks the first preferred family that offers a replacement at the same
# footprint and height.
PREFERRED = {
    "roof":     ("slope", "slope_inv", "curve", "brick"),
    "wall":     ("brick", "plate"),
    "base":     ("plate", "tile", "brick"),
    "column":   ("brick",),
    "interior": ("brick", "plate"),  # usually hidden, cheap bricks are fine
    "other":    (),                  # don't touch
}

LABEL_NAME = {0: "base", 1: "wall", 2: "roof", 3: "column", 4: "interior", 5: "other"}


def _vocab_by_size() -> dict[tuple[int, int, int, str], list[VPart]]:
    """{(h, w, d, shape_family): [VPart, ...]} for shape swaps.  Keyed by
    the YAW-0 footprint; the swap re-uses the caller's rotation."""
    out: dict = {}
    for v in vocabulary():
        shape = v.part.shape or ""
        family = ("slope_inv" if shape.startswith("slope_inv")
                   else "slope" if shape.startswith("slope")
                   else "curve" if shape == "curve"
                   else _box_kind(v))
        if family is None:
            continue
        for o in v.orients:
            if o.rot != 0:
                continue
            h, w, d = o.size
            out.setdefault((h, w, d, family), []).append(v)
    return out


def _dominant_label(p: Placement, region_grid: np.ndarray) -> int:
    H, X, Z = region_grid.shape
    y0, y1 = p.y_plate, p.y_plate + p.part.height_units
    x0, x1 = p.x_stud, p.x_stud + p.w_world
    z0, z1 = p.z_stud, p.z_stud + p.d_world
    y0, y1 = max(0, y0), min(H, y1)
    x0, x1 = max(0, x0), min(X, x1)
    z0, z1 = max(0, z0), min(Z, z1)
    if y1 <= y0 or x1 <= x0 or z1 <= z0:
        return -1
    sub = region_grid[y0:y1, x0:x1, z0:z1]
    vals = sub[sub >= 0]
    if not len(vals):
        return -1
    return int(np.bincount(vals).argmax())


def apply(placements: list[Placement], region_grid: np.ndarray) -> tuple[list[Placement], dict]:
    """Return (new_placements, counts).  Keeps the same (y, x, z, yaw,
    color_idx); only the part identity changes.  Preserved: height,
    footprint, yaw.  The sampler's cell-occupancy energy is unchanged
    by the swap; connector-level terms (side, bare) may shift."""
    by_size = _vocab_by_size()
    pal = palette()
    out: list[Placement] = []
    swaps: Counter = Counter()
    for p in placements:
        label = _dominant_label(p, region_grid)
        name = LABEL_NAME.get(label)
        if name is None or not PREFERRED.get(name):
            out.append(p); continue
        current_shape = p.part.shape or ""
        current_family = ("slope_inv" if current_shape.startswith("slope_inv")
                           else "slope" if current_shape.startswith("slope")
                           else "curve" if current_shape == "curve"
                           else "brick" if "brick" in current_shape
                           else "plate" if "plate" in current_shape
                           else "tile" if "tile" in current_shape
                           else "other")
        if current_family in PREFERRED[name][:1]:
            out.append(p); continue
        h = p.part.height_units
        candidate = None
        for fam in PREFERRED[name]:
            if fam == current_family:
                break
            picks = by_size.get((h, p.w_world, p.d_world, fam), ())
            picks = [v for v in picks if v.colours is None or pal[p.color_idx].code in v.colours]
            if picks:
                candidate = picks[0]; break
        if candidate is None:
            out.append(p); continue
        out.append(Placement(candidate.part, p.color_idx, p.x_stud, p.z_stud, p.y_plate,
                              p.w_world, p.d_world, p.rotation_deg, glass=p.glass))
        swaps[name] += 1
    return out, dict(swaps)
