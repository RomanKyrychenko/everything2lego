"""Part mix by surface type: what real sets build roofs, flat tops, walls and undersides from.

Every cell on a model's outer surface gets a type from the direction the surface faces there -- the gradient of the
blurred occupancy, as the sampler's direction term (surface_dir.target_normals) -- and a part takes the commonest
type among its cells (interior when none is on the surface).  Counted over real building sets
(tools/learn/surface_mix.py) this gives P(part class | surface type): slopes and tiles on roofs, bricks in walls,
whatever the build's overall mix.  The sampler's surface-mix term is the KL of the build's mix in each type to it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

TYPES = ("interior", "roof", "top", "wall", "under")
CLASSES = ("slope", "tile", "plate", "brick", "round", "other")
DATA = Path(__file__).parent / "data" / "surface_mix.json"


def part_class(category: str) -> int:
    c = category.lower()
    if "slope" in c or "curved" in c or "wedge" in c:
        return 0
    if "round" in c or "cone" in c or "dish" in c:
        return 4
    if c.startswith("tiles"):
        return 1
    if c.startswith("plates"):
        return 2
    if c.startswith("bricks"):
        return 3
    return 5


def cell_types(normals: np.ndarray, shape) -> np.ndarray:
    """(H, X, Z) int8 surface type per cell from unit outward directions (H*X*Z*3, y first; zero = not on the surface)."""
    n = normals.reshape(*shape, 3)
    ny = n[..., 0]
    on = np.abs(n).sum(-1) > 0
    t = np.zeros(shape, np.int8)
    t[on & (ny >= 0.95)] = 2
    t[on & (ny > 0.3) & (ny < 0.95)] = 1
    t[on & (np.abs(ny) <= 0.3)] = 3
    t[on & (ny < -0.3)] = 4
    return t


def item_type(types: np.ndarray, cells) -> int:
    """The commonest surface type (1..4) among a part's cells, interior (0) when none is on the surface."""
    c = np.bincount([int(types[y, x, z]) for y, x, z in cells], minlength=len(TYPES))
    return int(np.argmax(c[1:]) + 1) if c[1:].any() else 0


def real_mix(kind: str | None = "building") -> np.ndarray | None:
    """(len(TYPES), len(CLASSES)) P(class | type) in real sets (+1 smoothing), or None when not mined."""
    if not DATA.exists():
        return None
    d = json.loads(DATA.read_text())
    if kind is not None and d.get("kind") != kind:
        return None
    c = np.array(d["counts"], float) + 1.0
    return c / c.sum(1, keepdims=True)


def all_mix(kind: str | None = "building") -> np.ndarray:
    """(len(CLASSES),) P(class) over all parts of real sets (+1 smoothing)."""
    c = np.array(json.loads(DATA.read_text())["counts"], float).sum(0) + 1.0
    return c / c.sum()


def roof_mix(kind: str | None = "building") -> np.ndarray | None:
    """(len(CLASSES),) P(class) on real sets' pitched roofs (tools/learn/roof_mix.py), or None when not mined."""
    f = DATA.with_name("roof_mix.json")
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    if kind is not None and d.get("kind") != kind:
        return None
    c = np.array(d["counts"], float) + 1.0
    return c / c.sum()


def _tops(occ: np.ndarray):
    """Per stud column: filled, the top cell's height, its 3x3 median, and the median's slope (plates per stud)."""
    from scipy.ndimage import median_filter
    H = occ.shape[0]
    filled = occ.any(0)
    top = np.where(filled, H - 1 - np.argmax(occ[::-1], axis=0), -1).astype(float)
    sm = median_filter(np.where(filled, top, np.nan_to_num(top)), size=3)
    gy, gx = np.gradient(sm)
    return filled, top, sm, np.hypot(gy, gx)


def pitched_columns(occ: np.ndarray, lo: float = 0.9, hi: float = 4.3, min_cols: int = 30) -> np.ndarray:
    """(X, Z) bool: stud columns under a pitched roof -- the largest connected area where the top of the model rises
    steadily, lo..hi plates per stud (20-60 deg), when it covers at least `min_cols` columns (else none)."""
    from scipy.ndimage import label, median_filter
    filled, top, sm, g = _tops(occ)
    near = median_filter(filled.astype(np.uint8), size=3).astype(bool)
    jump = np.abs(top - sm) > 3                                      # a wall's edge, not a slope
    pitched = filled & near & ~jump & (g >= lo) & (g <= hi)
    lab, n = label(pitched)
    if n == 0:
        return np.zeros_like(filled)
    sizes = np.bincount(lab.ravel())[1:]
    keep = int(np.argmax(sizes)) + 1
    return (lab == keep) if sizes.max() >= min_cols else np.zeros_like(filled)


def roof_cells(occ: np.ndarray, depth: int = 4) -> np.ndarray:
    """(H, X, Z) bool: the top `depth` plates of each pitched-roof column (pitched_columns)."""
    cols = pitched_columns(occ)
    H = occ.shape[0]
    top = H - 1 - np.argmax(occ[::-1], axis=0)
    y = np.arange(H)[:, None, None]
    return cols[None] & (y <= top[None]) & (y > top[None] - depth)


def flat_cells(occ: np.ndarray, depth: int = 2, max_rise: float = 4.3) -> np.ndarray:
    """(H, X, Z) bool: the top `depth` plates of each column whose top is not under the pitched roof and not a
    wall's edge (rise at most `max_rise` plates per stud, a pitched roof's upper bound): flat roofs, ledges, and the
    ground or base around a building, level or sloping."""
    filled, top, sm, g = _tops(occ)
    cols = filled & (g <= max_rise) & (np.abs(top - sm) <= 1) & ~pitched_columns(occ)
    t = top.astype(int)
    y = np.arange(occ.shape[0])[:, None, None]
    return cols[None] & (y <= t[None]) & (y > t[None] - depth)


def flat_mix(kind: str | None = "building") -> np.ndarray | None:
    """(len(CLASSES),) P(class) on real sets' flat tops (flat_cells; tools/learn/roof_mix.py), or None when not mined."""
    f = DATA.with_name("roof_mix.json")
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    if (kind is not None and d.get("kind") != kind) or "flat_counts" not in d:
        return None
    c = np.array(d["flat_counts"], float) + 1.0
    return c / c.sum()
