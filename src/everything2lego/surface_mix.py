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
