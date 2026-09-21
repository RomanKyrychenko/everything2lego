"""LEGO part catalog loader + geometry constants."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files


# LDraw unit conversions.
LDU_PER_STUD = 20      # 1 stud footprint = 20 LDU
LDU_PER_PLATE = 8      # 1 plate height = 8 LDU (brick = 3 plates = 24 LDU)

# Real millimeter equivalents (for mesh scaling).
MM_PER_STUD = 8.0
MM_PER_PLATE = 3.2


@dataclass(frozen=True)
class Part:
    ldraw: str             # e.g. "3001.dat"
    bl: str                # Bricklink part id
    shape: str             # brick | plate | tile | slope | slope_inv
    footprint: tuple[int, int]   # (x_studs, z_studs) with x >= z conventionally
    height_units: int      # in plate-heights (brick=3, plate=1)
    name: str
    slope_dir: str | None = None  # for slopes: default low direction ("+x", "-x", "+z", "-z")

    @property
    def w(self) -> int:
        return self.footprint[0]

    @property
    def d(self) -> int:
        return self.footprint[1]

    @property
    def is_slope(self) -> bool:
        return self.shape in ("slope", "slope_inv")

    def orientations(self) -> list[tuple[int, int]]:
        """Distinct (w,d) footprints under 90-degree rotation.

        Rectangular bricks look identical after a 180-degree rotation, so we
        only enumerate two orientations. Slopes are asymmetric — the direction
        the low side points depends on rotation — but their *footprint* still
        only takes two values, and the refinement pass chooses rotation
        directly, so this method still returns two entries for them.
        """
        w, d = self.footprint
        if w == d:
            return [(w, d)]
        return [(w, d), (d, w)]


def _make_part(p: dict) -> Part:
    return Part(
        ldraw=p["ldraw"],
        bl=p["bl"],
        shape=p["shape"],
        footprint=tuple(p["footprint"]),
        height_units=p["height_units"],
        name=p["name"],
        slope_dir=p.get("slope_dir"),
    )


@lru_cache(maxsize=1)
def catalog() -> tuple[Part, ...]:
    """The union of the curated catalog and any Rebrickable-imported parts.

    Dedupe key is the LDraw filename so a curated entry always wins over
    an imported one (curated entries have hand-tuned `slope_dir` etc.).
    """
    data_pkg = files("everything2lego.data")
    raw = json.loads(data_pkg.joinpath("brick_catalog.json").read_text())
    parts: list[Part] = [_make_part(p) for p in raw["parts"]]
    seen = {p.ldraw for p in parts}
    imported_path = data_pkg.joinpath("imported_parts.json")
    try:
        imported = json.loads(imported_path.read_text())
    except FileNotFoundError:
        imported = None
    if imported:
        for p in imported["parts"]:
            if p["ldraw"] in seen:
                continue
            parts.append(_make_part(p))
            seen.add(p["ldraw"])
    return tuple(parts)


@lru_cache(maxsize=None)
def parts_by_shape(shape: str) -> tuple[Part, ...]:
    return tuple(p for p in catalog() if p.shape == shape)


@lru_cache(maxsize=None)
def parts_by_height(height_units: int, *, exclude_tiles: bool = True) -> tuple[Part, ...]:
    return tuple(
        p for p in catalog()
        if p.height_units == height_units and (not exclude_tiles or p.shape != "tile")
    )


def largest_first(parts: tuple[Part, ...]) -> tuple[Part, ...]:
    """Sort parts by descending stud area, so a greedy fill picks the biggest first."""
    return tuple(sorted(parts, key=lambda p: (-p.w * p.d, -max(p.w, p.d))))
