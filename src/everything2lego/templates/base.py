"""Template protocol + shared helpers."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Protocol

from ..bricks import Part, catalog
from ..colors import LegoColor, palette
from ..pack import Placement


class Template(Protocol):
    """Protocol every template must satisfy.

    Concrete templates subclass this and override `Params` + `build()`.
    """

    name: str
    """Short kebab-case name used at the CLI (e.g. "gable-cabin")."""

    @dataclass
    class Params:
        """Per-template parameters. Subclasses replace with their own."""

    def build(self, params) -> list[Placement]:
        """Deterministic: same `params` → same placements."""
        ...


def register(registry: dict[str, type[Template]], cls: type[Template]) -> None:
    """Add `cls` to `registry` under its `.name`."""
    registry[cls.name] = cls


# ------------------------- shared building helpers -------------------------

def color_index_by_code(code: int) -> int:
    """Palette index for the given LDraw color code. Falls back to Black (0)."""
    pal = palette()
    for i, c in enumerate(pal):
        if c.code == code:
            return i
    return 0


def part_by_ldraw(ldraw: str) -> Part | None:
    """Look up a catalog entry by LDraw filename."""
    for p in catalog():
        if p.ldraw == ldraw:
            return p
    return None


def place_rect(part: Part, *, x: int, z: int, y: int,
                color_idx: int, rotation_deg: int = 0) -> Placement:
    """Construct a Placement for a rectangular part at (x, y, z)."""
    if rotation_deg in (0, 180):
        w, d = part.footprint
    else:
        w, d = part.footprint[1], part.footprint[0]
    return Placement(
        part=part, color_idx=color_idx,
        x_stud=x, z_stud=z, y_plate=y,
        w_world=w, d_world=d, rotation_deg=rotation_deg,
    )


def tile_area_with_plates(x0: int, z0: int, w: int, d: int, *,
                           y: int, color_idx: int,
                           preferred_footprints: list[tuple[int, int]] | None = None
                           ) -> list[Placement]:
    """Cover a rectangular area (starting at x0,z0, size w×d) at plate layer y
    using the largest available plate that fits, iteratively. Falls back to
    2x2 and 1x1 plates for leftover cells."""
    if preferred_footprints is None:
        # Largest area first so we tile with big plates when possible.
        preferred_footprints = [
            (8, 8), (8, 6), (10, 6), (12, 6), (16, 2), (12, 2), (10, 2),
            (8, 2), (8, 4), (10, 4), (12, 4), (6, 6), (6, 4), (4, 4),
            (8, 1), (6, 2), (6, 1), (4, 2), (4, 1), (3, 2), (3, 1),
            (2, 2), (2, 1), (1, 1),
        ]

    # Map footprint -> Part, considering rotated variants.
    lookup: dict[tuple[int, int], Part] = {}
    for p in catalog():
        if p.shape != "plate":
            continue
        for w_p, d_p in p.orientations():
            lookup.setdefault((w_p, d_p), p)

    filled = [[False] * d for _ in range(w)]
    placements: list[Placement] = []
    for x in range(w):
        for z in range(d):
            if filled[x][z]:
                continue
            for pw, pd in preferred_footprints:
                if x + pw > w or z + pd > d:
                    continue
                if any(filled[x + dx][z + dz]
                       for dx in range(pw) for dz in range(pd)):
                    continue
                part = lookup.get((pw, pd))
                if part is None:
                    continue
                rot = 0 if (pw, pd) == part.footprint else 90
                placements.append(place_rect(
                    part, x=x0 + x, z=z0 + z, y=y,
                    color_idx=color_idx, rotation_deg=rot,
                ))
                for dx in range(pw):
                    for dz in range(pd):
                        filled[x + dx][z + dz] = True
                break
    return placements


def tile_area_with_tiles(x0: int, z0: int, w: int, d: int, *,
                          y: int, color_idx: int) -> list[Placement]:
    """Cover a rectangular area with tiles (studless) for smooth surfaces
    like roof caps, cornices, and floors. Falls back to progressively
    smaller tiles down to 1x1."""
    preferred = [
        (8, 1), (6, 1), (4, 1), (3, 1), (2, 1),
        (4, 2), (6, 2), (2, 2), (1, 1),
    ]
    lookup: dict[tuple[int, int], Part] = {}
    for p in catalog():
        if p.shape != "tile":
            continue
        for w_p, d_p in p.orientations():
            lookup.setdefault((w_p, d_p), p)

    filled = [[False] * d for _ in range(w)]
    placements: list[Placement] = []
    for x in range(w):
        for z in range(d):
            if filled[x][z]:
                continue
            for pw, pd in preferred:
                if x + pw > w or z + pd > d:
                    continue
                if any(filled[x + dx][z + dz]
                       for dx in range(pw) for dz in range(pd)):
                    continue
                part = lookup.get((pw, pd))
                if part is None:
                    continue
                rot = 0 if (pw, pd) == part.footprint else 90
                placements.append(place_rect(
                    part, x=x0 + x, z=z0 + z, y=y,
                    color_idx=color_idx, rotation_deg=rot,
                ))
                for dx in range(pw):
                    for dz in range(pd):
                        filled[x + dx][z + dz] = True
                break
    return placements
