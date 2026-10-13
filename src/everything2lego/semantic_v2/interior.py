"""InteriorVolumeFiller — fills the mesh's interior mass with cheap bricks.

Surface builders emit visible skin.  The interior of the mesh (cells that
are inside the voxel occupancy volume AND not covered by any surface
placement) needs to be filled so the model is structurally solid.

Interior is invisible, so we use a single structural colour and the biggest-
first plate/brick greedy — no colour or texture concerns.

This is the CLEAN SEPARATION the v2 architecture makes explicit:  surface
builders care about appearance; the InteriorVolumeFiller cares about mass
and buildability.  Two orthogonal stages instead of one confused loss.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from ..bricks import Part, catalog, MM_PER_STUD, MM_PER_PLATE
from ..colors import palette
from ..pack import Placement
from ..templates.base import part_by_ldraw, place_rect
from .mesh_target import MeshTarget


@dataclass
class InteriorConfig:
    structural_color_code: int = 71   # Light Bluish Gray (cheapest bulk part)
    # Hollowing.  `hollow=False` = solid interior (old behaviour).
    # `hollow=True` = keep only a load-bearing skeleton:
    #   - a `shell_thickness`-thick shell just below the visible surface
    #   - a floor slab every `floor_spacing` plates (real building floors)
    #   - a column at every `(column_spacing, column_spacing)` (x, z) node
    #   - a solid foundation layer at y = 0
    # Cuts interior part count by 70-80% on large models while keeping
    # them physically buildable.
    hollow: bool = False
    shell_thickness: int = 2
    floor_spacing: int = 12       # ~4 bricks — matches real building floors
    column_spacing: int = 8       # ~one 8-stud brick between columns
    foundation_plates: int = 3    # 1 brick of solid base


class InteriorVolumeFiller:
    """Fills the residual interior volume of the mesh with structural bricks.

    Called AFTER surface builders have placed their parts.  Uses the mesh's
    voxel grid as the volume envelope, subtracts cells already covered by
    surface placements, and fills the remainder with big-first bricks.
    """

    def __init__(self, config: InteriorConfig | None = None) -> None:
        self.config = config or InteriorConfig()

    def fill(self, target: MeshTarget,
              surface_placements: Iterable[Placement],
              exclude: np.ndarray | None = None) -> list[Placement]:
        """`exclude` marks cells some other stage owns outright (a recipe's
        roof prism, say): they are never filled, even where the voxel
        envelope says the mesh was there and nothing was built."""
        occ = target.voxel_grid.occupied()
        Y, X, Z = occ.shape
        # Mark cells already covered by surface parts.
        covered = _mark_covered(occ.shape, surface_placements)
        remaining = occ & ~covered
        if exclude is not None:
            remaining &= ~exclude
        if self.config.hollow:
            remaining = _hollow_with_supports(
                remaining, occ,
                shell_thickness=self.config.shell_thickness,
                floor_spacing=self.config.floor_spacing,
                column_spacing=self.config.column_spacing,
                foundation_plates=self.config.foundation_plates,
            )
        if not remaining.any():
            return []

        color_idx = _palette_index_for_code(self.config.structural_color_code)

        out: list[Placement] = []
        # 3-plate brick promotion first — biggest bricks land in interior
        # slabs where 3 consecutive Y cells are all `remaining`.
        out.extend(_promote_interior_bricks(remaining, color_idx))
        # Whatever is left after brick promotion gets tiled per y-level
        # with the biggest available plates.
        out.extend(_pack_interior_plates(remaining, color_idx))
        return out


# Footprint (studs per side) of a hollow-mode support column.
_COLUMN_STUDS = 2
# Height of a standard brick in plates; interior courses are this tall.
_COURSE_PLATES = 3


def _hollow_with_supports(remaining: np.ndarray, occ: np.ndarray,
                            *, shell_thickness: int,
                            floor_spacing: int, column_spacing: int,
                            foundation_plates: int) -> np.ndarray:
    """Return `remaining` with the interior mass carved away except for
    a structural skeleton: shell + periodic floors + periodic columns +
    solid foundation slab.  The output is a strict SUBSET of the input.
    """
    Y, X, Z = occ.shape
    try:
        from scipy.ndimage import binary_erosion
    except ImportError:
        return remaining  # scipy unavailable — leave interior solid
    # Shell = cells that are occupied but NOT inside the eroded volume.
    eroded = binary_erosion(occ, iterations=max(1, shell_thickness))
    shell = occ & ~eroded
    keep = shell & remaining
    # Solid foundation for the bottom `foundation_plates` layers.
    if foundation_plates > 0:
        y_top = min(Y, max(1, foundation_plates))
        keep[:y_top] |= remaining[:y_top]
    # Floor slabs every `floor_spacing` plates.  A slab is a full Y-plane
    # intersected with occ (so it doesn't extend past the mesh).
    if floor_spacing > 0:
        for y in range(0, Y, max(1, floor_spacing)):
            keep[y] |= remaining[y]
    # Column supports at each (x, z) modulo `column_spacing`.  A column
    # is a 2x2-stud vertical strip through occ wherever the grid has room
    # (clipped to 2x1 / 1x1 at the far edge), so each course of the column
    # has clutch on more than one stud and the 2x2x5 tall brick fits it.
    if column_spacing > 0:
        for x in range(0, X, max(1, column_spacing)):
            for z in range(0, Z, max(1, column_spacing)):
                col = np.s_[:, x:x + _COLUMN_STUDS, z:z + _COLUMN_STUDS]
                keep[col] |= remaining[col]
    return keep


# ---------------------------------------------------------------------------
# Coverage tracking
# ---------------------------------------------------------------------------

def _mark_covered(shape_yxz: tuple[int, int, int],
                    placements: Iterable[Placement]) -> np.ndarray:
    """Mask of grid cells that are already covered by these placements."""
    Y, X, Z = shape_yxz
    covered = np.zeros((Y, X, Z), dtype=bool)
    for p in placements:
        h = p.part.height_units
        for dy in range(h):
            yi = p.y_plate + dy
            if yi < 0 or yi >= Y:
                continue
            for dx in range(p.w_world):
                xi = p.x_stud + dx
                if xi < 0 or xi >= X:
                    continue
                for dz in range(p.d_world):
                    zi = p.z_stud + dz
                    if zi < 0 or zi >= Z:
                        continue
                    covered[yi, xi, zi] = True
    return covered


# ---------------------------------------------------------------------------
# Brick promotion (3-plate stacks)
# ---------------------------------------------------------------------------

# Interior structural bricks.  brick_tall variants (2454, 30360) are 15
# plates tall = 5 real bricks in a single part — massive part-count
# reduction on load-bearing columns.
_INTERIOR_TALL_BRICKS = [
    ("30360.dat", (2, 2), 15),   # Brick 2x2x5 = 15 plates tall
    ("2454.dat",  (2, 1), 15),   # Brick 1x2x5
]
# Standard 3-plate bricks, ordered by footprint AREA descending so the
# greedy really is biggest-first.  Ties go to the 2-wide brick, which has
# clutch across two rows of studs.
_INTERIOR_BRICKS = [
    ("3006.dat", (10, 2)),   # Brick 2x10  (20)
    ("3007.dat", (8, 2)),    # Brick 2x8   (16)
    ("2465.dat", (16, 1)),   # Brick 1x16  (16)
    ("2456.dat", (6, 2)),    # Brick 2x6   (12)
    ("6112.dat", (12, 1)),   # Brick 1x12  (12)
    ("3001.dat", (4, 2)),    # Brick 2x4   (8)
    ("3008.dat", (8, 1)),    # Brick 1x8   (8)
    ("3009.dat", (6, 1)),    # Brick 1x6   (6)
    ("3003.dat", (2, 2)),    # Brick 2x2   (4)
    ("3010.dat", (4, 1)),    # Brick 1x4   (4)
    ("3622.dat", (3, 1)),    # Brick 1x3   (3)
    ("3004.dat", (2, 1)),    # Brick 1x2   (2)
    ("3005.dat", (1, 1)),    # Brick 1x1   (1)
]


def _course_parity(y: int) -> int:
    """0 for even courses, 1 for odd — course index counted in bricks from
    the foundation (y = 0)."""
    return (y // _COURSE_PLATES) % 2


def _staggered_scan(parity: int, orientations: list[tuple[int, int, int]],
                     X: int, Z: int):
    """Yield (pw, pd, rot, xs, zs) for one course.

    Odd courses reverse the orientation order (long axis along z instead
    of x) and roll the scan start by one stud, the same running-bond trick
    `pack.py` uses, so brick seams on consecutive courses don't coincide
    and the fill is bonded instead of a set of independent stacks.
    """
    orients = list(orientations)
    if parity:
        orients.reverse()
    for pw, pd, rot in orients:
        xs = list(range(X - pw + 1))
        zs = list(range(Z - pd + 1))
        if parity:
            xs = xs[1:] + xs[:1]
            zs = zs[1:] + zs[:1]
        yield pw, pd, rot, xs, zs


def _promote_interior_bricks(remaining: np.ndarray, color_idx: int
                               ) -> list[Placement]:
    """Biggest-first brick promotion with a running bond between courses.

    Two passes:
      1. brick_tall (15 plates tall) — load-bearing columns.  Only placed
         on course boundaries (y % 3 == 0 from the foundation) so their
         top seam lines up with the regular courses around them.
      2. brick (3 plates tall) — standard walls, staggered per course.
    """
    Y, X, Z = remaining.shape
    processed = np.zeros_like(remaining)
    out: list[Placement] = []

    def _try_promote(ldraw: str, fw: int, fd: int, height: int,
                     *, course_aligned: bool) -> None:
        part = part_by_ldraw(ldraw)
        if part is None:
            return
        orientations = [(fw, fd, 0)]
        if fw != fd:
            orientations.append((fd, fw, 90))
        for y in range(Y - height + 1):
            if course_aligned and y % _COURSE_PLATES:
                continue
            if not remaining[y:y + height].any():
                continue
            for pw, pd, rot, xs, zs in _staggered_scan(
                    _course_parity(y), orientations, X, Z):
                for x in xs:
                    for z in zs:
                        slab_slice = np.s_[y:y + height, x:x + pw, z:z + pd]
                        if processed[slab_slice].any():
                            continue
                        if not remaining[slab_slice].all():
                            continue
                        out.append(place_rect(part, x=x, z=z, y=y,
                                                color_idx=color_idx,
                                                rotation_deg=rot))
                        processed[slab_slice] = True

    # PASS 1: 15-plate-tall bricks (5-brick columns in one part).
    for ldraw, (fw, fd), height in _INTERIOR_TALL_BRICKS:
        _try_promote(ldraw, fw, fd, height, course_aligned=True)
    # PASS 2: standard 3-plate-tall bricks.
    for ldraw, (fw, fd) in _INTERIOR_BRICKS:
        _try_promote(ldraw, fw, fd, _COURSE_PLATES, course_aligned=False)

    # Update remaining in place for the plate pass.
    remaining &= ~processed
    return out


# ---------------------------------------------------------------------------
# Plate packing (per-y remainder)
# ---------------------------------------------------------------------------

_INTERIOR_PLATES: list[tuple[int, int]] = []


def _refresh_plate_list() -> None:
    if _INTERIOR_PLATES:
        return
    sizes: set[tuple[int, int]] = set()
    for p in catalog():
        if p.shape == "plate":
            for w, d in p.orientations():
                sizes.add((w, d))
    _INTERIOR_PLATES.extend(sorted(sizes, key=lambda wd: (-wd[0] * wd[1], -max(wd))))


def _plate_lookup() -> dict[tuple[int, int], Part]:
    out: dict[tuple[int, int], Part] = {}
    for p in catalog():
        if p.shape != "plate":
            continue
        for w, d in p.orientations():
            out.setdefault((w, d), p)
    return out


def _pack_interior_plates(remaining: np.ndarray, color_idx: int
                            ) -> list[Placement]:
    _refresh_plate_list()
    lookup = _plate_lookup()
    Y, X, Z = remaining.shape
    out: list[Placement] = []
    for y in range(Y):
        layer = remaining[y]
        if not layer.any():
            continue
        filled = ~layer.copy()
        # Plates alternate scan start per layer too, so a stack of plate
        # layers is bonded rather than a column of identical tilings.
        parity = y % 2
        for pw, pd in _INTERIOR_PLATES:
            part = lookup.get((pw, pd))
            if part is None:
                continue
            rot = 0 if (pw, pd) == part.footprint else 90
            xs = list(range(X - pw + 1))
            zs = list(range(Z - pd + 1))
            if parity:
                xs = xs[1:] + xs[:1]
                zs = zs[1:] + zs[:1]
            for x in xs:
                for z in zs:
                    if filled[x, z]:
                        continue
                    if filled[x:x + pw, z:z + pd].any():
                        continue
                    out.append(place_rect(part, x=x, z=z, y=y,
                                            color_idx=color_idx,
                                            rotation_deg=rot))
                    filled[x:x + pw, z:z + pd] = True
    return out


def _palette_index_for_code(code: int) -> int:
    pal = palette()
    for i, c in enumerate(pal):
        if c.code == code:
            return i
    warnings.warn(f"LDraw colour {code} is not in the palette; using {pal[0].name}",
                  RuntimeWarning, stacklevel=2)
    return 0
