"""Greedy layer-by-layer brick packer.

Given a colored voxel grid (`voxelize.VoxelGrid`), place real LDraw parts so that
every occupied voxel is claimed by exactly one part. Bricks preferred over
stacked plates when a 3-plate slab of the same color exists; leftover voxels
become plates. Tiles are used for exposed top surfaces if requested.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .bricks import Part, catalog, largest_first, parts_by_shape, full_catalog
from .voxelize import VoxelGrid


@dataclass(frozen=True)
class Placement:
    part: Part
    color_idx: int      # index into colors.palette()
    x_stud: int
    z_stud: int
    y_plate: int
    w_world: int        # footprint width after rotation (along +x)
    d_world: int        # footprint depth after rotation (along +z)
    rotation_deg: int   # 0, 90, 180 or 270 (bricks use 0/90; slopes use all four)
    # A window frame's glass, clipped into the frame as one unit: the LDraw
    # colour code of the glass part (bricks.GLASS), or None.  It shares the
    # frame's cells, so the structure never sees it; the exporters write it.
    glass: int | None = None


def _try_fit(voxels: np.ndarray, filled: np.ndarray,
             y: int, x: int, z: int, w: int, hy: int, d: int) -> int | None:
    """Return the color index of a (w, hy, d) brick starting at (y, x, z),
    or None if it doesn't fit uniformly."""
    H, W, D = voxels.shape
    if y + hy > H or x + w > W or z + d > D or y < 0 or x < 0 or z < 0:
        return None
    sub_v = voxels[y:y + hy, x:x + w, z:z + d]
    sub_f = filled[y:y + hy, x:x + w, z:z + d]
    if (sub_v < 0).any() or sub_f.any():
        return None
    c = int(sub_v.flat[0])
    if not np.all(sub_v == c):
        return None
    return c


# Footprints the strict pass will allow at top-of-column, in two rings:
#   - COMPACT: 2x1/1x2/2x2. These are the sizes that reliably slot into edge
#     and corner step-downs during refinement. Tried first so corners aren't
#     eaten by a wider brick that spans multiple step directions.
#   - WIDE: larger slope footprints. Tried after compact, for long single-
#     direction step edges.
_SLOPEABLE_FOOTPRINTS_COMPACT = frozenset({(2, 1), (1, 2), (2, 2)})
_SLOPEABLE_FOOTPRINTS_WIDE = frozenset({(3, 2), (2, 3), (4, 2), (2, 4)})


def _level_consistent(col_top: np.ndarray, top_y: int,
                       x: int, z: int, w: int, d: int,
                       *, height_units: int = 3,
                       top_footprints: frozenset[tuple[int, int]] | None = None) -> bool:
    """A placement is level-consistent if its footprint is either uniformly
    covered by upper bricks (an interior fill) or uniformly at column top
    with a footprint in `top_footprints`. Anything else straddles a step
    edge and is deferred to the relaxed pass."""
    window = col_top[x:x + w, z:z + d]
    at_top = bool((window == top_y).any())
    covered = bool((window > top_y).any())
    if at_top and covered:
        return False
    if at_top and not covered:
        if height_units != 3:
            return False
        if top_footprints is None:
            return True
        return (w, d) in top_footprints
    return True


def _windows(a: np.ndarray, w: int, d: int) -> np.ndarray:
    """Every (w, d) window of a 2-D array, as an array of shape (W-w+1, D-d+1, w, d)."""
    return sliding_window_view(a, (w, d))


def _candidate_starts(voxels: np.ndarray, filled: np.ndarray, y: int, hy: int,
                      w: int, d: int) -> np.ndarray:
    """(W-w+1, D-d+1) mask of starts where a (w, hy, d) block is unfilled and
    one colour.  Computed once per part and layer; `filled` only grows, so
    a start that fails here can never fit later and the ones that pass are
    confirmed by `_try_fit` in scan order."""
    sub = voxels[y:y + hy]
    free = ~filled[y:y + hy].any(axis=0)
    column_uniform = (sub == sub[:1]).all(axis=0) & (sub[0] >= 0)
    base = _windows(sub[0], w, d)
    return (_windows(free & column_uniform, w, d).all(axis=(2, 3))
            & (base.min(axis=(2, 3)) == base.max(axis=(2, 3))))


def _consistent_starts(col_top: np.ndarray, top_y: int, w: int, d: int,
                       hy: int, top_footprints: frozenset[tuple[int, int]] | None
                       ) -> np.ndarray:
    """`_level_consistent` for every start at once."""
    windows = _windows(col_top, w, d)
    at_top = (windows == top_y).any(axis=(2, 3))
    covered = (windows > top_y).any(axis=(2, 3))
    top_ok = hy == 3 and (top_footprints is None or (w, d) in top_footprints)
    bad = at_top & covered
    if not top_ok:
        bad |= at_top & ~covered
    return ~bad


def _place_parts(voxels: np.ndarray, filled: np.ndarray,
                 parts: tuple[Part, ...], hy: int,
                 y_range: list[int] | range,
                 support: np.ndarray | None = None,
                 col_top: np.ndarray | None = None,
                 strict_level: bool = False,
                 top_footprints: frozenset[tuple[int, int]] | None = None) -> list[Placement]:
    """Greedy pass: for each part (largest first) and each orientation, walk
    the scan order and place wherever it fits.

    Two parts with the same footprint see the same grid, so the second can
    place nothing the first did not; each (w, d) is walked once per layer.
    """
    placements: list[Placement] = []
    H, W, D = voxels.shape
    for y in y_range:
        if y + hy > H:
            continue
        top_y = y + hy - 1
        tried: set[tuple[int, int]] = set()
        for part in largest_first(parts):
            orientations = part.orientations()
            if y % 2:
                orientations.reverse()
            for w, d in orientations:
                if (w, d) in tried or w > W or d > D:
                    continue
                tried.add((w, d))
                rot = 0 if (w, d) == part.footprint else 90
                starts = _candidate_starts(voxels, filled, y, hy, w, d)
                if support is not None:
                    starts &= _windows(support, w, d).any(axis=(2, 3))
                if strict_level and col_top is not None:
                    starts &= _consistent_starts(col_top, top_y, w, d, hy, top_footprints)
                if not starts.any():
                    continue
                xs = np.arange(W - w + 1)
                zs = np.arange(D - d + 1)
                if y % 2:
                    xs = np.roll(xs, -1)
                    zs = np.roll(zs, -1)
                for x in xs[starts[xs].any(axis=1)]:
                    for z in zs[starts[x, zs]]:
                        c = _try_fit(voxels, filled, y, x, z, w, hy, d)
                        if c is None:
                            continue
                        filled[y:y + hy, x:x + w, z:z + d] = True
                        placements.append(Placement(
                            part=part, color_idx=c,
                            x_stud=int(x), z_stud=int(z), y_plate=y,
                            w_world=w, d_world=d, rotation_deg=rot,
                        ))
    return placements


def pack(grid: VoxelGrid, *, use_tiles: bool = True, use_slopes: bool = True,
         foundation_layers: int = 0, use_tall: bool = True,
         use_cheese: bool = True) -> list[Placement]:
    """Pack a voxel grid into real LEGO parts.

    Placement strategy:
      1. Bottom-up courses: aligned bricks every three plate layers, preferring
         placements that attach to parts below. Foundation courses use plates.
      2. Fill remaining voxels with plates; alternate course directions and seams.
      3. Tile pass — if `use_tiles`, convert top-exposed 1-plate placements
         into tiles where a same-shape tile exists.
      4. Slope pass — if `use_slopes`, convert top-of-column bricks at step-down
         edges into equivalent slope parts. Volume-preserving: same (x, z, y),
         same voxels claimed, only the .dat and rotation change.
    """
    voxels = grid.palette_idx
    H, W, D = voxels.shape
    filled = np.zeros_like(voxels, dtype=bool)
    filled[voxels < 0] = True  # empty cells are "already accounted for"

    brick_parts = parts_by_shape("brick")
    plate_parts = parts_by_shape("plate")

    # Column-top map; used to bias the packer against bricks that straddle step
    # edges, so the slope refinement pass has narrow candidates to work with.
    col_top = _column_tops(voxels) if use_slopes else None

    placements: list[Placement] = []

    # Multi-height pre-pass: for solid tall columns of uniform color, try the
    # tallest wall/panel parts first (Brick 1x2x5, Panel 1x2x2, Panel 1x4x3
    # etc). One 15-plate tall brick replaces five 3-plate bricks.
    #
    # Restricted to INTERIOR positions (col_top above the placement) so we
    # never eat the top-of-column bricks the slope refinement wants. Aligned
    # to brick-course positions so seams line up with the regular pack.
    if use_tall and col_top is not None:
        tall_parts = parts_by_shape("brick_tall") + parts_by_shape("panel")
        heights_tall = sorted({p.height_units for p in tall_parts}, reverse=True)
        for hy in heights_tall:
            if hy <= 3:
                continue  # panels-at-brick-height are handled with regular bricks
            parts_at_h = tuple(p for p in tall_parts if p.height_units == hy)
            for y in range(foundation_layers, H - hy + 1, 3):
                # strict_level with top_footprints=empty means "interior only":
                # _level_consistent returns True only when covered by upper
                # bricks. Prevents tall parts from stealing surface cells the
                # slope refinement wants.
                placements += _place_parts(
                    voxels, filled, parts_at_h, hy=hy, y_range=[y],
                    col_top=col_top, strict_level=True,
                    top_footprints=frozenset(),
                )

    tops = np.zeros_like(filled)
    for y in range(H):
        support = tops[y] if y > 0 else None
        current = []
        brick_course = y >= foundation_layers and (y - foundation_layers) % 3 == 0
        # Strict pass, two sub-phases so corners get 2x2 candidates before
        # long edges get 2x4 candidates.
        if use_slopes and brick_course:
            for tf in (_SLOPEABLE_FOOTPRINTS_COMPACT, _SLOPEABLE_FOOTPRINTS_WIDE):
                current += _place_parts(voxels, filled, brick_parts, hy=3, y_range=[y],
                                         support=support, col_top=col_top,
                                         strict_level=True, top_footprints=tf)
        if use_slopes:
            current += _place_parts(voxels, filled, plate_parts, hy=1, y_range=[y],
                                     support=support, col_top=col_top, strict_level=True)
        if brick_course:
            current += _place_parts(voxels, filled, brick_parts, hy=3, y_range=[y], support=support)
        current += _place_parts(voxels, filled, plate_parts, hy=1, y_range=[y], support=support)
        # Preserve geometry even when it cannot attach yet. The pipeline can
        # add support columns for these remaining placements and repack.
        if brick_course:
            current += _place_parts(voxels, filled, brick_parts, hy=3, y_range=[y])
        current += _place_parts(voxels, filled, plate_parts, hy=1, y_range=[y])
        for p in current:
            top = p.y_plate + p.part.height_units
            if top < H:
                tops[top, p.x_stud:p.x_stud + p.w_world, p.z_stud:p.z_stud + p.d_world] = True
        placements.extend(current)

    # Anything left? Fill with 1x1 plates as a last resort (shouldn't happen if
    # 1x1 plate is in the catalog, but defensive).
    leftover = np.argwhere(~filled)
    if len(leftover) > 0:
        raise RuntimeError(
            f"{len(leftover)} voxels could not be filled — is Plate 1x1 missing?"
        )

    if use_tiles:
        placements = _tile_top(voxels, placements)
    if use_slopes:
        placements = _refine_slopes(voxels, placements, cheese=use_cheese)

    return placements


def _tile_top(voxels: np.ndarray, placements: list[Placement]) -> list[Placement]:
    """Swap top-exposed plates for equivalent tiles when a matching tile part
    exists in the catalog. Only applied to plates (height 1) that occupy the
    topmost plate-slot in their column-region."""
    tile_by_footprint: dict[tuple[int, int], Part] = {}
    for t in parts_by_shape("tile"):
        for w, d in t.orientations():
            tile_by_footprint.setdefault(tuple(sorted((w, d))), t)  # type: ignore[arg-type]

    result: list[Placement] = []
    for p in placements:
        if p.part.height_units != 1 or p.part.shape != "plate":
            result.append(p)
            continue
        # Check every cell in the footprint is top-exposed.
        y = p.y_plate
        exposed = all(
            (y + 1 >= voxels.shape[0]) or voxels[y + 1, p.x_stud + dx, p.z_stud + dz] < 0
            for dx in range(p.w_world) for dz in range(p.d_world)
        )
        key = tuple(sorted((p.w_world, p.d_world)))  # type: ignore[assignment]
        tile = tile_by_footprint.get(key)
        if exposed and tile is not None:
            # Preserve the orientation the plate had.
            rot = 0 if (p.w_world, p.d_world) == tile.footprint else 90
            result.append(Placement(
                part=tile, color_idx=p.color_idx,
                x_stud=p.x_stud, z_stud=p.z_stud, y_plate=p.y_plate,
                w_world=p.w_world, d_world=p.d_world, rotation_deg=rot,
            ))
        else:
            result.append(p)
    return result


# -- Slope refinement --------------------------------------------------------

import re  # noqa: E402  (slope-family name parsing)
from dataclasses import field  # noqa: E402
#
# Post-pack pass that turns the staircase a voxel roof packs into onto real
# roof parts.  A sloped brick claims the same voxel volume as the box it is
# cut from, so every replacement below is volume-preserving: the new part
# (plus the leftovers of any box part it was cut out of) covers exactly the
# cells the replaced parts covered.
#
# The local pitch is read off the column-top map: a step edge is a column at
# the top of a tread whose neighbour in one direction is `drop` plates lower,
# and the tread widths either side of the edge give rise over run.  Each
# (drop, tread) pair maps to a slope family:
#
#   drop 1-2 plates            -> cheese slopes (30°, 54200), 2 plates tall
#   drop 1 brick, tread 1      -> 45° slopes  (3040/3039/3038/3037...), 2 studs
#   drop 1 brick, tread >= 2   -> 33° slopes  (4286/3298/4161/3297/3939), 3 studs
#   drop 2 bricks, short tread -> 65° slopes  (60481/3678b), 2 studs, 2 bricks
#   drop 3 bricks, short tread -> 75° slopes  (3684), 2 studs, 3 bricks
#
# A slope's low strip must be at column top; the high strip may be covered
# by the next course up, which is how real slope roofs stack (each course
# sits on the full-height back stud of the course below).  Along the ridge
# direction, consecutive edge columns of one colour form a run that is tiled
# with the widest parts of the family (2x4 over two 2x2), seams alternating
# between courses.  Outside corners take the double-convex corner slope
# (3045), valleys the double-concave one (3046), and a 1- or 2-stud ridge
# between two opposite slopes is tiled instead of left as a studded brick.
#
# The rotation-to-low-direction cycle below is measured against `io_ldraw._ROT`;
# CYCLE[i] is the world direction the slope's low side points to when the part
# is rotated `i * 90` degrees, assuming the part's default low direction is +x.

_LOW_DIR_CYCLE = ("+x", "-z", "-x", "+z")

# Corner-slope low-corner cycle: the (+x, +z) corner at rotation 0, then
# rotating clockwise around +Y (in LDraw's Y-down convention). Indexed by
# rotation // 90. Matches how corner-slope 3045 rotates.
_CORNER_LOW_CYCLE = ("+x+z", "+x-z", "-x-z", "-x+z")

_DIR_VEC = {"+x": (1, 0), "-x": (-1, 0), "+z": (0, 1), "-z": (0, -1)}


def _column_tops(voxels: np.ndarray) -> np.ndarray:
    """(W, D) integer array: highest occupied y_plate for each column, -1 if empty."""
    _, W, D = voxels.shape
    occupied = voxels >= 0
    # top plate-level: argmax on reversed y axis gives the first True from the top.
    flipped = occupied[::-1]
    any_hit = flipped.any(axis=0)
    top_from_top = flipped.argmax(axis=0)
    top = (voxels.shape[0] - 1 - top_from_top).astype(np.int32)
    top[~any_hit] = -1
    return top


def slope_world_low(p: Placement) -> str | None:
    """World direction a placed slope's low side points to ("+x", "-z",
    "+x+z" for a corner...), from the part's default and the placement's
    rotation; None for anything that is not a slope."""
    part = p.part
    if part.shape not in ("slope", "slope_inv", "slope_corner"):
        return None
    steps = (p.rotation_deg // 90) % 4
    if part.shape == "slope_corner":
        default = part.slope_dir if part.slope_dir in _CORNER_LOW_CYCLE else "+x+z"
        return _CORNER_LOW_CYCLE[(_CORNER_LOW_CYCLE.index(default) + steps) % 4]
    default = part.slope_dir if part.slope_dir in _LOW_DIR_CYCLE else "+x"
    return _LOW_DIR_CYCLE[(_LOW_DIR_CYCLE.index(default) + steps) % 4]


def _slope_rotation(slope: Part, world_footprint: tuple[int, int],
                    low_dir: str) -> int | None:
    """Rotation (deg) that yields the requested world footprint AND world low
    direction, or None if the slope can't produce that combination.

    `low_dir` is one of "+x", "-x", "+z", "-z" for edge slopes, or a compound
    like "+x+z" for corner slopes.
    """
    declared = slope.footprint
    if declared[0] == declared[1] and declared == world_footprint:
        # Square footprint: all four rotations preserve the world footprint.
        candidates = (0, 90, 180, 270)
    elif declared == world_footprint:
        candidates = (0, 180)
    elif (declared[1], declared[0]) == world_footprint:
        candidates = (90, 270)
    else:
        return None
    if slope.shape == "slope_corner":
        default = slope.slope_dir or "+x+z"
        default_idx = _CORNER_LOW_CYCLE.index(default)
        cycle = _CORNER_LOW_CYCLE
    else:
        default = slope.slope_dir or "+x"
        default_idx = _LOW_DIR_CYCLE.index(default)
        cycle = _LOW_DIR_CYCLE
    for rot in candidates:
        world_low = cycle[(default_idx + rot // 90) % 4]
        if world_low == low_dir:
            return rot
    return None


def _slope_rank(part: Part) -> int:
    """Prefer canonical slopes over specialty variants (parts with "with",
    "Cutout", "Hollow", "Bottom Pin", "Stud Holder" in the name).

    0 = plain "Slope 45 N x M" or "Slope 45 N x M Inverted"
    1 = named-angle variants (30°/33°/65°/75°) or plain "Slope Curved"
    2 = anything with a bracketed suffix or slot-descriptor
    """
    name = part.name
    lname = name.lower()
    if any(t in lname for t in (
        "cutout", "hollow", "bottom pin", "stud holder",
        "inside stud", "grate", "ingot", "gold bar",
        "wedged, curved", "double / inverted",
        "double convex", "double concave",
    )):
        return 3
    if lname.startswith("slope 45 ") or lname.startswith("slope 45°"):
        return 0
    if "sloped 45" in lname or "slope 45" in lname:
        # "Brick Sloped 45° N x M" plain variant.
        return 1
    if "slope curved" in lname:
        return 1
    if "sloped 30" in lname or "sloped 33" in lname or "slope 33" in lname:
        return 2
    if "sloped 65" in lname or "sloped 75" in lname:
        return 2
    if "sloped 18" in lname:
        return 3
    return 2


# -- slope families ----------------------------------------------------------

# A plain wedge: "Slope 45 2 x 1", "Brick Sloped 33° 3 x 1", "Slope 75 2 x 2
# x 3", "Brick Sloped 30° 1 x 1 x 2/3 (Cheese Slope)".  Anything else (curved,
# double, inverted, with-studs...) is not a roof-face part.
_PLAIN_SLOPE_NAME = re.compile(
    r"^(?:Slope|Brick Sloped) (\d+)°? \d+ x \d+(?: x \d+(?:/\d+)?)?(?: \(Cheese Slope\))?$")


@dataclass(frozen=True)
class _SlopeFamily:
    """One pitch class of roof slopes.

    `run` is the footprint length along the slope (the part's declared x),
    `height` its height in plates, `drops` the column-top drop beyond the low
    edge that the part's face spans, `exposed` how many strips from the low
    edge must be at column top (a 33° slope slants over two studs, a 45° one
    over one).  `parts` maps ridge-direction width to the canonical part.
    """
    name: str
    run: int
    height: int
    drops: tuple[int, ...]
    exposed: int
    angles: tuple[int, ...]
    parts: dict[int, Part] = field(default_factory=dict)


_FAMILY_SPECS: tuple[tuple[str, int, int, tuple[int, ...], int, tuple[int, ...]], ...] = (
    ("75", 2, 9, (9,), 1, (75,)),
    ("65", 2, 6, (6,), 1, (65,)),
    ("33", 3, 3, (3,), 2, (33,)),
    ("45", 2, 3, (3,), 1, (45,)),
    ("cheese", 1, 2, (1, 2), 1, (30,)),
)


def _slope_families() -> dict[str, _SlopeFamily]:
    """Slope families built from the catalog: for each (family, width) the
    canonical plain wedge whose geometry matches the family's run and height."""
    families = {spec[0]: _SlopeFamily(*spec) for spec in _FAMILY_SPECS}
    for part in parts_by_shape("slope"):
        m = _PLAIN_SLOPE_NAME.match(part.name)
        if not m:
            continue
        angle = int(m.group(1))
        run, width = part.footprint
        for fam in families.values():
            if angle in fam.angles and run == fam.run and part.height_units == fam.height:
                have = fam.parts.get(width)
                if have is None or (_slope_rank(part), part.ldraw) < (_slope_rank(have), have.ldraw):
                    fam.parts[width] = part
    return families


def _corner_parts() -> tuple[Part | None, Part | None]:
    """(outside corner, valley corner) 2x2 brick-height corner slopes: the
    double-convex and double-concave 45° corners. Inverted corners are
    underside parts and never top a column."""
    convex: Part | None = None
    concave: Part | None = None
    for part in sorted(parts_by_shape("slope_corner"), key=lambda p: p.ldraw):
        if part.height_units != 3 or part.footprint != (2, 2):
            continue
        lname = part.name.lower()
        if "inverted" in lname:
            continue
        if "convex" in lname and convex is None:
            convex = part
        elif "concave" in lname and concave is None:
            concave = part
    return convex, concave


# -- geometry of the packed model --------------------------------------------

def _plain_box_part(part: Part) -> bool:
    """A part that fills its box: the pre-pass packs tall bricks and panels
    into covered interior columns as solid fill, so they cut like bricks."""
    return part.plain and part.shape in ("brick", "plate", "tile", "brick_tall", "panel")


_PLAIN_BOX_NAME = re.compile(r"^(?:Brick|Plate|Tile) \d+ x \d+$")


def _box_cover_parts(shape: str, height: int) -> tuple[Part, ...]:
    """Catalog parts of one box shape and height, largest first, canonical
    names ahead of textured and side-stud variants of the same size."""
    opts = [p for p in parts_by_shape(shape) if p.height_units == height and p.plain]
    return tuple(sorted(opts, key=lambda p: (-p.w * p.d, -max(p.w, p.d),
                                             not _PLAIN_BOX_NAME.match(p.name), p.ldraw)))


def _cover_rect(shape: str, height: int, color_idx: int,
                x0: int, z0: int, w: int, d: int, y: int) -> list[Placement] | None:
    """Greedily cover a (w, d) rectangle at level y with parts of one shape
    and height. None if the shape has no parts that can cover it."""
    opts = _box_cover_parts(shape, height)
    if not opts:
        return None
    free = np.ones((w, d), dtype=bool)
    out: list[Placement] = []
    for ix in range(w):
        for iz in range(d):
            if not free[ix, iz]:
                continue
            placed = False
            for part in opts:
                for pw, pd in part.orientations():
                    if ix + pw > w or iz + pd > d or not free[ix:ix + pw, iz:iz + pd].all():
                        continue
                    free[ix:ix + pw, iz:iz + pd] = False
                    out.append(Placement(
                        part=part, color_idx=color_idx,
                        x_stud=x0 + ix, z_stud=z0 + iz, y_plate=y,
                        w_world=pw, d_world=pd,
                        rotation_deg=0 if (pw, pd) == part.footprint else 90))
                    placed = True
                    break
                if placed:
                    break
            if not placed:
                return None
    return out


def _cover_block(owner: Part, color_idx: int, x0: int, z0: int, w: int, d: int,
                 y0: int, y1: int) -> list[Placement] | None:
    """Cover a (w, d) rectangle over plate levels [y0, y1) with the parts the
    leftover of `owner` should become: plates or tiles stay their shape;
    anything taller is rebuilt from brick courses with plates on top."""
    if y1 <= y0:
        return []
    if owner.shape in ("plate", "tile"):
        out: list[Placement] = []
        for y in range(y0, y1):
            cover = _cover_rect(owner.shape, 1, color_idx, x0, z0, w, d, y)
            if cover is None:
                return None
            out += cover
        return out
    out = []
    y = y0
    while y1 - y >= 3:
        cover = _cover_rect("brick", 3, color_idx, x0, z0, w, d, y)
        if cover is None:
            return None
        out += cover
        y += 3
    while y < y1:
        cover = _cover_rect("plate", 1, color_idx, x0, z0, w, d, y)
        if cover is None:
            return None
        out += cover
        y += 1
    return out


def _rect_minus(ax0: int, az0: int, aw: int, ad: int,
                bx0: int, bz0: int, bw: int, bd: int) -> list[tuple[int, int, int, int]]:
    """Rectangle A minus rectangle B as up to four rectangles (x0, z0, w, d)."""
    ax1, az1 = ax0 + aw, az0 + ad
    bx1, bz1 = bx0 + bw, bz0 + bd
    ix0, ix1 = max(ax0, bx0), min(ax1, bx1)
    iz0, iz1 = max(az0, bz0), min(az1, bz1)
    if ix0 >= ix1 or iz0 >= iz1:
        return [(ax0, az0, aw, ad)]
    out = []
    if ix0 > ax0:
        out.append((ax0, az0, ix0 - ax0, ad))
    if ix1 < ax1:
        out.append((ix1, az0, ax1 - ix1, ad))
    if iz0 > az0:
        out.append((ix0, az0, ix1 - ix0, iz0 - az0))
    if iz1 < az1:
        out.append((ix0, iz1, ix1 - ix0, az1 - iz1))
    return out


class _Assembly:
    """Mutable view of the packed model: which placement owns each voxel."""

    def __init__(self, voxels: np.ndarray, placements: list[Placement]) -> None:
        self.voxels = voxels
        self.col_top = _column_tops(voxels)
        self.owner = np.full(voxels.shape, -1, dtype=np.int32)
        self.alive: dict[int, Placement] = {}
        for p in placements:
            self._add(p)

    def _add(self, p: Placement) -> int:
        idx = len(self.alive)
        while idx in self.alive:
            idx += 1
        self.alive[idx] = p
        self.owner[p.y_plate:p.y_plate + p.part.height_units,
                   p.x_stud:p.x_stud + p.w_world,
                   p.z_stud:p.z_stud + p.d_world] = idx
        return idx

    def placements(self) -> list[Placement]:
        return [self.alive[i] for i in sorted(self.alive)]

    def replace_box(self, x0: int, z0: int, w: int, d: int, y0: int, h: int,
                    color: int, new: list[Placement]) -> bool:
        """Replace whatever owns the (w, h, d) box at (x0, y0, z0) with `new`.

        The box must be one colour and owned by plain box parts that lie
        within its plate range; the parts of each owner outside the box are
        re-covered by parts of the owner's shape, so the cells covered before
        and after are the same. Returns False, changing nothing, if the box
        cannot be cut out cleanly."""
        H, W, D = self.voxels.shape
        if x0 < 0 or z0 < 0 or y0 < 0 or x0 + w > W or z0 + d > D or y0 + h > H:
            return False
        if not (self.voxels[y0:y0 + h, x0:x0 + w, z0:z0 + d] == color).all():
            return False
        owners = np.unique(self.owner[y0:y0 + h, x0:x0 + w, z0:z0 + d])
        if (owners < 0).any():
            return False
        remainders: list[Placement] = []
        for idx in owners:
            p = self.alive.get(int(idx))
            if p is None or not _plain_box_part(p.part) or p.color_idx != color:
                return False
            py0, py1 = p.y_plate, p.y_plate + p.part.height_units
            pieces: list[tuple[int, int, int, int, int, int]] = []
            # Below and above the box, over the owner's whole footprint...
            if py0 < y0:
                pieces.append((p.x_stud, p.z_stud, p.w_world, p.d_world, py0, y0))
            if py1 > y0 + h:
                pieces.append((p.x_stud, p.z_stud, p.w_world, p.d_world, y0 + h, py1))
            # ...and beside it over the plate levels the two share.
            for rx, rz, rw, rd in _rect_minus(p.x_stud, p.z_stud, p.w_world, p.d_world,
                                              x0, z0, w, d):
                pieces.append((rx, rz, rw, rd, max(py0, y0), min(py1, y0 + h)))
            for rx, rz, rw, rd, ya, yb in pieces:
                cover = _cover_block(p.part, color, rx, rz, rw, rd, ya, yb)
                if cover is None:
                    return False
                remainders.extend(cover)
        removed = [self.alive[int(idx)] for idx in owners]
        for idx in owners:
            self._remove(int(idx))
        added = [self._add(p) for p in new + remainders]
        # Every new part must grip studs below it (or stand on the ground),
        # otherwise the cut has turned a held part into a floating one.
        if all(self._gripped(self.alive[i]) for i in added):
            return True
        for i in added:
            self._remove(i)
        for p in removed:
            self._add(p)
        return False

    def _remove(self, idx: int) -> None:
        p = self.alive.pop(idx)
        self.owner[p.y_plate:p.y_plate + p.part.height_units,
                   p.x_stud:p.x_stud + p.w_world,
                   p.z_stud:p.z_stud + p.d_world] = -1

    def _gripped(self, p: Placement) -> bool:
        """Does `p` stand on the ground or on a stud of a part below it?"""
        if p.y_plate == 0:
            return True
        from .stability import bottom_cells, top_studs
        for x, z in bottom_cells(p):
            idx = int(self.owner[p.y_plate - 1, x, z])
            if idx >= 0 and (x, z) in top_studs(self.alive[idx]):
                return True
        return False


def _tread(col_top: np.ndarray, x: int, z: int, dx: int, dz: int,
           level: int, limit: int = 4) -> int:
    """Consecutive columns from (x, z) along (dx, dz) whose top is `level`."""
    W, D = col_top.shape
    n = 0
    while n < limit and 0 <= x < W and 0 <= z < D and col_top[x, z] == level:
        n += 1
        x += dx
        z += dz
    return n


def _edge_drops(col_top: np.ndarray, x: int, z: int) -> dict[str, int]:
    """Directions in which the column's in-grid neighbour is lower, with the
    drop in plates. Out-of-grid neighbours are not steps."""
    W, D = col_top.shape
    top = int(col_top[x, z])
    drops: dict[str, int] = {}
    for name, (dx, dz) in _DIR_VEC.items():
        nx, nz = x + dx, z + dz
        if 0 <= nx < W and 0 <= nz < D and col_top[nx, nz] < top:
            drops[name] = top - int(col_top[nx, nz])
    return drops


def _is_ridge(col_top: np.ndarray, x: int, z: int) -> bool:
    """A column on a flat run of at most two studs along one axis that falls
    away on both ends: the ridge between two roof faces."""
    W, D = col_top.shape
    top = int(col_top[x, z])
    for dx, dz in ((1, 0), (0, 1)):
        back = _tread(col_top, x - dx, z - dz, -dx, -dz, top, limit=3)
        fwd = _tread(col_top, x + dx, z + dz, dx, dz, top, limit=3)
        if back + 1 + fwd > 2:
            continue
        bx, bz = x - dx * (back + 1), z - dz * (back + 1)
        fx, fz = x + dx * (fwd + 1), z + dz * (fwd + 1)
        if (0 <= bx < W and 0 <= bz < D and col_top[bx, bz] < top
                and 0 <= fx < W and 0 <= fz < D and col_top[fx, fz] < top):
            return True
    return False


def _families_for(drop: int, tread_here: int, tread_beyond: int) -> list[str]:
    """Slope families to try, best first, for a step edge with this rise and
    these tread widths (studs at the upper and lower levels)."""
    if drop in (1, 2):
        return ["cheese"]
    if drop == 3:
        return ["33", "45"] if min(tread_here, tread_beyond) >= 2 else ["45"]
    if drop in (6, 9) and tread_beyond <= 2:
        return ["65" if drop == 6 else "75"]
    return []


def _tile_ridges(asm: _Assembly, ridge: set[tuple[int, int]]) -> int:
    """Swap bricks whose whole top is ridge for two plates and a tile."""
    swapped = 0
    for idx in sorted(asm.alive):
        p = asm.alive[idx]
        if p.part.shape != "brick" or p.part.height_units != 3 or not p.part.plain:
            continue
        cells = [(p.x_stud + dx, p.z_stud + dz)
                 for dx in range(p.w_world) for dz in range(p.d_world)]
        if not all(c in ridge and asm.col_top[c] == p.y_plate + 2 for c in cells):
            continue
        new: list[Placement] = []
        ok = True
        for dy, shape in ((0, "plate"), (1, "plate"), (2, "tile")):
            cover = _cover_rect(shape, 1, p.color_idx, p.x_stud, p.z_stud,
                                p.w_world, p.d_world, p.y_plate + dy)
            if cover is None:
                ok = False
                break
            new.extend(cover)
        if ok and asm.replace_box(p.x_stud, p.z_stud, p.w_world, p.d_world,
                                  p.y_plate, 3, p.color_idx, new):
            swapped += 1
    return swapped


def _place_corner(asm: _Assembly, part: Part, x0: int, z0: int, top: int,
                  low_dir: str, color: int, exposed: list[tuple[int, int]]) -> bool:
    """A 2x2 corner slope whose top is at `top`; `exposed` are the cells of
    its sloped faces, which nothing may sit on."""
    rot = _slope_rotation(part, (2, 2), low_dir)
    if rot is None:
        return False
    if any(asm.col_top[x, z] != top for x, z in exposed):
        return False
    new = Placement(part=part, color_idx=color, x_stud=x0, z_stud=z0,
                    y_plate=top - 2, w_world=2, d_world=2, rotation_deg=rot)
    return asm.replace_box(x0, z0, 2, 2, top - 2, 3, color, [new])


def _place_edge_run(asm: _Assembly, fam: _SlopeFamily, d: str, edge: int,
                    top: int, perp: tuple[int, int], color: int) -> bool:
    """Place one `fam` slope whose low edge is the column line `edge` along
    direction `d`, spanning perpendicular coordinates perp = (lo, hi)."""
    dx, dz = _DIR_VEC[d]
    width = perp[1] - perp[0] + 1
    part = fam.parts.get(width)
    if part is None:
        return False
    step = dx + dz  # +1 or -1 along the run axis
    far = edge - step * (fam.run - 1)
    lo_run, hi_run = min(edge, far), max(edge, far)
    if dx:
        x0, z0, w, dd = lo_run, perp[0], fam.run, width
        world = (fam.run, width)
    else:
        x0, z0, w, dd = perp[0], lo_run, width, fam.run
        world = (width, fam.run)
    rot = _slope_rotation(part, world, d)
    if rot is None:
        return False
    # The strips the face slants over must be at column top.
    for i in range(fam.exposed):
        c = edge - step * i
        for q in range(perp[0], perp[1] + 1):
            x, z = (c, q) if dx else (q, c)
            if not (0 <= x < asm.col_top.shape[0] and 0 <= z < asm.col_top.shape[1]):
                return False
            if asm.col_top[x, z] != top:
                return False
    new = Placement(part=part, color_idx=color, x_stud=x0, z_stud=z0,
                    y_plate=top - fam.height + 1, w_world=w, d_world=dd,
                    rotation_deg=rot)
    return asm.replace_box(x0, z0, w, dd, top - fam.height + 1, fam.height, color, [new])


def _tile_run(asm: _Assembly, fam: _SlopeFamily, d: str, edge: int, top: int,
              cells: list[int], color: int) -> list[int]:
    """Cover a run of edge columns (perpendicular coordinates, consecutive)
    with the widest parts of `fam` that fit. Returns the coordinates that
    could not be covered. Odd courses are tiled from the far end so seams
    alternate between courses."""
    widths = sorted(fam.parts, reverse=True)
    order = list(cells)
    if (top // 3) % 2 == 1:
        order.reverse()
    failed: list[int] = []
    i = 0
    while i < len(order):
        done = False
        for wdt in widths:
            if i + wdt > len(order):
                continue
            seg = order[i:i + wdt]
            lo, hi = min(seg), max(seg)
            if hi - lo + 1 != wdt:
                continue
            if _place_edge_run(asm, fam, d, edge, top, (lo, hi), color):
                i += wdt
                done = True
                break
        if not done:
            failed.append(order[i])
            i += 1
    return failed


# A cheese slope on a lone one-plate step is noise, not a roof: an organic
# shape has hundreds of such steps, and swapping each for a cheese slope and
# the plates split around it half again as many parts.  Cheese courses need
# a run of this many studs along the edge; the other families take any run.
_MIN_RUN = {"cheese": 3}


def _refine_slopes(voxels: np.ndarray, placements: list[Placement], *,
                   tile_ridges: bool = True, cheese: bool = True) -> list[Placement]:
    """Post-pack pass: fit roof parts to the step edges of the column-top map.

    Order: ridges are tiled, valleys get the concave corner, outside corners
    the convex one, then step edges are covered run by run from the steepest
    family down (75°, 65°, 33° falling back to 45°, 45°, cheese).  Every swap
    is volume-preserving (see `_Assembly.replace_box`).  `cheese=False`
    leaves one-plate steps as plates: on an organic shape with hundreds of
    them, cheese courses add about a third more parts."""
    asm = _Assembly(voxels, placements)
    col_top = asm.col_top
    W, D = col_top.shape
    families = _slope_families()
    convex, concave = _corner_parts()

    drops: dict[tuple[int, int], dict[str, int]] = {}
    ridge: set[tuple[int, int]] = set()
    for x in range(W):
        for z in range(D):
            if col_top[x, z] < 0:
                continue
            dr = _edge_drops(col_top, x, z)
            if dr:
                drops[(x, z)] = dr
            if _is_ridge(col_top, x, z):
                ridge.add((x, z))

    if tile_ridges and ridge:
        _tile_ridges(asm, ridge)

    def color_at(x: int, z: int) -> int:
        return int(voxels[col_top[x, z], x, z])

    # Valleys: a column whose two neighbours in adjacent directions are
    # higher, whose opposite neighbours are level with it and whose far
    # diagonal is a brick lower, is the low corner of a concave corner slope
    # whose high L is covered by the course above.
    if concave is not None:
        for x in range(W):
            for z in range(D):
                top = int(col_top[x, z])
                if top < 2 or (x, z) in ridge:
                    continue
                for sx in (1, -1):
                    for sz in (1, -1):
                        hx, hz = x + sx, z + sz
                        lx, lz = x - sx, z - sz
                        if not (0 <= hx < W and 0 <= hz < D and 0 <= lx < W and 0 <= lz < D):
                            continue
                        if not (col_top[hx, z] > top and col_top[x, hz] > top
                                and col_top[hx, hz] > top
                                and col_top[lx, z] == top and col_top[x, lz] == top
                                and col_top[lx, lz] == top - 3):
                            continue
                        low_dir = ("+x" if sx < 0 else "-x") + ("+z" if sz < 0 else "-z")
                        _place_corner(asm, concave, min(x, hx), min(z, hz), top,
                                      low_dir, color_at(x, z), [(x, z)])

    # Outside corners: a brick drop in two adjacent directions.
    if convex is not None:
        for (x, z), dr in drops.items():
            if (x, z) in ridge or len(dr) != 2:
                continue
            xs = [k for k in dr if "x" in k]
            zs = [k for k in dr if "z" in k]
            if len(xs) != 1 or len(zs) != 1 or dr[xs[0]] != 3 or dr[zs[0]] != 3:
                continue
            sx, sz = _DIR_VEC[xs[0]][0], _DIR_VEC[zs[0]][1]
            _place_corner(asm, convex, min(x, x - sx), min(z, z - sz), int(col_top[x, z]),
                          xs[0] + zs[0], color_at(x, z), [(x, z), (x - sx, z), (x, z - sz)])

    # Step edges, grouped into runs along the ridge direction.
    # key: (family, direction, edge coordinate along the run axis, top, colour)
    groups: dict[tuple[str, str, int, int, int], list[int]] = {}
    fallback: dict[tuple[str, str, int, int, int], list[int]] = {}
    for (x, z), dr in drops.items():
        if (x, z) in ridge or len(dr) > 2:
            continue
        if len(dr) == 2 and len({k[1] for k in dr}) == 1:
            continue  # opposite drops: a flat too wide to be a ridge, not a roof edge
        top = int(col_top[x, z])
        for d, drop in dr.items():
            dx, dz = _DIR_VEC[d]
            here = _tread(col_top, x, z, -dx, -dz, top)
            beyond = _tread(col_top, x + dx, z + dz, dx, dz, top - drop)
            fams = [f for f in _families_for(drop, here, beyond)
                    if families[f].parts and (cheese or f != "cheese")]
            if not fams:
                continue
            edge, perp = (x, z) if dx else (z, x)
            key = (fams[0], d, edge, top, color_at(x, z))
            groups.setdefault(key, []).append(perp)
            if len(fams) > 1:
                fallback[key] = fams[1:]

    def runs(coords: list[int]) -> list[list[int]]:
        coords = sorted(set(coords))
        out: list[list[int]] = []
        for c in coords:
            if out and c == out[-1][-1] + 1:
                out[-1].append(c)
            else:
                out.append([c])
        return out

    order = {spec[0]: i for i, spec in enumerate(_FAMILY_SPECS)}
    pending = sorted(groups.items(), key=lambda kv: (order[kv[0][0]], -kv[0][3], kv[0][1], kv[0][2]))
    while pending:
        (fname, d, edge, top, color), coords = pending.pop(0)
        failed: list[int] = []
        for run in runs(coords):
            if len(run) < _MIN_RUN.get(fname, 1):
                continue
            failed += _tile_run(asm, families[fname], d, edge, top, run, color)
        nxt = fallback.get((fname, d, edge, top, color))
        if failed and nxt:
            key = (nxt[0], d, edge, top, color)
            if len(nxt) > 1:
                fallback[key] = nxt[1:]
            pending.append((key, failed))
            pending.sort(key=lambda kv: (order[kv[0][0]], -kv[0][3], kv[0][1], kv[0][2]))

    return asm.placements()


# -- Texture substitution ----------------------------------------------------
#
# Post-pack part-for-part swap that preserves footprint and rotation. Used to
# turn plain 1x2 bricks into log-profile bricks for cabins, plain 1x1 bricks
# into masonry bricks for castles, 45-degree slopes into curved slopes for a
# smoother roof, etc.
#
# Each preset is a mapping from source LDraw part id to the substitute id. If
# the substitute isn't in the catalog we leave the placement alone (silently),
# so unknown presets or partial catalogs don't crash the pipeline.

TEXTURE_PRESETS: dict[str, dict[str, str]] = {
    "log":     {"3004.dat": "30136.dat"},              # wood-cabin walls
    "masonry": {"3004.dat": "98283.dat",               # castle/stone walls
                "3005.dat": "15533.dat"},
    "curved":  {"3039.dat": "15068.dat",               # softer roof edges
                "3037.dat": "61678.dat"},
    "smooth":  {},
    "default": {},
}


# Named groups of visually-similar wall bricks that are safe to swap around by
# position hash without changing geometry. Only *rectangular* full-volume
# bricks — no round/corner/arch. Users get real visible variety on walls.

# Context-aware variety: the pool of alternates for each footprint depends on
# whether the placement sits in a wall corner, along a straight wall, or fully
# surrounded. Side-stud and headlight bricks land ONLY at exterior corners
# (where they read as intentional decorative accents); straight-wall runs get
# plain + textured variants only.

VARIETY_GROUPS: dict[str, dict[str, list[str]]] = {
    "wall_1x1_h3": {
        # 1x1 bricks: corners get the full side-stud pool, straight walls get
        # plain + masonry, and every ~sixth "wall" position gets a subtler
        # accent (single-side-stud only, no four-side which looks busy on a
        # straight run).
        "corner":       ["4070.dat", "87087.dat", "26604.dat", "47905.dat", "4733.dat"],
        "wall_accent":  ["4070.dat", "87087.dat"],
        "wall":         ["3005.dat", "15533.dat"],
        "interior":     ["3005.dat"],
    },
    "wall_1x2_h3": {
        # 1x2 bricks are rectangular — same textured pool everywhere.
        "corner":       ["3004.dat", "30136.dat", "98283.dat"],
        "wall_accent":  ["3004.dat", "30136.dat", "98283.dat"],
        "wall":         ["3004.dat", "30136.dat", "98283.dat"],
        "interior":     ["3004.dat", "30136.dat", "98283.dat"],
    },
}

# Every Nth "wall" placement picks the accent pool instead. Higher N = rarer.
_ACCENT_MODULO = 6

_VARIETY_GROUP_BY_LDRAW: dict[str, str] = {
    pid: name for name, ctx_map in VARIETY_GROUPS.items()
    for pool in ctx_map.values() for pid in pool
}


def _classify_context(cell_owner: dict[tuple[int, int, int], int],
                       p: Placement) -> str:
    """Corner / wall / interior classification from horizontal neighbours at
    the placement's middle plate."""
    x, y, z = p.x_stud, p.y_plate, p.z_stud
    w, d, h = p.w_world, p.d_world, p.part.height_units
    y_mid = y + h // 2
    has_neg_x = any((x - 1, y_mid, z + dz) in cell_owner for dz in range(d))
    has_pos_x = any((x + w, y_mid, z + dz) in cell_owner for dz in range(d))
    has_neg_z = any((x + dx, y_mid, z - 1) in cell_owner for dx in range(w))
    has_pos_z = any((x + dx, y_mid, z + d) in cell_owner for dx in range(w))
    x_count = int(has_neg_x) + int(has_pos_x)
    z_count = int(has_neg_z) + int(has_pos_z)
    # Interior: neighbours on all four horizontal sides.
    if x_count == 2 and z_count == 2:
        return "interior"
    # Corner: exactly one x-neighbour AND one z-neighbour (an L-junction).
    if x_count == 1 and z_count == 1:
        return "corner"
    # Straight wall run or edge. Every Nth cell picks the accent pool via a
    # scale-invariant position hash — reads as intentional stud rows on
    # otherwise-plain walls, not random speckling.
    accent_seed = ((p.x_stud * 6151) ^ (p.z_stud * 3373) ^ p.y_plate)
    if accent_seed % _ACCENT_MODULO == 0:
        return "wall_accent"
    return "wall"


def apply_variety(placements: list[Placement], *, seed: int = 0
                   ) -> list[Placement]:
    """Position-hash swap among visually-similar wall bricks, restricted by
    context: corners get side-stud/headlight accents; straight walls get
    plain + masonry variety; interior surfaces stay plain. Purely visual —
    footprint, height and volume are unchanged."""
    parts_by_ldraw = {p.ldraw: p for p in full_catalog()}

    # Cell-owner map for the neighbourhood classifier.
    cell_owner: dict[tuple[int, int, int], int] = {}
    for i, p in enumerate(placements):
        for dx in range(p.w_world):
            for dz in range(p.d_world):
                for dy in range(p.part.height_units):
                    cell_owner[(p.x_stud + dx, p.y_plate + dy, p.z_stud + dz)] = i

    result: list[Placement] = []
    for p in placements:
        group = _VARIETY_GROUP_BY_LDRAW.get(p.part.ldraw)
        if group is None:
            result.append(p)
            continue
        context = _classify_context(cell_owner, p)
        options_ids = VARIETY_GROUPS[group].get(context, [])
        options = [parts_by_ldraw[pid] for pid in options_ids if pid in parts_by_ldraw]
        if not options:
            result.append(p)
            continue
        h = ((p.x_stud * 73856093)
             ^ (p.y_plate * 19349663)
             ^ (p.z_stud * 83492791) ^ seed) & 0x7fffffff
        new_part = options[h % len(options)]
        if new_part.ldraw == p.part.ldraw:
            result.append(p)
        else:
            result.append(Placement(
                part=new_part, color_idx=p.color_idx,
                x_stud=p.x_stud, z_stud=p.z_stud, y_plate=p.y_plate,
                w_world=p.w_world, d_world=p.d_world, rotation_deg=p.rotation_deg,
            ))
    return result


def substitute_by_color(placements: list[Placement],
                         color_swaps: dict[int, str]) -> list[Placement]:
    """Swap parts whose LDraw color code matches any key of `color_swaps`
    with the substitute LDraw part id.

    Use this to turn 1x2 bricks in the 'window' color into actual 1x2 window
    frames, brown bricks into door parts, etc. The user picks the colour;
    we handle the swap.

    The substitute must have the placement's footprint.  Where it is taller
    than the placement (a window frame is two bricks, a door five), the
    placement is swapped together with the parts stacked directly on it:
    a run of same-colour, same-footprint parts whose heights add up to the
    substitute's height becomes the one taller part.  A placement with
    nothing suitable above it is left alone, so nothing ever overlaps.
    """
    if not color_swaps:
        return list(placements)
    from .colors import palette
    pal = palette()
    parts_by_ldraw = {p.ldraw: p for p in full_catalog()}

    def wants(p: Placement) -> Part | None:
        target = color_swaps.get(pal[p.color_idx].code)
        new_part = parts_by_ldraw.get(target) if target else None
        if new_part is None:
            return None
        world = (p.w_world, p.d_world)
        if world not in (new_part.footprint, new_part.footprint[::-1]):
            return None
        return new_part

    # Parts stacked exactly on one another, by footprint column.
    by_column: dict[tuple[int, int, int, int, int], Placement] = {
        (p.x_stud, p.z_stud, p.w_world, p.d_world, p.y_plate): p for p in placements}
    consumed: set[int] = set()
    result: list[Placement] = []
    for p in sorted(placements, key=lambda q: q.y_plate):
        if id(p) in consumed:
            continue
        new_part = wants(p)
        if new_part is None:
            result.append(p)
            continue
        stack = [p]
        height = p.part.height_units
        while height < new_part.height_units:
            above = by_column.get((p.x_stud, p.z_stud, p.w_world, p.d_world, p.y_plate + height))
            if above is None or id(above) in consumed or above.color_idx != p.color_idx:
                break
            stack.append(above)
            height += above.part.height_units
        if height != new_part.height_units:
            result.append(p)
            continue
        consumed.update(id(q) for q in stack)
        result.append(Placement(
            part=new_part, color_idx=p.color_idx,
            x_stud=p.x_stud, z_stud=p.z_stud, y_plate=p.y_plate,
            w_world=p.w_world, d_world=p.d_world, rotation_deg=p.rotation_deg,
        ))
    return result


def substitute_texture(placements: list[Placement], preset: str) -> list[Placement]:
    """Swap parts according to the named preset. Returns a new list; input is
    unchanged. Unknown presets and missing catalog entries pass through as-is."""
    if not preset or preset == "default":
        return list(placements)
    subs = TEXTURE_PRESETS.get(preset, {})
    if not subs:
        return list(placements)
    parts_by_ldraw = {p.ldraw: p for p in full_catalog()}
    result: list[Placement] = []
    for p in placements:
        target_ldraw = subs.get(p.part.ldraw)
        new_part = parts_by_ldraw.get(target_ldraw) if target_ldraw else None
        if new_part is None:
            result.append(p)
            continue
        # Only substitute when the substitute's declared footprint can hold
        # the placement's world footprint (either identical or 90-rotated).
        world = (p.w_world, p.d_world)
        if world not in (new_part.footprint, (new_part.footprint[1], new_part.footprint[0])):
            result.append(p)
            continue
        result.append(Placement(
            part=new_part, color_idx=p.color_idx,
            x_stud=p.x_stud, z_stud=p.z_stud, y_plate=p.y_plate,
            w_world=p.w_world, d_world=p.d_world, rotation_deg=p.rotation_deg,
        ))
    return result


# -- Semantic feature detection (windows, doors) ------------------------------
#
# Post-pack pass that finds connected same-color regions of non-wall colour,
# checks whether their shape matches a window (thin-and-short) or a door
# (thin-and-tall-reaching-ground), and if so replaces the whole cluster with
# a single LEGO window/door part. Runs after all other substitutions.


def _cluster_by_color(placements: list[Placement],
                       indices: list[int]) -> dict[int, list[int]]:
    """6-connected clustering of the placements listed by index. Returns a
    mapping cluster-root -> list of placement indices."""
    cell_owner: dict[tuple[int, int, int], int] = {}
    for idx in indices:
        p = placements[idx]
        for dx in range(p.w_world):
            for dz in range(p.d_world):
                for dy in range(p.part.height_units):
                    cell_owner[(p.x_stud + dx, p.y_plate + dy, p.z_stud + dz)] = idx

    parent = {idx: idx for idx in indices}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (x, y, z), idx in cell_owner.items():
        for (dx, dy, dz) in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
                              (0, 0, 1), (0, 0, -1)):
            j = cell_owner.get((x + dx, y + dy, z + dz))
            if j is not None and j != idx:
                pi, pj = find(idx), find(j)
                if pi != pj:
                    parent[pi] = pj

    clusters: dict[int, list[int]] = {}
    for idx in indices:
        root = find(idx)
        clusters.setdefault(root, []).append(idx)
    return clusters


# Window frames and doors by (studs wide, plates tall).
_WINDOW_BY_SIZE = {(2, 6): "60592.dat", (2, 9): "60593.dat",
                   (4, 9): "60594.dat", (6, 9): "3853.dat"}
_DOOR_BY_SIZE = {(4, 15): "60623.dat", (4, 18): "60614.dat"}


def detect_windows_and_doors(placements: list[Placement], *,
                              wall_color_top_n: int = 2,
                              min_cluster_size: int = 2,
                              min_door_plates: int = 9) -> tuple[list[Placement], dict]:
    """Detect window and door-shaped color clusters and swap them for real
    LEGO window/door parts. Returns (new_placements, stats).

    A cluster is swapped only when it is a full rectangle the exact size of
    a window frame or door; the smallest, a 1x2x2 frame, is two bricks."""
    from collections import Counter, defaultdict
    parts_by_ldraw = {p.ldraw: p for p in full_catalog()}

    if not placements:
        return [], {"clusters_found": 0, "windows_placed": 0, "doors_placed": 0}

    color_counts = Counter(p.color_idx for p in placements)
    wall_colors = {c for c, _ in color_counts.most_common(wall_color_top_n)}
    stats = {"clusters_found": 0, "windows_placed": 0, "doors_placed": 0}

    by_color: dict[int, list[int]] = defaultdict(list)
    for i, p in enumerate(placements):
        if p.color_idx not in wall_colors:
            by_color[p.color_idx].append(i)

    to_replace: dict[int, Placement | None] = {}

    for color_idx, indices in by_color.items():
        if len(indices) < min_cluster_size:
            continue
        clusters = _cluster_by_color(placements, indices)
        for cluster_indices in clusters.values():
            if len(cluster_indices) < min_cluster_size:
                continue
            stats["clusters_found"] += 1

            xs, x_ends, zs, z_ends, ys, y_ends = [], [], [], [], [], []
            for idx in cluster_indices:
                p = placements[idx]
                xs.append(p.x_stud); x_ends.append(p.x_stud + p.w_world)
                zs.append(p.z_stud); z_ends.append(p.z_stud + p.d_world)
                ys.append(p.y_plate); y_ends.append(p.y_plate + p.part.height_units)

            x_extent = max(x_ends) - min(xs)
            z_extent = max(z_ends) - min(zs)
            y_extent = max(y_ends) - min(ys)
            y_min = min(ys)

            # Feature must sit on a single wall face: thin in exactly one
            # horizontal axis.
            thin_x = x_extent <= 1
            thin_z = z_extent <= 1
            if thin_x == thin_z:
                continue
            width = z_extent if thin_x else x_extent

            # The replacement must be exactly the cluster's size, or it
            # would overlap its neighbours or leave a hole: a window frame
            # is 2 or 3 bricks tall and a door 5 or 6, so the cluster has
            # to be a full rectangle of one of those sizes.
            cells = sum(p.w_world * p.d_world * p.part.height_units
                        for p in (placements[i] for i in cluster_indices))
            if cells != x_extent * z_extent * y_extent:
                continue
            target: str | None = None
            # Window: not at the floor.
            if y_min >= 3:
                target = _WINDOW_BY_SIZE.get((width, y_extent))
            # Door: reaches the floor.
            elif y_extent >= min_door_plates and y_min <= 4:
                target = _DOOR_BY_SIZE.get((width, y_extent))

            if target is None:
                continue
            new_part = parts_by_ldraw.get(target)
            if new_part is None:
                continue

            if thin_x:
                w_world, d_world = 1, width
            else:
                w_world, d_world = width, 1
            if new_part.footprint == (w_world, d_world):
                rot = 0
            elif new_part.footprint == (d_world, w_world):
                rot = 90
            else:
                continue

            base = placements[min(cluster_indices, key=lambda i: placements[i].y_plate)]
            replacement = Placement(
                part=new_part, color_idx=base.color_idx,
                x_stud=min(xs), z_stud=min(zs), y_plate=y_min,
                w_world=w_world, d_world=d_world, rotation_deg=rot,
            )
            for idx in cluster_indices:
                to_replace[idx] = None
            to_replace[min(cluster_indices)] = replacement

            if new_part.shape == "window":
                stats["windows_placed"] += 1
            else:
                stats["doors_placed"] += 1

    result: list[Placement] = []
    for i, p in enumerate(placements):
        if i in to_replace:
            r = to_replace[i]
            if r is not None:
                result.append(r)
        else:
            result.append(p)
    return result, stats


# -- Architectural styling ---------------------------------------------------
#
# Post-pack pass that gives a house-like model the "LEGO Ideas modular"
# character: distinct color bands for foundation / walls / cornice / ridge,
# shingle pattern on roof surfaces, and alternating "quoin" cornerstones at
# vertical wall corners. Detects the wall-to-roof transition by looking at
# where each Y-level's occupied cross-section area starts falling off
# sharply.


def _detect_roof_line(placements: list[Placement], *,
                       foundation_plates: int = 3) -> int:
    """Y-plate below which the model is walls (roughly constant cross-section)
    and above which it becomes roof (shrinking cross-section).

    Locates the sharpest downward derivative in cross-section area, skipping
    the foundation. Falls back to 60% of the model height if the derivative
    is flat (uncommon)."""
    from collections import defaultdict
    if not placements:
        return 0
    area_by_y: dict[int, int] = defaultdict(int)
    for p in placements:
        for y in range(p.y_plate, p.y_plate + p.part.height_units):
            area_by_y[y] += p.w_world * p.d_world
    if not area_by_y:
        return 0
    min_y = min(area_by_y)
    max_y = max(area_by_y)
    ys_sorted = sorted(area_by_y)

    # Skip the foundation slab. Its area is typically the full footprint and
    # dominates the derivative below it — bogus signal for a roof line.
    non_foundation = [y for y in ys_sorted if y >= min_y + foundation_plates]
    if len(non_foundation) < 4:
        return max_y + 1

    # Sharpest downward drop in cross-section area. Uses a 2-plate stride to
    # smooth out per-plate jitter from the alternating brick/plate courses.
    best_drop = 0
    best_y = min_y + int((max_y - min_y) * 0.6)  # fallback = 60% up
    stride = 2
    for i in range(stride, len(non_foundation)):
        y = non_foundation[i]
        prev_y = non_foundation[i - stride]
        drop = area_by_y[prev_y] - area_by_y[y]
        if drop > best_drop:
            best_drop = drop
            best_y = y
    return best_y


def _wall_corner_columns(placements: list[Placement],
                          roof_line: int) -> set[tuple[int, int]]:
    """(x_stud, z_stud) columns that sit on an outside wall corner — i.e.
    have exactly one x-neighbour AND one z-neighbour among wall placements.
    Detected at the midpoint of the wall region."""
    if not placements:
        return set()
    mid_y = max(0, roof_line // 2)
    # Studs occupied at mid-wall.
    occupied: set[tuple[int, int]] = set()
    for p in placements:
        if p.y_plate > mid_y or p.y_plate + p.part.height_units <= mid_y:
            continue
        for dx in range(p.w_world):
            for dz in range(p.d_world):
                occupied.add((p.x_stud + dx, p.z_stud + dz))
    corners: set[tuple[int, int]] = set()
    for (x, z) in occupied:
        neg_x = (x - 1, z) in occupied
        pos_x = (x + 1, z) in occupied
        neg_z = (x, z - 1) in occupied
        pos_z = (x, z + 1) in occupied
        if (neg_x + pos_x == 1) and (neg_z + pos_z == 1):
            corners.add((x, z))
    return corners


def apply_architectural_style(placements: list[Placement], *,
                               foundation_plates: int = 3,
                               cornice: bool = True,
                               ridge: bool = True,
                               shingles: bool = True,
                               quoins: bool = True,
                               foundation_code: int = 72,   # Dark Bluish Gray
                               cornice_code: int = 70,       # Reddish Brown
                               ridge_code: int = 15,         # White
                               shingle_a_code: int = 72,     # Dark Bluish Gray
                               shingle_b_code: int = 0,      # Black
                               quoin_code: int = 71,         # Light Bluish Gray
                               ) -> tuple[list[Placement], dict]:
    """Recolor the model into architectural bands and add texture pattern to
    roof + wall corners. Returns (new_placements, stats)."""
    from .colors import palette
    if not placements:
        return list(placements), {"changed": 0}
    pal = palette()
    idx_by_code = {c.code: i for i, c in enumerate(pal)}

    def _to_idx(code: int, fallback: int) -> int:
        if code not in idx_by_code:
            warnings.warn(f"LDraw colour {code} is not in the palette; using {pal[fallback].name}",
                          RuntimeWarning, stacklevel=2)
        return idx_by_code.get(code, fallback)

    default_idx = 0
    foundation_idx = _to_idx(foundation_code, default_idx)
    cornice_idx = _to_idx(cornice_code, default_idx)
    ridge_idx = _to_idx(ridge_code, default_idx)
    shingle_a_idx = _to_idx(shingle_a_code, default_idx)
    shingle_b_idx = _to_idx(shingle_b_code, default_idx)
    quoin_idx = _to_idx(quoin_code, default_idx)

    roof_line = _detect_roof_line(placements)
    max_y_top = max(p.y_plate + p.part.height_units for p in placements)
    min_y = min(p.y_plate for p in placements)
    corners = _wall_corner_columns(placements, roof_line) if quoins else set()

    stats = {"foundation": 0, "cornice": 0, "ridge": 0, "shingles": 0, "quoins": 0}
    result: list[Placement] = []
    for p in placements:
        new_idx = p.color_idx
        y_bottom = p.y_plate
        y_top = p.y_plate + p.part.height_units

        # 1. Foundation band (bottom `foundation_plates`).
        if y_bottom < min_y + foundation_plates:
            new_idx = foundation_idx
            stats["foundation"] += 1
        # 2. Cornice (last full brick-row before the roof line — 3 plates,
        #    so the brown strip is visible instead of a single-plate hairline).
        elif cornice and roof_line - 3 <= y_bottom < roof_line:
            new_idx = cornice_idx
            stats["cornice"] += 1
        # 3. Ridge (top two plates of the roof, so the white cap stands out).
        elif ridge and y_top >= max_y_top - 1:
            new_idx = ridge_idx
            stats["ridge"] += 1
        # 4. Roof shingles: on tiles/plates above the roof line, alternate
        #    between two dark colours in an offset-checkerboard pattern so it
        #    reads as staggered shingle rows.
        elif shingles and y_bottom >= roof_line and p.part.shape in ("tile", "plate"):
            parity = ((p.x_stud + p.y_plate + p.z_stud) // 1) % 2
            new_idx = shingle_a_idx if parity == 0 else shingle_b_idx
            stats["shingles"] += 1
        # 5. Quoin cornerstones: for wall bricks at exact spatial corners,
        #    alternate between the wall colour and quoin_code every 3 plates
        #    (one brick height).
        elif quoins and y_bottom < roof_line and p.part.shape == "brick":
            studs = {(p.x_stud + dx, p.z_stud + dz)
                     for dx in range(p.w_world) for dz in range(p.d_world)}
            if studs & corners:
                brick_row = (y_bottom - min_y) // 3
                if brick_row % 2 == 1:
                    new_idx = quoin_idx
                    stats["quoins"] += 1

        if new_idx != p.color_idx:
            result.append(Placement(
                part=p.part, color_idx=new_idx,
                x_stud=p.x_stud, z_stud=p.z_stud, y_plate=p.y_plate,
                w_world=p.w_world, d_world=p.d_world, rotation_deg=p.rotation_deg,
            ))
        else:
            result.append(p)
    stats["changed"] = sum(v for k, v in stats.items() if k != "changed")
    stats["roof_line"] = roof_line
    return result, stats


def smooth_isolated_colors(placements: list[Placement], *,
                            majority_ratio: float = 0.55) -> list[Placement]:
    """Recolor isolated single-brick color outliers to match the surrounding
    majority. For each placement, look at all 6-connected neighbours in
    (x, y, z); if fewer than 25% of the neighbours share its color and a
    strict majority share some *other* color, recolor to that other color.
    Kills single-brick speckles without touching intentional accents.
    """
    from collections import Counter

    if len(placements) < 4:
        return list(placements)

    cell_owner: dict[tuple[int, int, int], int] = {}
    for i, p in enumerate(placements):
        for dx in range(p.w_world):
            for dz in range(p.d_world):
                for dy in range(p.part.height_units):
                    cell_owner[(p.x_stud + dx, p.y_plate + dy, p.z_stud + dz)] = i

    result = list(placements)
    for i, p in enumerate(placements):
        neighbour_colors: list[int] = []
        seen_neighbours: set[int] = set()
        for dx in range(p.w_world):
            for dz in range(p.d_world):
                for dy in range(p.part.height_units):
                    for step in ((1, 0, 0), (-1, 0, 0),
                                  (0, 1, 0), (0, -1, 0),
                                  (0, 0, 1), (0, 0, -1)):
                        n = cell_owner.get((p.x_stud + dx + step[0],
                                            p.y_plate + dy + step[1],
                                            p.z_stud + dz + step[2]))
                        if n is not None and n != i and n not in seen_neighbours:
                            seen_neighbours.add(n)
                            neighbour_colors.append(placements[n].color_idx)
        if len(neighbour_colors) < 3:
            continue
        own = sum(1 for c in neighbour_colors if c == p.color_idx)
        if own / len(neighbour_colors) >= 0.25:
            continue
        counts = Counter(c for c in neighbour_colors if c != p.color_idx)
        if not counts:
            continue
        majority_color, majority_count = counts.most_common(1)[0]
        if majority_count / len(neighbour_colors) < majority_ratio:
            continue
        result[i] = Placement(
            part=p.part, color_idx=majority_color,
            x_stud=p.x_stud, z_stud=p.z_stud, y_plate=p.y_plate,
            w_world=p.w_world, d_world=p.d_world, rotation_deg=p.rotation_deg,
        )
    return result


def summarize(placements: list[Placement]) -> dict[tuple[str, int], int]:
    """(ldraw_part, ldraw_color_code) -> count."""
    from .colors import palette
    pal = palette()
    counts: dict[tuple[str, int], int] = {}
    for p in placements:
        key = (p.part.ldraw, pal[p.color_idx].code)
        counts[key] = counts.get(key, 0) + 1
    return counts
