"""Convert an optimized cell Assembly into a list of `pack.Placement`.

The cell grid is at plate resolution (1 stud × 1 plate × 1 stud). This
module runs a two-pass consolidator:

1. **Brick promotion** — walk vertical 1-stud columns; wherever 3 consecutive
   same-color same-type cells stack, emit one brick (3005 / 15533 / 30136 /
   4070 …) covering that 3-plate span. This is the only way MASONRY / LOG /
   HEADLIGHT cells become their real (3-plate-tall) LEGO parts.

2. **Plate / tile packing** — for each (y-level, cell_type, color) group,
   run a greedy 2D packing over the remaining cells using the largest
   available plate (for PLAIN, LOG, MASONRY, HEADLIGHT that didn't promote)
   or tile (for TILE cells). Cheese slopes (54200) are emitted per-cell.

Result: parts count drops 5-10× vs the pre-consolidator export, with zero
shape/color change and full preservation of MASONRY / LOG texture where it
fits a real brick.
"""

from __future__ import annotations

import numpy as np

from ..bricks import Part, catalog
from ..pack import Placement
from ..templates.base import part_by_ldraw, place_rect
from .loss import (Assembly, EMPTY, PLAIN, LOG, MASONRY, TILE,
                    SLOPE_YP, SLOPE_YM, HEADLIGHT,
                    SLOPE_XP, SLOPE_XM, SLOPE_ZP, SLOPE_ZM,
                    SLOPE_INV_XP, SLOPE_INV_XM, SLOPE_INV_ZP, SLOPE_INV_ZM,
                    SLOPE_CORNER_PP, SLOPE_CORNER_PN,
                    SLOPE_CORNER_NP, SLOPE_CORNER_NN,
                    PANEL_XP, PANEL_XM, PANEL_ZP, PANEL_ZM,
                    PANEL_TALL_XP, PANEL_TALL_XM, PANEL_TALL_ZP, PANEL_TALL_ZM,
                    WINDOW_XP, WINDOW_XM, WINDOW_ZP, WINDOW_ZM,
                    DOOR_XP, DOOR_XM, DOOR_ZP, DOOR_ZM,
                    BRICK_TALL,
                    CHEESE_XP, CHEESE_XM, CHEESE_ZP, CHEESE_ZM,
                    CURVED_XP, CURVED_XM, CURVED_ZP, CURVED_ZM,
                    SLOPE_4X2_XP, SLOPE_4X2_XM, SLOPE_4X2_ZP, SLOPE_4X2_ZM,
                    CURVED_2X4_XP, CURVED_2X4_XM, CURVED_2X4_ZP, CURVED_2X4_ZM,
                    CURVED_4X1_XP, CURVED_4X1_XM, CURVED_4X1_ZP, CURVED_4X1_ZM,
                    CURVED_6X1_XP, CURVED_6X1_XM, CURVED_6X1_ZP, CURVED_6X1_ZM)


# Each new tall-part cell type maps to (LDraw part, footprint (w, d),
# height_in_plates, rotation_deg). The consolidator promotes a matching
# stack of cells to one part when the whole span fits.
#
# Rotation convention: LDraw parts default to slope_dir=+x (tall/face at +x
# side). We rotate 0/90/180/270 to map (XP,ZM,XM,ZP) semantics.
_TALL_PARTS: dict[int, tuple[str, tuple[int, int], int, int]] = {
    # 45° slopes (3 plates tall, 2x1 footprint) — 4 orientations.
    SLOPE_XP:         ("3040.dat",  (2, 1),  3,   0),
    SLOPE_XM:         ("3040.dat",  (2, 1),  3, 180),
    SLOPE_ZP:         ("3040.dat",  (2, 1),  3, 270),
    SLOPE_ZM:         ("3040.dat",  (2, 1),  3,  90),
    # Inverted slopes (3 plates tall, 2x1) — for overhangs.
    SLOPE_INV_XP:     ("3665.dat",  (2, 1),  3,   0),
    SLOPE_INV_XM:     ("3665.dat",  (2, 1),  3, 180),
    SLOPE_INV_ZP:     ("3665.dat",  (2, 1),  3, 270),
    SLOPE_INV_ZM:     ("3665.dat",  (2, 1),  3,  90),
    # Corner slopes (3 plates, 2x2) — 4 quadrants.
    SLOPE_CORNER_PP:  ("3045.dat",  (2, 2),  3,   0),
    SLOPE_CORNER_PN:  ("3045.dat",  (2, 2),  3,  90),
    SLOPE_CORNER_NN:  ("3045.dat",  (2, 2),  3, 180),
    SLOPE_CORNER_NP:  ("3045.dat",  (2, 2),  3, 270),
    # 1-brick panels (Panel 1x2x1, 3 plates tall, 1-stud thick).
    PANEL_XP:         ("4865.dat",  (2, 1),  3,   0),
    PANEL_XM:         ("4865.dat",  (2, 1),  3, 180),
    PANEL_ZP:         ("4865.dat",  (2, 1),  3, 270),
    PANEL_ZM:         ("4865.dat",  (2, 1),  3,  90),
    # 2-brick panels (Panel 1x2x2, 6 plates tall).
    PANEL_TALL_XP:    ("4864b.dat", (2, 1),  6,   0),
    PANEL_TALL_XM:    ("4864b.dat", (2, 1),  6, 180),
    PANEL_TALL_ZP:    ("4864b.dat", (2, 1),  6, 270),
    PANEL_TALL_ZM:    ("4864b.dat", (2, 1),  6,  90),
    # Window (2x1x2 frame, 6 plates tall).
    WINDOW_XP:        ("60592.dat", (2, 1),  6,   0),
    WINDOW_XM:        ("60592.dat", (2, 1),  6, 180),
    WINDOW_ZP:        ("60592.dat", (2, 1),  6, 270),
    WINDOW_ZM:        ("60592.dat", (2, 1),  6,  90),
    # Door (4x1x5, 15 plates tall).
    DOOR_XP:          ("60623.dat", (4, 1), 15,   0),
    DOOR_XM:          ("60623.dat", (4, 1), 15, 180),
    DOOR_ZP:          ("60623.dat", (4, 1), 15, 270),
    DOOR_ZM:          ("60623.dat", (4, 1), 15,  90),
    # Brick 1x2x5 (15 plates tall, 2 studs wide).
    BRICK_TALL:       ("2454.dat",  (2, 1), 15,   0),
    # Cheese slopes (54200): 1x1x2, tile-anywhere primitive for curved
    # regions. Four orientations map to standard rotations.
    CHEESE_XP:        ("54200.dat", (1, 1),  2,   0),
    CHEESE_XM:        ("54200.dat", (1, 1),  2, 180),
    CHEESE_ZP:        ("54200.dat", (1, 1),  2, 270),
    CHEESE_ZM:        ("54200.dat", (1, 1),  2,  90),
    # Curved slopes (15068 Slope Curved 2x2): the primitive real LEGO
    # spheres and dome shapes are built from. 2x2 footprint, 3 plates tall.
    CURVED_XP:        ("15068.dat", (2, 2),  3,   0),
    CURVED_XM:        ("15068.dat", (2, 2),  3, 180),
    CURVED_ZP:        ("15068.dat", (2, 2),  3, 270),
    CURVED_ZM:        ("15068.dat", (2, 2),  3,  90),
    # Slope 45 4x2 (3037) — the biggest flat 45° slope primitive we have.
    # Footprint (4, 2), 3 plates tall. Claims 24 cells per instance.
    SLOPE_4X2_XP:     ("3037.dat", (4, 2),  3,   0),
    SLOPE_4X2_XM:     ("3037.dat", (4, 2),  3, 180),
    SLOPE_4X2_ZP:     ("3037.dat", (4, 2),  3, 270),
    SLOPE_4X2_ZM:     ("3037.dat", (4, 2),  3,  90),
    # Curved slope 2x4 (24309) — biggest curved slope primitive. 24 cells.
    CURVED_2X4_XP:    ("24309.dat", (4, 2),  3,   0),
    CURVED_2X4_XM:    ("24309.dat", (4, 2),  3, 180),
    CURVED_2X4_ZP:    ("24309.dat", (4, 2),  3, 270),
    CURVED_2X4_ZM:    ("24309.dat", (4, 2),  3,  90),
    # Curved 4x1 (61678) — long curved primitive. 12 cells.
    CURVED_4X1_XP:    ("61678.dat", (4, 1),  3,   0),
    CURVED_4X1_XM:    ("61678.dat", (4, 1),  3, 180),
    CURVED_4X1_ZP:    ("61678.dat", (4, 1),  3, 270),
    CURVED_4X1_ZM:    ("61678.dat", (4, 1),  3,  90),
    # Curved 6x1 (62361) — longest curved slope. 18 cells.
    CURVED_6X1_XP:    ("62361.dat", (6, 1),  3,   0),
    CURVED_6X1_XM:    ("62361.dat", (6, 1),  3, 180),
    CURVED_6X1_ZP:    ("62361.dat", (6, 1),  3, 270),
    CURVED_6X1_ZM:    ("62361.dat", (6, 1),  3,  90),
}


# ---------------------------------------------------------------------------
# Brick-promotion table
# ---------------------------------------------------------------------------

# For each cell type that has real brick-height parts, list the parts sorted
# largest first so the consolidator prefers 1x2 to two 1x1s. Order is by
# stud area (bigger = fewer parts overall).
_BRICK_TABLE: dict[int, list[tuple[str, tuple[int, int]]]] = {
    PLAIN:     [("3006.dat", (10, 2)),  # Brick 2x10, area 20
                ("3007.dat", (8, 2)),   # Brick 2x8, area 16
                ("2465.dat", (16, 1)),  # Brick 1x16
                ("6112.dat", (12, 1)),  # Brick 1x12
                ("3008.dat", (8, 1)),   # Brick 1x8
                ("3001.dat", (4, 2)),   # Brick 2x4
                ("2456.dat", (6, 2)),   # Brick 2x6, area 12
                ("3003.dat", (2, 2)),   # Brick 2x2
                ("3009.dat", (6, 1)),
                ("3010.dat", (4, 1)),
                ("3622.dat", (3, 1)),
                ("3004.dat", (2, 1)),
                ("3005.dat", (1, 1))],
    MASONRY:   [("98283.dat", (2, 1)), ("15533.dat", (1, 1))],
    LOG:       [("30136.dat", (2, 1))],
    HEADLIGHT: [("4070.dat", (1, 1))],
}

# Fallback plate/tile catalogs — built dynamically from the actual `catalog()`
# so newly-imported parts (Rebrickable additions, custom catalog entries)
# feed the biggest-first packer without needing this file edited.
def _built_preferred(shape: str) -> list[tuple[int, int]]:
    from ..bricks import catalog
    sizes: set[tuple[int, int]] = set()
    for p in catalog():
        if p.shape != shape:
            continue
        for w, d in p.orientations():
            sizes.add((w, d))
    # Sort by area descending, then by longest edge descending as a tiebreak.
    return sorted(sizes, key=lambda wd: (-wd[0] * wd[1], -max(wd)))


_PLATE_PREFERRED: list[tuple[int, int]] = []
_TILE_PREFERRED: list[tuple[int, int]] = []


def _refresh_preferred() -> None:
    """Populate the preferred-size lists from the current catalog. Called
    lazily so imports settle before we read the catalog."""
    global _PLATE_PREFERRED, _TILE_PREFERRED
    if not _PLATE_PREFERRED:
        _PLATE_PREFERRED[:] = _built_preferred("plate")
    if not _TILE_PREFERRED:
        _TILE_PREFERRED[:] = _built_preferred("tile")


def _shape_lookup(shape: str) -> dict[tuple[int, int], Part]:
    out: dict[tuple[int, int], Part] = {}
    for p in catalog():
        if p.shape != shape:
            continue
        for w, d in p.orientations():
            out.setdefault((w, d), p)
    return out


# ---------------------------------------------------------------------------
# Pass 1 — brick promotion
# ---------------------------------------------------------------------------

def _promote_bricks(a: Assembly, processed: np.ndarray) -> list[Placement]:
    """Emit real bricks wherever a vertical run of 3 same-(type,color) cells
    (and horizontally, if the brick's footprint spans >1 stud) can be built."""
    out: list[Placement] = []
    Y, X, Z = a.cell_type.shape

    def _brick_column_stack_matches(y: int, x: int, z: int,
                                     w: int, d: int,
                                     want_type: int, want_color: int) -> bool:
        """3 stacked layers of (w × d) footprint all match?"""
        if y + 3 > Y or x + w > X or z + d > Z:
            return False
        for dy in range(3):
            for dx in range(w):
                for dz in range(d):
                    if processed[y + dy, x + dx, z + dz]:
                        return False
                    if int(a.cell_type[y + dy, x + dx, z + dz]) != want_type:
                        return False
                    if int(a.cell_color[y + dy, x + dx, z + dz]) != want_color:
                        return False
        return True

    # Size-first sweep: for each cell type, try to place the LARGEST brick
    # everywhere it fits, then the next size, etc. Top-left scan-order lost
    # big bricks (2x10, 2x8) at disc-edge cells; going biggest-first-across-
    # positions lets a 2x10 land in the middle of an interior slab even
    # when the top-left cell can't anchor it.
    for cell_type, part_options in _BRICK_TABLE.items():
        parts_by_size: list[tuple[Part, tuple[int, int]]] = []
        for ldraw, fp in part_options:
            p = part_by_ldraw(ldraw)
            if p is not None:
                parts_by_size.append((p, fp))

        for part, (fw, fd) in parts_by_size:
            orientations = ((fw, fd, 0), (fd, fw, 90)) if fw != fd else ((fw, fd, 0),)
            for pw, pd, rot in orientations:
                for y in range(Y):
                    for x in range(X - pw + 1):
                        for z in range(Z - pd + 1):
                            if processed[y, x, z]:
                                continue
                            if int(a.cell_type[y, x, z]) != cell_type:
                                continue
                            color = int(a.cell_color[y, x, z])
                            if color < 0:
                                continue
                            if not _brick_column_stack_matches(
                                    y, x, z, pw, pd, cell_type, color):
                                continue
                            out.append(place_rect(part, x=x, z=z, y=y,
                                                    color_idx=color,
                                                    rotation_deg=rot))
                            for dy in range(3):
                                for dx in range(pw):
                                    for dz in range(pd):
                                        processed[y + dy, x + dx, z + dz] = True
    return out


# ---------------------------------------------------------------------------
# Pass 2 — 2D packing per (y, type, color)
# ---------------------------------------------------------------------------

def _pack_layer(mask: np.ndarray, *, y: int, color_idx: int,
                shape_lookup: dict[tuple[int, int], Part],
                preferred: list[tuple[int, int]]) -> list[Placement]:
    """Pack a boolean X×Z mask with the largest parts that fit — biggest-
    first across ALL positions, not first-position-biggest-first.

    The old top-left-scan-order greedy stuck small parts at the disc's
    curved edge and starved the center of big plates. This size-first sweep
    lays 16×16 wherever it fits, then 12×12 in the leftovers, etc., so a
    round interior actually gets 8×16 / 6×16 / 8×8 plates in its middle
    instead of a swarm of 1×1s."""
    X, Z = mask.shape
    filled = ~mask.copy()
    out: list[Placement] = []
    for pw, pd in preferred:
        part = shape_lookup.get((pw, pd))
        if part is None:
            continue
        rot = 0 if (pw, pd) == part.footprint else 90
        # Sweep every valid origin for this footprint. Placements never
        # overlap because we mark `filled` immediately.
        for x in range(X - pw + 1):
            for z in range(Z - pd + 1):
                if filled[x, z]:
                    continue
                sub = filled[x:x + pw, z:z + pd]
                if sub.any():
                    continue
                out.append(place_rect(part, x=x, z=z, y=y,
                                        color_idx=color_idx,
                                        rotation_deg=rot))
                filled[x:x + pw, z:z + pd] = True
    return out


def _pack_plates_and_tiles(a: Assembly, processed: np.ndarray) -> list[Placement]:
    """Greedy 2D pack the remaining cells (things brick-promotion couldn't
    consume) by (y, type, color) group. Uses the catalog-derived
    biggest-first preferred lists so newly-imported big plates (16x16 etc.)
    are tried before smaller ones."""
    _refresh_preferred()
    plate_lookup = _shape_lookup("plate")
    tile_lookup = _shape_lookup("tile")
    out: list[Placement] = []
    Y, X, Z = a.cell_type.shape

    for y in range(Y):
        cell_type = a.cell_type[y]
        cell_color = a.cell_color[y]
        p_layer = processed[y]

        # Colors that still need placing at this y-level.
        unclaimed = ~p_layer & (cell_type != EMPTY)
        if not unclaimed.any():
            continue

        for t in (TILE, PLAIN, LOG, MASONRY, HEADLIGHT):
            type_mask = unclaimed & (cell_type == t)
            if not type_mask.any():
                continue
            for color_idx in np.unique(cell_color[type_mask]):
                if color_idx < 0:
                    continue
                mask = type_mask & (cell_color == color_idx)
                if not mask.any():
                    continue
                if t == TILE:
                    layer = _pack_layer(mask, y=y, color_idx=int(color_idx),
                                          shape_lookup=tile_lookup,
                                          preferred=_TILE_PREFERRED)
                else:
                    # LOG / MASONRY / HEADLIGHT that couldn't become their real
                    # brick fall back to plain plates. Diversity for the loss
                    # was still counted; the export just picks the safest part.
                    layer = _pack_layer(mask, y=y, color_idx=int(color_idx),
                                          shape_lookup=plate_lookup,
                                          preferred=_PLATE_PREFERRED)
                out.extend(layer)
                processed[y] |= mask
    return out


# ---------------------------------------------------------------------------
# Pass 3 — leftovers (slopes + anything else 1-cell)
# ---------------------------------------------------------------------------

def _emit_leftovers(a: Assembly, processed: np.ndarray) -> list[Placement]:
    """Slopes are the only remaining 1-cell parts (SA slope_align is still
    stubbed, so this should almost never fire in practice)."""
    out: list[Placement] = []
    slope_1x1 = part_by_ldraw("54200.dat")
    plate_1x1 = part_by_ldraw("3024.dat")
    Y, X, Z = a.cell_type.shape
    for y in range(Y):
        for x in range(X):
            for z in range(Z):
                if processed[y, x, z]:
                    continue
                t = int(a.cell_type[y, x, z])
                if t == EMPTY:
                    continue
                color = int(a.cell_color[y, x, z])
                if color < 0:
                    color = 0
                if t in (SLOPE_YP, SLOPE_YM) and slope_1x1 is not None:
                    rot = 180 if t == SLOPE_YM else 0
                    out.append(place_rect(slope_1x1, x=x, z=z, y=y,
                                            color_idx=color, rotation_deg=rot))
                elif plate_1x1 is not None:
                    out.append(place_rect(plate_1x1, x=x, z=z, y=y,
                                            color_idx=color, rotation_deg=0))
                processed[y, x, z] = True
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _emit_tall_parts(a: Assembly, processed: np.ndarray) -> list[Placement]:
    """For each new tall-part cell type, find every column/block that fully
    matches the required footprint × height × color and emit the real LEGO
    part covering it. Any cells that fail to promote fall back to plates
    later — the SA already paid the diversity cost for setting the type."""
    out: list[Placement] = []
    Y, X, Z = a.cell_type.shape
    for cell_type, (ldraw, (fw, fd), h_plates, rot_deg) in _TALL_PARTS.items():
        part = part_by_ldraw(ldraw)
        if part is None:
            continue
        # For rot 90/270 the footprint in world axes is transposed.
        if rot_deg in (0, 180):
            wx, wz = fw, fd
        else:
            wx, wz = fd, fw
        # Sweep all valid origins.
        for y in range(Y - h_plates + 1):
            for x in range(X - wx + 1):
                for z in range(Z - wz + 1):
                    # Fast check: origin cell has this type + is unprocessed.
                    if processed[y, x, z]:
                        continue
                    if int(a.cell_type[y, x, z]) != cell_type:
                        continue
                    c = int(a.cell_color[y, x, z])
                    if c < 0:
                        continue
                    # Full slab check.
                    slab_type = a.cell_type[y:y + h_plates, x:x + wx, z:z + wz]
                    slab_color = a.cell_color[y:y + h_plates, x:x + wx, z:z + wz]
                    slab_proc = processed[y:y + h_plates, x:x + wx, z:z + wz]
                    if slab_proc.any():
                        continue
                    if not (slab_type == cell_type).all():
                        continue
                    if not (slab_color == c).all():
                        continue
                    out.append(place_rect(part, x=x, z=z, y=y,
                                            color_idx=c, rotation_deg=rot_deg))
                    processed[y:y + h_plates, x:x + wx, z:z + wz] = True
    return out


def _recolor_interior(a: Assembly) -> None:
    """Repaint every cell that has all 6 face-neighbours occupied to a single
    structural colour. Interior cells are invisible, so this doesn't change
    what the model looks like from outside — but it lets the packer span
    whole interior regions with the biggest plates instead of being
    fragmented by the surface's color gradient bleeding inward through the
    cell-by-cell voxelization.

    Structural colour = Light Bluish Gray (LDraw 71) — the cheapest,
    most-abundant LEGO structural colour in real Pick-a-Brick.
    """
    from .. import colors
    pal = colors.palette()
    structural_code = 71
    structural_idx = 0
    for i, c in enumerate(pal):
        if c.code == structural_code:
            structural_idx = i
            break

    occ = a.cell_type != EMPTY
    interior = occ.copy()
    interior[:-1, :, :] &= occ[1:, :, :]
    interior[1:, :, :]  &= occ[:-1, :, :]
    interior[:, :-1, :] &= occ[:, 1:, :]
    interior[:, 1:, :]  &= occ[:, :-1, :]
    interior[:, :, :-1] &= occ[:, :, 1:]
    interior[:, :, 1:]  &= occ[:, :, :-1]
    # Boundary cells (touching the grid edge) are NOT interior — they
    # can still be visible when the model sits on a baseplate.
    interior[0, :, :] = False
    interior[-1, :, :] = False
    interior[:, 0, :] = False
    interior[:, -1, :] = False
    interior[:, :, 0] = False
    interior[:, :, -1] = False

    # Only touch "textural" cell types (PLAIN, LOG, MASONRY, TILE) — leave
    # slope-family and specialty parts alone so their blocks still consolidate
    # into their big LEGO parts. Editing one cell inside a 2x4 curved block
    # would break that block's validity check.
    #
    # For interior cells, ALSO reset the TYPE to PLAIN. SA scatters LOG /
    # MASONRY texture through the interior for diversity, but that
    # fragments the same-type groups the packer needs to emit big plates.
    # Interior isn't visible, so a plain plate is fine there.
    only_texture = ((a.cell_type == PLAIN) | (a.cell_type == LOG)
                     | (a.cell_type == MASONRY) | (a.cell_type == TILE))
    recolor_mask = interior & only_texture
    a.cell_color[recolor_mask] = structural_idx
    a.cell_type[recolor_mask] = PLAIN


def assembly_to_placements(a: Assembly) -> list[Placement]:
    """Consolidate the cell assembly into as few LEGO parts as possible."""
    _recolor_interior(a)   # invisible interior → single colour so big plates fit
    processed = np.zeros_like(a.cell_type, dtype=bool)
    out: list[Placement] = []
    # New families first (slopes, panels, windows, doors, brick_tall)
    # so they get first pick on the cells the SA labelled for them.
    out.extend(_emit_tall_parts(a, processed))
    # Then 3-plate brick promotion for plain/log/masonry/headlight.
    out.extend(_promote_bricks(a, processed))
    # Then 2D packing for everything else.
    out.extend(_pack_plates_and_tiles(a, processed))
    # Leftovers (1x1 fallbacks).
    out.extend(_emit_leftovers(a, processed))
    return out
