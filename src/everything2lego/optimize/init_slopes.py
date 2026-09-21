"""Greedy slope pre-fill — truly biggest first.

Ordered passes over the surface, largest slope primitives first. Each
pass claims cells before smaller primitives get a chance to fragment
the region.  Order (by cells per instance):

    1. CURVED 2×4 (24309)     — 24 cells, curved face
    2. SLOPE  4×2 (3037)      — 24 cells, flat 45°
    3. CURVED 6×1 (62361)     — 18 cells, long curved
    4. CURVED 4×1 (61678)     — 12 cells, medium curved
    5. CURVED 2×2 (15068)     — 12 cells, square curved
    6. CORNER 2×2 (3045)      — 12 cells, diagonal corner
    7. SLOPE  2×1 (3040)      — 6  cells, small flat
    8. INV    2×1 (3665)      — 6  cells, overhang
    9. CHEESE 1×1 (54200)     — 2  cells, last resort

Every pass validates the block (all cells occupied, none yet claimed by
an earlier pass) before placing.
"""

from __future__ import annotations

import numpy as np

from .loss import (
    Assembly, EMPTY, PLAIN, TILE,
    SLOPE_XP, SLOPE_XM, SLOPE_ZP, SLOPE_ZM,
    SLOPE_INV_XP, SLOPE_INV_XM, SLOPE_INV_ZP, SLOPE_INV_ZM,
    SLOPE_CORNER_PP, SLOPE_CORNER_PN, SLOPE_CORNER_NP, SLOPE_CORNER_NN,
    CURVED_XP, CURVED_XM, CURVED_ZP, CURVED_ZM,
    CHEESE_XP, CHEESE_XM, CHEESE_ZP, CHEESE_ZM,
    SLOPE_4X2_XP, SLOPE_4X2_XM, SLOPE_4X2_ZP, SLOPE_4X2_ZM,
    CURVED_2X4_XP, CURVED_2X4_XM, CURVED_2X4_ZP, CURVED_2X4_ZM,
    CURVED_4X1_XP, CURVED_4X1_XM, CURVED_4X1_ZP, CURVED_4X1_ZM,
    CURVED_6X1_XP, CURVED_6X1_XM, CURVED_6X1_ZP, CURVED_6X1_ZM,
    SLOPE_XZ_FAMILY, SLOPE_INV_FAMILY, SLOPE_CORNER_FAMILY,
)
from .target import Target


# Maps SLOPE_XZ_FAMILY ideal -> the big-primitive family in the same
# orientation direction. Index cycle: 0=XP, 1=XM, 2=ZP, 3=ZM.
def _mapped(family, ideal_t: int) -> int:
    idx = SLOPE_XZ_FAMILY.index(ideal_t)
    return family[idx]


def _block_valid(a: Assembly, cells) -> bool:
    for (cy, cx, cz) in cells:
        t = int(a.cell_type[cy, cx, cz])
        if t == EMPTY:
            return False
        if t != PLAIN and t != TILE:
            return False
    return True


def _set_block(a: Assembly, cells, new_type: int, new_color: int) -> None:
    for (cy, cx, cz) in cells:
        a.cell_type[cy, cx, cz] = new_type
        a.cell_color[cy, cx, cz] = new_color


def _cells_rect(y: int, x: int, z: int, w: int, d: int, h: int):
    return [(y + dy, x + dx, z + dz)
             for dy in range(h) for dx in range(w) for dz in range(d)]


def _rect_dims_for_orient(orient_t: int, wide: int, thin: int) -> tuple[int, int]:
    """For a slope oriented in +x/-x, the "wide" axis is X; for +z/-z it's Z.
    Returns (w_x, d_z) for the block extent."""
    if orient_t in (SLOPE_XP, SLOPE_XM, SLOPE_INV_XP, SLOPE_INV_XM):
        return (wide, thin)
    return (thin, wide)


def _pass_slope_family(a: Assembly, target: Target, surface_coords,
                        target_family: tuple[int, ...],
                        wide: int, thin: int) -> None:
    """Generic pass: for every surface cell whose ideal is a SLOPE_XZ
    orientation, try placing the specified `target_family` primitive
    (wide × thin footprint, 3 plates tall) with the block *extending into
    the shape* — not outside it. For SLOPE_XP (surface faces +X, material
    extends -X), the block starts at `x_seed - w + 1` so its +X edge sits
    at the surface. Same principle for +Z / -Z orientations."""
    Y, X, Z = a.cell_type.shape
    ideal = target.ideal_type
    for y, x, z in surface_coords:
        y, x, z = int(y), int(x), int(z)
        ideal_t = int(ideal[y, x, z])
        if ideal_t not in SLOPE_XZ_FAMILY:
            continue
        w_x, d_z = _rect_dims_for_orient(ideal_t, wide, thin)
        # Anchor by orientation so the block extends INTO the shape.
        if ideal_t == SLOPE_XP:
            ax, az = x - w_x + 1, z
        elif ideal_t == SLOPE_XM:
            ax, az = x, z
        elif ideal_t == SLOPE_ZP:
            ax, az = x, z - d_z + 1
        else:  # SLOPE_ZM
            ax, az = x, z
        if ax < 0 or az < 0 or ax + w_x > X or az + d_z > Z or y + 3 > Y:
            continue
        block = _cells_rect(y, ax, az, w_x, d_z, 3)
        if not _block_valid(a, block):
            continue
        c = int(a.cell_color[y, x, z])
        if c < 0:
            continue
        _set_block(a, block, _mapped(target_family, ideal_t), c)


def initial_assembly_with_slopes(target: Target) -> Assembly:
    """Truly biggest-first greedy pre-fill."""
    a = Assembly.from_target(target)
    Y, X, Z = target.shape
    ideal = target.ideal_type
    surface_coords = np.argwhere(target.surface).tolist()

    # Pass 1: CURVED 2×4 (24309), 24-cell footprint
    _pass_slope_family(a, target, surface_coords, CURVED_2X4_FAMILY := (
        CURVED_2X4_XP, CURVED_2X4_XM, CURVED_2X4_ZP, CURVED_2X4_ZM),
                        wide=4, thin=2)

    # Pass 2: SLOPE 4×2 (3037), 24-cell flat
    _pass_slope_family(a, target, surface_coords, SLOPE_4X2_FAMILY := (
        SLOPE_4X2_XP, SLOPE_4X2_XM, SLOPE_4X2_ZP, SLOPE_4X2_ZM),
                        wide=4, thin=2)

    # Pass 3: CURVED 6×1 (62361), 18-cell long
    _pass_slope_family(a, target, surface_coords, CURVED_6X1_FAMILY := (
        CURVED_6X1_XP, CURVED_6X1_XM, CURVED_6X1_ZP, CURVED_6X1_ZM),
                        wide=6, thin=1)

    # Pass 4: CURVED 4×1 (61678), 12-cell
    _pass_slope_family(a, target, surface_coords, CURVED_4X1_FAMILY := (
        CURVED_4X1_XP, CURVED_4X1_XM, CURVED_4X1_ZP, CURVED_4X1_ZM),
                        wide=4, thin=1)

    # Pass 5: CURVED 2×2 (15068), 12-cell square
    _pass_slope_family(a, target, surface_coords,
                        (CURVED_XP, CURVED_XM, CURVED_ZP, CURVED_ZM),
                        wide=2, thin=2)

    # Pass 6: SLOPE_CORNER 2×2 (3045)
    for y, x, z in surface_coords:
        y, x, z = int(y), int(x), int(z)
        ideal_t = int(ideal[y, x, z])
        if ideal_t not in SLOPE_CORNER_FAMILY:
            continue
        if x + 2 > X or z + 2 > Z or y + 3 > Y:
            continue
        block = _cells_rect(y, x, z, 2, 2, 3)
        if not _block_valid(a, block):
            continue
        c = int(a.cell_color[y, x, z])
        if c < 0:
            continue
        _set_block(a, block, ideal_t, c)

    # Pass 7: SLOPE 2×1 (3040), 6-cell
    _pass_slope_family(a, target, surface_coords, SLOPE_XZ_FAMILY,
                        wide=2, thin=1)

    # Pass 8: SLOPE_INV 2×1 (3665) for overhangs
    Y, X, Z = a.cell_type.shape
    for y, x, z in surface_coords:
        y, x, z = int(y), int(x), int(z)
        ideal_t = int(ideal[y, x, z])
        if ideal_t not in SLOPE_INV_FAMILY:
            continue
        if y + 3 > Y:
            continue
        if ideal_t in (SLOPE_INV_XP, SLOPE_INV_XM):
            if x + 2 > X:
                continue
            block = _cells_rect(y, x, z, 2, 1, 3)
        else:
            if z + 2 > Z:
                continue
            block = _cells_rect(y, x, z, 1, 2, 3)
        if not _block_valid(a, block):
            continue
        c = int(a.cell_color[y, x, z])
        if c < 0:
            continue
        _set_block(a, block, ideal_t, c)

    # Pass 9: CHEESE 1×1 (54200) — last-resort tiling
    cheese_map = {
        SLOPE_XP: CHEESE_XP, SLOPE_XM: CHEESE_XM,
        SLOPE_ZP: CHEESE_ZP, SLOPE_ZM: CHEESE_ZM,
    }
    for y, x, z in surface_coords:
        y, x, z = int(y), int(x), int(z)
        ideal_t = int(ideal[y, x, z])
        if ideal_t not in SLOPE_XZ_FAMILY:
            continue
        if y + 2 > Y:
            continue
        block = [(y + dy, x, z) for dy in range(2)]
        if not _block_valid(a, block):
            continue
        c = int(a.cell_color[y, x, z])
        if c < 0:
            continue
        _set_block(a, block, cheese_map[ideal_t], c)

    return a
