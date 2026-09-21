"""Proposal generators for simulated annealing.

Each proposer returns a `Move` — a description of one edit to the assembly.
`apply(a, move)` mutates the assembly and returns an inverse move so the SA
driver can revert cheaply if the proposal is rejected.

Moves are **region-aware** on purpose: single-cell swaps encourage speckle
(bad for coherence), so most moves edit a contiguous cluster of cells.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .. import colors
from .loss import (Assembly, EMPTY, PLAIN, LOG, MASONRY, TILE,
                    SLOPE_YP, SLOPE_YM, HEADLIGHT, N_TYPES,
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
                    SLOPE_XZ_FAMILY, SLOPE_INV_FAMILY, SLOPE_CORNER_FAMILY,
                    PANEL_FAMILY, PANEL_TALL_FAMILY, WINDOW_FAMILY, DOOR_FAMILY,
                    CHEESE_FAMILY, CURVED_FAMILY,
                    SLOPE_4X2_FAMILY, CURVED_2X4_FAMILY,
                    CURVED_4X1_FAMILY, CURVED_6X1_FAMILY)
from .target import Target


@dataclass
class Move:
    """One proposed edit. Cells is a list of (y, x, z) tuples all set to the
    same (new_type, new_color). Inverse recorded as parallel lists."""
    cells: list[tuple[int, int, int]]
    new_type: int
    new_color: int
    # populated on apply(): parallel lists of prior state for cheap revert
    prev_types: list[int] = field(default_factory=list)
    prev_colors: list[int] = field(default_factory=list)


def apply(a: Assembly, move: Move) -> Move:
    """Mutate `a` in-place with `move`. Records prev state on the move."""
    move.prev_types = []
    move.prev_colors = []
    for (y, x, z) in move.cells:
        move.prev_types.append(int(a.cell_type[y, x, z]))
        move.prev_colors.append(int(a.cell_color[y, x, z]))
        a.cell_type[y, x, z] = move.new_type
        a.cell_color[y, x, z] = move.new_color if move.new_type != EMPTY else -1
    return move


def revert(a: Assembly, move: Move) -> None:
    """Undo `move` on `a` using the state it recorded during apply()."""
    for (y, x, z), pt, pc in zip(move.cells, move.prev_types, move.prev_colors):
        a.cell_type[y, x, z] = pt
        a.cell_color[y, x, z] = pc


# ---------------------------------------------------------------------------
# Cluster utility
# ---------------------------------------------------------------------------

def _pick_random_occupied_cell(a: Assembly, rng: random.Random) -> tuple[int, int, int] | None:
    occ = np.argwhere(a.occupied())
    if occ.size == 0:
        return None
    idx = rng.randrange(len(occ))
    y, x, z = occ[idx]
    return int(y), int(x), int(z)


def _grow_region(a: Assembly, seed: tuple[int, int, int], *,
                  match_type: bool, match_color: bool,
                  max_size: int, rng: random.Random) -> list[tuple[int, int, int]]:
    """Face-connected flood from `seed` while matching (optionally) type/color
    of the seed. Terminates at max_size."""
    y0, x0, z0 = seed
    seed_type = int(a.cell_type[y0, x0, z0])
    seed_color = int(a.cell_color[y0, x0, z0])
    if seed_type == EMPTY:
        return []
    H, W, D = a.cell_type.shape
    visited = np.zeros((H, W, D), dtype=bool)
    out: list[tuple[int, int, int]] = []
    stack = [seed]
    while stack and len(out) < max_size:
        y, x, z = stack.pop()
        if visited[y, x, z]:
            continue
        visited[y, x, z] = True
        t = int(a.cell_type[y, x, z])
        c = int(a.cell_color[y, x, z])
        if t == EMPTY:
            continue
        if match_type and t != seed_type:
            continue
        if match_color and c != seed_color:
            continue
        out.append((y, x, z))
        neighbors = [(y+1,x,z), (y-1,x,z), (y,x+1,z),
                     (y,x-1,z), (y,x,z+1), (y,x,z-1)]
        rng.shuffle(neighbors)
        for ny, nx_, nz in neighbors:
            if 0 <= ny < H and 0 <= nx_ < W and 0 <= nz < D and not visited[ny, nx_, nz]:
                stack.append((ny, nx_, nz))
    return out


# ---------------------------------------------------------------------------
# Proposal generators
# ---------------------------------------------------------------------------

_TEXTURE_TYPES = (PLAIN, LOG, MASONRY)

# Cell types that carry hard-earned shape information and should NOT be
# overwritten by texture proposers (region_swap / texture_band). If SA
# spent moves placing a curved slope, converting it to a log brick undoes
# the shape gain — which the loss can see but per-move stochastic acceptance
# routinely misses at low temperature.
_SHAPE_PROTECTED = frozenset(
    (TILE,)
    + SLOPE_XZ_FAMILY + SLOPE_INV_FAMILY + SLOPE_CORNER_FAMILY
    + CHEESE_FAMILY + CURVED_FAMILY
    + SLOPE_4X2_FAMILY + CURVED_2X4_FAMILY
    + CURVED_4X1_FAMILY + CURVED_6X1_FAMILY
    + PANEL_FAMILY + PANEL_TALL_FAMILY
    + WINDOW_FAMILY + DOOR_FAMILY
    + (BRICK_TALL, HEADLIGHT)
)


def propose_region_swap(a: Assembly, target: Target, rng: random.Random,
                         max_size: int = 40) -> Move | None:
    """Change a coherent (same type + color) cluster to a new (type, color).

    The most powerful move: shifts a whole material region to a new material,
    boosting diversity globally while keeping coherence locally.

    Refuses seed cells whose type is shape-protected (slopes, tiles, panels,
    windows, doors, brick_tall, headlights) — those types were placed for a
    shape reason and overwriting them with a texture is a net loss.
    """
    seed = _pick_random_occupied_cell(a, rng)
    if seed is None:
        return None
    if int(a.cell_type[seed]) in _SHAPE_PROTECTED:
        return None
    cells = _grow_region(a, seed, match_type=True, match_color=True,
                          max_size=max_size, rng=rng)
    if not cells:
        return None
    # Pick a new (type, color) different from the seed.
    old_type = int(a.cell_type[seed])
    old_color = int(a.cell_color[seed])
    # Prefer texture swaps to preserve shape; occasionally change color slightly.
    new_type_choices = [t for t in _TEXTURE_TYPES if t != old_type]
    new_type = rng.choice(new_type_choices) if new_type_choices else old_type
    # Keep color the same 70% of the time.
    if rng.random() < 0.7:
        new_color = old_color
    else:
        # Sample from target colors nearby.
        y, x, z = seed
        H, W, D = a.cell_type.shape
        y0, y1 = max(0, y-2), min(H, y+3)
        x0, x1 = max(0, x-2), min(W, x+3)
        z0, z1 = max(0, z-2), min(D, z+3)
        nearby = target.color_idx[y0:y1, x0:x1, z0:z1]
        candidates = nearby[nearby >= 0]
        new_color = int(rng.choice(candidates.tolist())) if candidates.size else old_color
    return Move(cells=cells, new_type=new_type, new_color=new_color)


def propose_texture_band(a: Assembly, target: Target, rng: random.Random,
                          max_width: int = 24) -> Move | None:
    """Convert a horizontal (fixed-Y) strip to a texture type — mimics the
    architectural banding humans use (foundation course, cornice, mid-band)."""
    occ = a.occupied()
    if not occ.any():
        return None
    ys = np.where(occ.any(axis=(1, 2)))[0]
    if len(ys) == 0:
        return None
    y = int(rng.choice(ys.tolist()))
    # Pick a random contiguous span in one axis (X or Z).
    axis = rng.choice(["x", "z"])
    if axis == "x":
        row = occ[y, :, :]  # (W, D)
        # Pick a Z, then a span of X.
        zs = np.where(row.any(axis=0))[0]
        if len(zs) == 0:
            return None
        z = int(rng.choice(zs.tolist()))
        xs = np.where(row[:, z])[0]
        if len(xs) == 0:
            return None
        x_start = int(rng.choice(xs.tolist()))
        span = rng.randint(2, max_width)
        cells = [(y, x_start + dx, z) for dx in range(span)
                 if 0 <= x_start + dx < a.cell_type.shape[1]
                 and a.cell_type[y, x_start + dx, z] != EMPTY]
    else:
        row = occ[y, :, :]
        xs = np.where(row.any(axis=1))[0]
        if len(xs) == 0:
            return None
        x = int(rng.choice(xs.tolist()))
        zs = np.where(row[x, :])[0]
        if len(zs) == 0:
            return None
        z_start = int(rng.choice(zs.tolist()))
        span = rng.randint(2, max_width)
        cells = [(y, x, z_start + dz) for dz in range(span)
                 if 0 <= z_start + dz < a.cell_type.shape[2]
                 and a.cell_type[y, x, z_start + dz] != EMPTY]
    # Drop shape-protected cells so a texture band never eats a slope row.
    cells = [c for c in cells if int(a.cell_type[c]) not in _SHAPE_PROTECTED]
    if not cells:
        return None
    new_type = rng.choice(_TEXTURE_TYPES)
    old_colors = [int(a.cell_color[c]) for c in cells]
    # Keep dominant color.
    new_color = max(set(old_colors), key=old_colors.count)
    return Move(cells=cells, new_type=new_type, new_color=new_color)


def _pick_slope_orientation_from_normal(n: np.ndarray,
                                          xz_family, y_family) -> int | None:
    """Given a surface normal (n[0]=Y, n[1]=X, n[2]=Z), pick the slope
    orientation from `xz_family = (SLOPE_XP, SLOPE_XM, SLOPE_ZP, SLOPE_ZM)`
    that best matches the horizontal tilt of the surface. Returns None if the
    normal is close to purely vertical (use `y_family` instead)."""
    ny, nx_, nz = float(n[0]), float(n[1]), float(n[2])
    if abs(nx_) < 0.15 and abs(nz) < 0.15:
        # Almost purely vertical — a horizontal top / bottom surface.
        if abs(ny) > 0.6 and y_family is not None:
            return y_family[0] if ny > 0 else y_family[1]
        return None
    if abs(nx_) >= abs(nz):
        return xz_family[0] if nx_ > 0 else xz_family[1]
    return xz_family[2] if nz > 0 else xz_family[3]


_SLOPE_IDEAL_TYPES = frozenset(
    SLOPE_XZ_FAMILY + SLOPE_INV_FAMILY + SLOPE_CORNER_FAMILY)


def propose_slope_align(a: Assembly, target: Target, rng: random.Random
                          ) -> Move | None:
    """Convert a cell (and its neighbours) to the slope orientation the mesh
    normal at that location prescribes.

    Only fires on cells whose *ideal type* (precomputed from mesh normals in
    `target.ideal_type`) is a slope variant — so every accepted swap
    directly reduces `L_shape_surface`.  We accept overwriting any
    non-EMPTY same-color block, not just PLAIN, so slopes keep landing even
    after texture/tile moves have converted some walls.
    """
    ideal = target.ideal_type
    # Cells whose ideal type wants a slope AND whose current type doesn't
    # already satisfy it. Curved and cheese slopes with matching orientation
    # count as satisfied (they're valid substitutes for a flat 45° slope in
    # the shape loss).
    slope_ideal_mask = np.isin(ideal, list(_SLOPE_IDEAL_TYPES))
    # All slope-family types in the same orientation-index count as valid
    # substitutes so slope_align doesn't try to overwrite a bigger primitive
    # that already sits at this cell.
    def _same_orient_types(orient_idx):
        return [family[orient_idx] for family in (
            CHEESE_FAMILY, CURVED_FAMILY,
            SLOPE_4X2_FAMILY, CURVED_2X4_FAMILY,
            CURVED_4X1_FAMILY, CURVED_6X1_FAMILY,
        )]
    already_satisfied = (
        (a.cell_type == ideal)
        | ((ideal == SLOPE_XP) & np.isin(a.cell_type, _same_orient_types(0)))
        | ((ideal == SLOPE_XM) & np.isin(a.cell_type, _same_orient_types(1)))
        | ((ideal == SLOPE_ZP) & np.isin(a.cell_type, _same_orient_types(2)))
        | ((ideal == SLOPE_ZM) & np.isin(a.cell_type, _same_orient_types(3)))
    )
    needs_swap = slope_ideal_mask & (a.cell_type != EMPTY) & ~already_satisfied
    if not needs_swap.any():
        return None
    cand = np.argwhere(needs_swap)
    idxs = list(range(len(cand)))
    rng.shuffle(idxs)
    Y, X, Z = a.cell_type.shape

    def _valid_block(cells: list[tuple[int, int, int]], c: int) -> bool:
        """A block is valid for a slope move if every cell is non-EMPTY.
        Color mismatches are allowed — SA weighs the color penalty against
        the shape gain automatically."""
        for (cy, cx, cz) in cells:
            if int(a.cell_type[cy, cx, cz]) == EMPTY:
                return False
        return True

    curved_map = {
        SLOPE_XP: CURVED_XP, SLOPE_XM: CURVED_XM,
        SLOPE_ZP: CURVED_ZP, SLOPE_ZM: CURVED_ZM,
    }
    cheese_map = {
        SLOPE_XP: CHEESE_XP, SLOPE_XM: CHEESE_XM,
        SLOPE_ZP: CHEESE_ZP, SLOPE_ZM: CHEESE_ZM,
    }

    for i in idxs[:60]:
        y, x, z = cand[i]
        y, x, z = int(y), int(x), int(z)
        ideal_type = int(ideal[y, x, z])
        c = int(a.cell_color[y, x, z])
        # For 45° slope-ideal cells, try in random order among {curved 2×2,
        # flat 2×1, cheese 1×1}. Randomising means each primitive gets a
        # roughly equal chance of landing — critical for curved surfaces
        # where 2×2 is what actually matches, and per-attempt greedy
        # ordering starves the larger primitive.
        block: list[tuple[int, int, int]] | None = None
        new_type = ideal_type

        if ideal_type in SLOPE_XZ_FAMILY:
            # Prefer curved > flat > cheese. On a curved surface all three
            # count as valid substitutes for the shape loss; larger
            # primitives (curved 2×2 > flat 2×1 > cheese 1×1) mean fewer,
            # nicer LEGO parts.
            attempts = ["curved", "flat", "cheese"]
            for attempt in attempts:
                if attempt == "curved":
                    if x + 2 <= X and z + 2 <= Z and y + 3 <= Y:
                        cb = [(y + dy, x + dx, z + dz)
                              for dy in range(3) for dx in range(2) for dz in range(2)]
                        if _valid_block(cb, c):
                            new_type = curved_map[ideal_type]
                            block = cb
                            break
                elif attempt == "flat":
                    if y + 3 <= Y:
                        if ideal_type in (SLOPE_XP, SLOPE_XM) and x + 2 <= X:
                            cb = [(y + dy, x + dx, z)
                                  for dy in range(3) for dx in range(2)]
                            if _valid_block(cb, c):
                                new_type = ideal_type
                                block = cb
                                break
                        elif ideal_type in (SLOPE_ZP, SLOPE_ZM) and z + 2 <= Z:
                            cb = [(y + dy, x, z + dz)
                                  for dy in range(3) for dz in range(2)]
                            if _valid_block(cb, c):
                                new_type = ideal_type
                                block = cb
                                break
                elif attempt == "cheese":
                    if y + 2 <= Y:
                        cb = [(y + dy, x, z) for dy in range(2)]
                        if _valid_block(cb, c):
                            new_type = cheese_map[ideal_type]
                            block = cb
                            break
        elif ideal_type in SLOPE_INV_FAMILY:
            if y + 3 <= Y:
                if ideal_type in (SLOPE_INV_XP, SLOPE_INV_XM) and x + 2 <= X:
                    cb = [(y + dy, x + dx, z)
                          for dy in range(3) for dx in range(2)]
                    if _valid_block(cb, c):
                        block = cb
                elif z + 2 <= Z:
                    cb = [(y + dy, x, z + dz)
                          for dy in range(3) for dz in range(2)]
                    if _valid_block(cb, c):
                        block = cb
        elif ideal_type in SLOPE_CORNER_FAMILY:
            if x + 2 <= X and z + 2 <= Z and y + 3 <= Y:
                cb = [(y + dy, x + dx, z + dz)
                      for dy in range(3) for dx in range(2) for dz in range(2)]
                if _valid_block(cb, c):
                    block = cb

        if block is None:
            continue
        return Move(cells=block, new_type=new_type, new_color=c)
    return None


def propose_panel(a: Assembly, target: Target, rng: random.Random
                    ) -> Move | None:
    """Replace a vertical wall region with a PANEL orientation (1-stud thick
    face). Panel cells are 3 plates tall; this fires for triples of stacked
    same-color cells whose target normal is horizontal."""
    Y, X, Z = a.cell_type.shape
    # Surface cells with horizontal normal and currently PLAIN.
    n = target.normal
    horiz = (np.abs(n[..., 0]) < 0.3) & target.surface & (a.cell_type == PLAIN)
    if not horiz.any():
        return None
    cand = np.argwhere(horiz)
    rng.shuffle(cand.tolist() if isinstance(cand, list) else list(range(len(cand))))
    for i in range(min(60, len(cand))):
        y, x, z = cand[i]
        y, x, z = int(y), int(x), int(z)
        # Need 3 same-color cells stacked at (y, y+1, y+2).
        if y + 2 >= Y:
            continue
        c = int(a.cell_color[y, x, z])
        if any(int(a.cell_color[y + dy, x, z]) != c
                or int(a.cell_type[y + dy, x, z]) != PLAIN
                for dy in range(3)):
            continue
        nrm = n[y, x, z]
        new_type = _pick_slope_orientation_from_normal(nrm, PANEL_FAMILY, None)
        if new_type is None:
            continue
        # Panel 1x2x1 needs 2 studs wide × 3 plates tall (footprint depends
        # on which face the panel is presenting).
        if new_type in (PANEL_XP, PANEL_XM):
            if x + 2 > X:
                continue
            cells = [(y + dy, x + dx, z)
                      for dy in range(3) for dx in range(2)]
        else:
            if z + 2 > Z:
                continue
            cells = [(y + dy, x, z + dz)
                      for dy in range(3) for dz in range(2)]
        if any(int(a.cell_type[cy, cx, cz]) != PLAIN
                or int(a.cell_color[cy, cx, cz]) != c
                for (cy, cx, cz) in cells):
            continue
        return Move(cells=cells, new_type=new_type, new_color=c)
    return None


def propose_window(a: Assembly, target: Target, rng: random.Random
                     ) -> Move | None:
    """Replace a 2-wide × 6-tall wall span with a WINDOW. Target must be a
    panel-like wall region (horizontal normal) and the span must not clip
    the top of the wall."""
    Y, X, Z = a.cell_type.shape
    horiz = (np.abs(target.normal[..., 0]) < 0.3) & target.surface
    if not horiz.any():
        return None
    cand = np.argwhere(horiz & (a.cell_type == PLAIN))
    if len(cand) == 0:
        return None
    idxs = list(range(len(cand)))
    rng.shuffle(idxs)
    for i in idxs[:80]:
        y, x, z = cand[i]
        y, x, z = int(y), int(x), int(z)
        # Try WINDOW oriented per normal.
        nrm = target.normal[y, x, z]
        new_type = _pick_slope_orientation_from_normal(nrm, WINDOW_FAMILY, None)
        if new_type is None:
            continue
        # Determine 2×6 region shape from orientation.
        if new_type in (WINDOW_XP, WINDOW_XM):
            span_axis = "z"
        else:
            span_axis = "x"
        if span_axis == "x":
            if x + 2 > X or y + 6 > Y:
                continue
            cells = [(y + dy, x + dx, z) for dy in range(6) for dx in range(2)]
        else:
            if z + 2 > Z or y + 6 > Y:
                continue
            cells = [(y + dy, x, z + dz) for dy in range(6) for dz in range(2)]
        # All must be currently PLAIN + same color.
        c = int(a.cell_color[y, x, z])
        if any(int(a.cell_type[cy, cx, cz]) != PLAIN
                or int(a.cell_color[cy, cx, cz]) != c
                for (cy, cx, cz) in cells):
            continue
        return Move(cells=cells, new_type=new_type, new_color=c)
    return None


def propose_door(a: Assembly, target: Target, rng: random.Random
                   ) -> Move | None:
    """Replace a 4-wide × 15-tall wall region starting at ground level with
    a DOOR. Uses a much stricter set of preconditions than propose_window."""
    Y, X, Z = a.cell_type.shape
    if Y < 15:
        return None
    # Ground-level cells (y == 0) that are on a wall surface.
    ground_wall = ((np.abs(target.normal[0, ..., 0]) < 0.3)
                    & target.surface[0]
                    & (a.cell_type[0] == PLAIN))
    if not ground_wall.any():
        return None
    cand = np.argwhere(ground_wall)
    idxs = list(range(len(cand)))
    rng.shuffle(idxs)
    for i in idxs[:30]:
        x, z = cand[i]
        x, z = int(x), int(z)
        nrm = target.normal[0, x, z]
        new_type = _pick_slope_orientation_from_normal(nrm, DOOR_FAMILY, None)
        if new_type is None:
            continue
        if new_type in (DOOR_XP, DOOR_XM):
            if z + 4 > Z:
                continue
            cells = [(dy, x, z + dz) for dy in range(15) for dz in range(4)]
        else:
            if x + 4 > X:
                continue
            cells = [(dy, x + dx, z) for dy in range(15) for dx in range(4)]
        c = int(a.cell_color[0, x, z])
        if any(int(a.cell_type[cy, cx, cz]) != PLAIN
                or int(a.cell_color[cy, cx, cz]) != c
                for (cy, cx, cz) in cells):
            continue
        return Move(cells=cells, new_type=new_type, new_color=c)
    return None


def propose_brick_tall(a: Assembly, target: Target, rng: random.Random
                         ) -> Move | None:
    """Promote a 15-plate-tall column of same-color PLAIN cells to a
    BRICK_TALL (Brick 1x2x5)."""
    Y, X, Z = a.cell_type.shape
    if Y < 15:
        return None
    # Any (x, z) column where the bottom cell is PLAIN and the whole column
    # top-to-bottom is one color.
    for _ in range(60):
        x = rng.randrange(X - 1)  # need x+1 available for 2-stud width
        z = rng.randrange(Z)
        if int(a.cell_type[0, x, z]) != PLAIN:
            continue
        c = int(a.cell_color[0, x, z])
        # Match 15-plate stack, 2 studs wide.
        ok = True
        for dy in range(15):
            for dx in range(2):
                if (int(a.cell_type[dy, x + dx, z]) != PLAIN
                        or int(a.cell_color[dy, x + dx, z]) != c):
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        cells = [(dy, x + dx, z) for dy in range(15) for dx in range(2)]
        return Move(cells=cells, new_type=BRICK_TALL, new_color=c)
    return None


def propose_tile_top(a: Assembly, target: Target, rng: random.Random
                       ) -> Move | None:
    """Where a cell is on the +Y (top) surface (no cell above), replace PLAIN
    with TILE. Improves smoothness at zero shape cost."""
    occ = a.occupied()
    if not occ.any():
        return None
    # Top surface: occupied and cell above is empty (or out of bounds).
    top_surface = np.zeros_like(occ)
    top_surface[:-1] = occ[:-1] & ~occ[1:]
    top_surface[-1] = occ[-1]
    plain_top = top_surface & (a.cell_type == PLAIN)
    if not plain_top.any():
        return None
    cand = np.argwhere(plain_top)
    y, x, z = cand[rng.randrange(len(cand))]
    return Move(cells=[(int(y), int(x), int(z))],
                new_type=TILE, new_color=int(a.cell_color[y, x, z]))


def propose_recolor(a: Assembly, target: Target, rng: random.Random
                      ) -> Move | None:
    """Recolor a small region to the target color at that region's cells."""
    seed = _pick_random_occupied_cell(a, rng)
    if seed is None:
        return None
    cells = _grow_region(a, seed, match_type=False, match_color=True,
                          max_size=12, rng=rng)
    if not cells:
        return None
    # Drop shape-protected cells so recolor doesn't split a slope block by
    # painting one of its cells.
    cells = [c for c in cells if int(a.cell_type[c]) not in _SHAPE_PROTECTED]
    if not cells:
        return None
    # Take the target color at the seed if it's occupied there.
    y, x, z = seed
    tgt_color = int(target.color_idx[y, x, z])
    if tgt_color < 0:
        return None
    old_type = int(a.cell_type[seed])
    if int(a.cell_color[seed]) == tgt_color:
        return None
    return Move(cells=cells, new_type=old_type, new_color=tgt_color)


# The set of proposers the SA driver samples from — weighted by roughly how
# useful each has been in practice.
def propose_ring_curved(a: Assembly, target: Target, rng: random.Random
                          ) -> Move | None:
    """Tile a full latitude ring with curved slopes in one atomic move.

    Picks a Y level and one slope orientation, finds every 2×2 patch at that
    Y whose four cells share (a) that same slope-ideal orientation and
    (b) a colour with the majority of the ring. Places a 12-cell CURVED
    block at each qualifying patch. The whole thing lands as a single Move
    so SA either accepts a whole ring of curved slopes or reverts to plate.

    This is what actually gets a LEGO sphere off the ground: instead of the
    SA placing one curved patch every hundred iterations, a single accepted
    ring move lays down 4–20 curved slopes at once.
    """
    ideal = target.ideal_type
    Y, X, Z = a.cell_type.shape

    # Score each Y level by how many slope-ideal cells sit there. Skip
    # levels with too few cells to make a ring worthwhile.
    slope_ideal_set = list(_SLOPE_IDEAL_TYPES)
    counts = np.array([int(np.isin(ideal[y], slope_ideal_set).sum())
                        for y in range(Y)])
    if counts.max() < 8:
        return None
    # y+3 must fit so the CURVED block has 3 plates of headroom above.
    valid_ys = [y for y in range(Y - 3) if counts[y] >= 8]
    if not valid_ys:
        return None
    # Weight the pick toward busier rings.
    weights = [int(counts[y]) for y in valid_ys]
    y = rng.choices(valid_ys, weights=weights, k=1)[0]

    # Pick a slope orientation to tile in this ring.
    orientation = rng.choice([SLOPE_XP, SLOPE_XM, SLOPE_ZP, SLOPE_ZM])
    curved_map = {SLOPE_XP: CURVED_XP, SLOPE_XM: CURVED_XM,
                   SLOPE_ZP: CURVED_ZP, SLOPE_ZM: CURVED_ZM}
    curved_type = curved_map[orientation]

    # Enumerate 2×2 patches at this Y that all four want `orientation`.
    # Step by 2 so patches don't overlap — otherwise the export consolidator
    # only emits one 15068 per overlapping region and half the effort is
    # wasted. Try both aligned offsets (0-start and 1-start) so ring-tiling
    # doesn't chronically miss cells stuck at odd positions.
    x_offset = rng.choice([0, 1])
    z_offset = rng.choice([0, 1])
    patches: list[tuple[int, int]] = []
    for x in range(x_offset, X - 1, 2):
        for z in range(z_offset, Z - 1, 2):
            if all(int(ideal[y, x + dx, z + dz]) == orientation
                    for dx in range(2) for dz in range(2)):
                # Skip patches whose 12-cell block would step on a
                # shape-protected cell (a bigger slope primitive already
                # placed there). Overwriting them destroys the big block.
                if all(int(a.cell_type[y + dy, x + dx, z + dz]) != EMPTY
                        and int(a.cell_type[y + dy, x + dx, z + dz]) not in _SHAPE_PROTECTED
                        for dy in range(3) for dx in range(2) for dz in range(2)):
                    patches.append((x, z))
    if len(patches) < 2:
        return None

    # Pick the dominant colour at the origin cells so the ring lands as one
    # visually coherent band. If multiple colours exist we drop the patches
    # whose seed cell has a minority colour — SA's colour term already
    # rewards using the "right" colour at every cell.
    from collections import Counter
    seed_colors = Counter(int(a.cell_color[y, x, z]) for (x, z) in patches)
    dominant_color, _ = seed_colors.most_common(1)[0]
    kept_patches = [(x, z) for (x, z) in patches
                     if int(a.cell_color[y, x, z]) == dominant_color]
    if len(kept_patches) < 2:
        return None

    # Assemble all 12 cells per patch, dedupe.
    cells_set: set[tuple[int, int, int]] = set()
    for (x, z) in kept_patches:
        for dy in range(3):
            for dx in range(2):
                for dz in range(2):
                    cells_set.add((y + dy, x + dx, z + dz))
    cells = list(cells_set)
    return Move(cells=cells, new_type=curved_type, new_color=dominant_color)


PROPOSERS = [
    (propose_region_swap, 3),
    (propose_texture_band, 2),
    (propose_tile_top, 2),
    (propose_recolor, 2),
    (propose_slope_align, 8),   # boosted — shape_surface can only fall via these
    (propose_panel, 2),
    (propose_window, 1),
    (propose_door, 1),
    (propose_brick_tall, 1),
    (propose_ring_curved, 4),   # new: atomically tiles a whole latitude
]


def sample_move(a: Assembly, target: Target, rng: random.Random) -> Move | None:
    """Draw one proposer weighted by its usefulness weight and call it."""
    total = sum(w for _, w in PROPOSERS)
    r = rng.random() * total
    acc = 0
    for fn, w in PROPOSERS:
        acc += w
        if r <= acc:
            return fn(a, target, rng)
    return None
