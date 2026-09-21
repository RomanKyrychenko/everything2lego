"""Composite loss L(A) over cell assemblies.

Assembly representation
-----------------------
At the optimizer's granularity, an assembly is a triple of int arrays over
the same (Y, X, Z) grid as the Target:

    cell_type   : int8  — one of {EMPTY, PLAIN, LOG, MASONRY, TILE,
                                    SLOPE_YP, SLOPE_YM, HEADLIGHT}
    cell_color  : int16 — palette index; -1 where EMPTY
    (implicit)  — orientation is derivable from the neighborhood + normal

Cell type IDs are integers so we can numpy them. Post-processing maps each
cell into an actual LDraw part (see `assembly_to_placements`).

Loss terms
----------
    L = w_shape     * (1 - IoU(A.occupied, target.occupied))
      + w_color     * mean_Lab_error(surface cells)
      + w_diversity * (-normalized_entropy(type_histogram_over_occupied))
      + w_coherence * TV(face-adjacent cells)
      + w_stability * (n_connected_components - 1) / normalizer
      + w_count     * expected_part_count / N_ref

All terms are on ~[0,1] scale so weights read as tradeoffs directly.

Delta computation
-----------------
`delta_swap_cell(state, target, idx, new_type, new_color) -> float`
returns dL from swapping a single cell in O(neighborhood size). The SA
driver uses this to skip full recomputes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .. import colors
from .target import Target


# ---------------------------------------------------------------------------
# Cell-type table
# ---------------------------------------------------------------------------

# Every LEGO part family in the catalog is reachable through a cell type.
# Grouping by family:
#   PLATE-height (1 plate):    PLAIN, TILE
#   BRICK-height (3 plates):   BRICK_PLAIN, LOG, MASONRY, HEADLIGHT,
#                              SLOPE_XP/XM/ZP/ZM, SLOPE_YP/YM (kept for compat),
#                              SLOPE_INV, SLOPE_CORNER, PANEL_1H_XP/XM/ZP/ZM
#   BRICK-2 height (6 plates): PANEL_2H_XP/XM/ZP/ZM, WINDOW
#   BRICK-3 height (9 plates): WINDOW_TALL
#   BRICK-5 height (15 plates): BRICK_TALL, DOOR
#
# Orientation is baked into the type so cell-swap moves don't need a separate
# rotation array — the export mapping resolves type → (part, rotation_deg).
(EMPTY,
 PLAIN, TILE,
 LOG, MASONRY, HEADLIGHT,
 SLOPE_YP, SLOPE_YM,                    # kept for backwards-compat with old runs
 SLOPE_XP, SLOPE_XM, SLOPE_ZP, SLOPE_ZM,
 SLOPE_INV_XP, SLOPE_INV_XM, SLOPE_INV_ZP, SLOPE_INV_ZM,
 SLOPE_CORNER_PP, SLOPE_CORNER_PN, SLOPE_CORNER_NP, SLOPE_CORNER_NN,
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
 CURVED_6X1_XP, CURVED_6X1_XM, CURVED_6X1_ZP, CURVED_6X1_ZM) = range(61)

TYPE_NAMES = {
    EMPTY: "empty",
    PLAIN: "plain", TILE: "tile",
    LOG: "log", MASONRY: "masonry", HEADLIGHT: "headlight",
    SLOPE_YP: "slope_yp", SLOPE_YM: "slope_ym",
    SLOPE_XP: "slope_xp", SLOPE_XM: "slope_xm",
    SLOPE_ZP: "slope_zp", SLOPE_ZM: "slope_zm",
    SLOPE_INV_XP: "slope_inv_xp", SLOPE_INV_XM: "slope_inv_xm",
    SLOPE_INV_ZP: "slope_inv_zp", SLOPE_INV_ZM: "slope_inv_zm",
    SLOPE_CORNER_PP: "slope_corner_pp", SLOPE_CORNER_PN: "slope_corner_pn",
    SLOPE_CORNER_NP: "slope_corner_np", SLOPE_CORNER_NN: "slope_corner_nn",
    PANEL_XP: "panel_xp", PANEL_XM: "panel_xm",
    PANEL_ZP: "panel_zp", PANEL_ZM: "panel_zm",
    PANEL_TALL_XP: "panel_tall_xp", PANEL_TALL_XM: "panel_tall_xm",
    PANEL_TALL_ZP: "panel_tall_zp", PANEL_TALL_ZM: "panel_tall_zm",
    WINDOW_XP: "window_xp", WINDOW_XM: "window_xm",
    WINDOW_ZP: "window_zp", WINDOW_ZM: "window_zm",
    DOOR_XP: "door_xp", DOOR_XM: "door_xm",
    DOOR_ZP: "door_zp", DOOR_ZM: "door_zm",
    BRICK_TALL: "brick_tall",
    CHEESE_XP: "cheese_xp", CHEESE_XM: "cheese_xm",
    CHEESE_ZP: "cheese_zp", CHEESE_ZM: "cheese_zm",
    CURVED_XP: "curved_xp", CURVED_XM: "curved_xm",
    CURVED_ZP: "curved_zp", CURVED_ZM: "curved_zm",
    SLOPE_4X2_XP: "slope_4x2_xp", SLOPE_4X2_XM: "slope_4x2_xm",
    SLOPE_4X2_ZP: "slope_4x2_zp", SLOPE_4X2_ZM: "slope_4x2_zm",
    CURVED_2X4_XP: "curved_2x4_xp", CURVED_2X4_XM: "curved_2x4_xm",
    CURVED_2X4_ZP: "curved_2x4_zp", CURVED_2X4_ZM: "curved_2x4_zm",
    CURVED_4X1_XP: "curved_4x1_xp", CURVED_4X1_XM: "curved_4x1_xm",
    CURVED_4X1_ZP: "curved_4x1_zp", CURVED_4X1_ZM: "curved_4x1_zm",
    CURVED_6X1_XP: "curved_6x1_xp", CURVED_6X1_XM: "curved_6x1_xm",
    CURVED_6X1_ZP: "curved_6x1_zp", CURVED_6X1_ZM: "curved_6x1_zm",
}
N_TYPES = len(TYPE_NAMES)

# For proposers: group orientations of one family so proposers can pick from
# a family and choose orientation by mesh normal.
SLOPE_XZ_FAMILY = (SLOPE_XP, SLOPE_XM, SLOPE_ZP, SLOPE_ZM)
CHEESE_FAMILY = (CHEESE_XP, CHEESE_XM, CHEESE_ZP, CHEESE_ZM)
CURVED_FAMILY = (CURVED_XP, CURVED_XM, CURVED_ZP, CURVED_ZM)
SLOPE_4X2_FAMILY = (SLOPE_4X2_XP, SLOPE_4X2_XM, SLOPE_4X2_ZP, SLOPE_4X2_ZM)
CURVED_2X4_FAMILY = (CURVED_2X4_XP, CURVED_2X4_XM, CURVED_2X4_ZP, CURVED_2X4_ZM)
CURVED_4X1_FAMILY = (CURVED_4X1_XP, CURVED_4X1_XM, CURVED_4X1_ZP, CURVED_4X1_ZM)
CURVED_6X1_FAMILY = (CURVED_6X1_XP, CURVED_6X1_XM, CURVED_6X1_ZP, CURVED_6X1_ZM)
SLOPE_INV_FAMILY = (SLOPE_INV_XP, SLOPE_INV_XM, SLOPE_INV_ZP, SLOPE_INV_ZM)
SLOPE_CORNER_FAMILY = (SLOPE_CORNER_PP, SLOPE_CORNER_PN,
                        SLOPE_CORNER_NP, SLOPE_CORNER_NN)
PANEL_FAMILY = (PANEL_XP, PANEL_XM, PANEL_ZP, PANEL_ZM)
PANEL_TALL_FAMILY = (PANEL_TALL_XP, PANEL_TALL_XM, PANEL_TALL_ZP, PANEL_TALL_ZM)
WINDOW_FAMILY = (WINDOW_XP, WINDOW_XM, WINDOW_ZP, WINDOW_ZM)
DOOR_FAMILY = (DOOR_XP, DOOR_XM, DOOR_ZP, DOOR_ZM)

# The visible types (all non-empty) — used for entropy normalization.
VISIBLE_TYPES = tuple(t for t in range(N_TYPES) if t != EMPTY)


# ---------------------------------------------------------------------------
# Assembly + weights
# ---------------------------------------------------------------------------

@dataclass
class Assembly:
    """Cell-grid assembly the optimizer mutates in place."""
    cell_type: np.ndarray   # (Y, X, Z) int8
    cell_color: np.ndarray  # (Y, X, Z) int16 (palette index, -1 when EMPTY)

    @classmethod
    def empty_like(cls, target: Target) -> "Assembly":
        return cls(
            cell_type=np.zeros(target.shape, dtype=np.int8),
            cell_color=np.full(target.shape, -1, dtype=np.int16),
        )

    @classmethod
    def from_target(cls, target: Target, default_type: int = PLAIN) -> "Assembly":
        """Baseline assembly: exactly match target occupancy with all-PLAIN
        bricks in the target color. This is the starting point SA improves."""
        cell_type = np.where(target.occupied, default_type, EMPTY).astype(np.int8)
        cell_color = target.color_idx.copy()
        return cls(cell_type=cell_type, cell_color=cell_color)

    def occupied(self) -> np.ndarray:
        return self.cell_type != EMPTY


@dataclass
class Weights:
    """Loss weights. Tune from the CLI or programmatically.

    Shape is split in two terms:
      - `shape_volume` — 1 − IoU of assembly vs voxelized target
      - `shape_surface` — mean distance from LEGO surface cells to *the
        original mesh surface* (in a normalised mm scale). This term is
        already non-zero at initialisation on any mesh that isn't a perfect
        LEGO-lattice shape — that's the voxelization error the volume term
        can't see.
    """
    shape_volume: float = 0.7
    shape_surface: float = 1.0
    color: float = 0.3
    diversity: float = 0.05
    coherence: float = 0.4
    stability: float = 0.1
    count: float = 0.001


@dataclass
class LossBreakdown:
    """Debug view of L(A) — each component before weighting."""
    shape_volume: float = 0.0
    shape_surface: float = 0.0
    color: float = 0.0
    diversity: float = 0.0   # already negative (reward)
    coherence: float = 0.0
    stability: float = 0.0
    count: float = 0.0
    total: float = 0.0

    def weighted(self, w: Weights) -> "LossBreakdown":
        b = LossBreakdown(
            shape_volume=w.shape_volume * self.shape_volume,
            shape_surface=w.shape_surface * self.shape_surface,
            color=w.color * self.color,
            diversity=w.diversity * self.diversity,
            coherence=w.coherence * self.coherence,
            stability=w.stability * self.stability,
            count=w.count * self.count,
        )
        b.total = (b.shape_volume + b.shape_surface + b.color + b.diversity
                   + b.coherence + b.stability + b.count)
        return b

    def as_line(self) -> str:
        return (f"L={self.total:.4f}  sV={self.shape_volume:.3f}  "
                f"sS={self.shape_surface:.3f}  color={self.color:.3f}  "
                f"div={self.diversity:.3f}  coh={self.coherence:.3f}  "
                f"stab={self.stability:.3f}  cnt={self.count:.3f}")


# ---------------------------------------------------------------------------
# Term computations
# ---------------------------------------------------------------------------

def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return inter / union if union > 0 else 1.0


def _mean_lab_error(a_color: np.ndarray, target: Target) -> float:
    """L2 in Lab over cells where BOTH assembly and target are occupied.
    Only surface cells count (interior colors invisible)."""
    mask = target.surface
    if not mask.any():
        return 0.0
    pal = colors.palette()
    rgb_pal = np.array([c.rgb for c in pal], dtype=np.uint8)
    # LEGO cell color at each cell (0 for EMPTY).
    a_flat = np.where(a_color >= 0, a_color, 0)
    a_rgb = rgb_pal[a_flat]
    # Convert both to Lab.
    from .target import _lab_from_rgb
    a_lab = _lab_from_rgb(a_rgb.astype(np.float32))
    diff = a_lab - target.lab
    err = np.linalg.norm(diff, axis=-1) * mask
    n = int(mask.sum())
    # Normalize to [0,1]-ish (Lab space ~100 units total).
    return float(err.sum() / n / 40.0) if n else 0.0


def _type_entropy(cell_type: np.ndarray) -> float:
    """Diversity reward: fraction of visible types that have a *meaningful*
    presence in the assembly (≥ 4 cells or ≥ 0.05 % of total). Returns
    negative so lower loss ⇒ more distinct types in use.

    We use "count-distinct-with-threshold" rather than Shannon entropy so
    that adding a *new* type — for example a first door or a first slope —
    is a large, discrete reward step (~1/N_families) instead of an
    infinitesimal shift in a wide histogram."""
    occ = cell_type != EMPTY
    total = int(occ.sum())
    if total == 0:
        return 0.0
    threshold = max(4, int(total * 0.0005))
    n_significant = 0
    for t in VISIBLE_TYPES:
        if int((cell_type == t).sum()) >= threshold:
            n_significant += 1
    return -n_significant / max(1, len(VISIBLE_TYPES))


def _tv_adjacency(cell_type: np.ndarray, cell_color: np.ndarray) -> float:
    """Total variation on face-adjacent occupied pairs. Fraction of adjacent
    pairs where either type or color differs. Lower ⇒ more contiguous."""
    occ = cell_type != EMPTY
    pairs = 0
    diffs = 0
    # +Y
    a_occ, b_occ = occ[:-1], occ[1:]
    both = a_occ & b_occ
    pairs += int(both.sum())
    if pairs:
        t_diff = (cell_type[:-1] != cell_type[1:]) & both
        c_diff = (cell_color[:-1] != cell_color[1:]) & both
        diffs += int(t_diff.sum() + c_diff.sum())
    # +X
    a_occ, b_occ = occ[:, :-1], occ[:, 1:]
    both = a_occ & b_occ
    p = int(both.sum())
    pairs += p
    if p:
        t_diff = (cell_type[:, :-1] != cell_type[:, 1:]) & both
        c_diff = (cell_color[:, :-1] != cell_color[:, 1:]) & both
        diffs += int(t_diff.sum() + c_diff.sum())
    # +Z
    a_occ, b_occ = occ[:, :, :-1], occ[:, :, 1:]
    both = a_occ & b_occ
    p = int(both.sum())
    pairs += p
    if p:
        t_diff = (cell_type[:, :, :-1] != cell_type[:, :, 1:]) & both
        c_diff = (cell_color[:, :, :-1] != cell_color[:, :, 1:]) & both
        diffs += int(t_diff.sum() + c_diff.sum())
    # Divide by (pairs * 2) because each pair contributes to two indicator terms.
    return diffs / (pairs * 2) if pairs else 0.0


def _stability_components(cell_type: np.ndarray) -> float:
    """Number of connected components − 1, normalized. Cheap 3-D CC via BFS
    seeded on unvisited occupied cells."""
    occ = cell_type != EMPTY
    if not occ.any():
        return 0.0
    visited = np.zeros_like(occ)
    n_components = 0
    H, W, D = occ.shape
    # Simple flood fill using numpy stack.
    starts = np.argwhere(occ)
    for y, x, z in starts:
        if visited[y, x, z]:
            continue
        # BFS
        n_components += 1
        stack = [(int(y), int(x), int(z))]
        while stack:
            cy, cx, cz = stack.pop()
            if visited[cy, cx, cz] or not occ[cy, cx, cz]:
                continue
            visited[cy, cx, cz] = True
            for dy, dx, dz in ((1, 0, 0), (-1, 0, 0),
                                (0, 1, 0), (0, -1, 0),
                                (0, 0, 1), (0, 0, -1)):
                ny, nx_, nz = cy + dy, cx + dx, cz + dz
                if (0 <= ny < H and 0 <= nx_ < W and 0 <= nz < D
                    and not visited[ny, nx_, nz] and occ[ny, nx_, nz]):
                    stack.append((ny, nx_, nz))
        if n_components > 200:
            # Enough to know the assembly is fragmented; stop early.
            break
    # Normalize by "reasonable" scale (assume ≤ 50 components is worst-case).
    return min(1.0, (n_components - 1) / 50.0)


def _count_estimate(cell_type: np.ndarray, cell_color: np.ndarray) -> float:
    """Estimate placement count: number of face-adjacent cell RUNs of the same
    (type, color). Approximated via number of X-axis run-start transitions."""
    occ = cell_type != EMPTY
    if not occ.any():
        return 0.0
    # Along +X: a cell starts a run if the previous cell is empty or differs.
    prev_type = np.roll(cell_type, 1, axis=1)
    prev_type[:, 0, :] = EMPTY
    prev_color = np.roll(cell_color, 1, axis=1)
    prev_color[:, 0, :] = -1
    starts = occ & ((prev_type != cell_type) | (prev_color != cell_color))
    return float(starts.sum())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _surface_residual(a: Assembly, target: Target) -> float:
    """Composite surface-quality term. Two parts:

    1. **Position error** — mean mm distance from each surface cell centre
       to the nearest point on the original mesh. Fixed per-cell.
    2. **Type mismatch** — for each surface cell whose ideal type (derived
       from the mesh normal at that location) does *not* match the
       assembly's actual type, add a full-stud penalty.  This is what makes
       the loss reward slopes on tilted roof cells and tiles on flat top
       cells instead of plain plates everywhere.

    Both are normalised by stud size so a ~4 mm error reads as ~0.5.
    """
    from .. import bricks
    surf = _assembly_surface(a)
    if not surf.any():
        return 0.0
    residual = target.surface_residual_mm[surf]
    ideal = target.ideal_type[surf]
    actual = a.cell_type[surf]
    # Mismatch penalty: 1 full stud (8 mm) for choosing the wrong shape at a
    # surface cell that has a strong shape preference.  Cells whose ideal is
    # PLAIN (no strong preference) never get a mismatch penalty because they
    # match anything with `actual != EMPTY`.
    #
    # Cheese slopes are treated as valid *substitutes* for their 45° slope
    # counterparts: when ideal is SLOPE_XP the actual CHEESE_XP is also fine
    # — same orientation, smaller footprint. This lets the SA pick cheese
    # for curved regions without paying a shape penalty.
    cheese_match = (
        ((ideal == SLOPE_XP) & (actual == CHEESE_XP)) |
        ((ideal == SLOPE_XM) & (actual == CHEESE_XM)) |
        ((ideal == SLOPE_ZP) & (actual == CHEESE_ZP)) |
        ((ideal == SLOPE_ZM) & (actual == CHEESE_ZM))
    )
    def _same_orient(ideal_v, big_family):
        """XP/XM/ZP/ZM position in the family matches slope XZ orientation."""
        return (
            ((ideal == SLOPE_XP) & (actual == big_family[0])) |
            ((ideal == SLOPE_XM) & (actual == big_family[1])) |
            ((ideal == SLOPE_ZP) & (actual == big_family[2])) |
            ((ideal == SLOPE_ZM) & (actual == big_family[3]))
        )
    curved_match = _same_orient(ideal, CURVED_FAMILY)
    curved_2x4_match = _same_orient(ideal, CURVED_2X4_FAMILY)
    curved_4x1_match = _same_orient(ideal, CURVED_4X1_FAMILY)
    curved_6x1_match = _same_orient(ideal, CURVED_6X1_FAMILY)
    slope_4x2_match = _same_orient(ideal, SLOPE_4X2_FAMILY)
    mismatch = ((ideal != PLAIN) & (ideal != actual)
                & ~cheese_match & ~curved_match
                & ~curved_2x4_match & ~curved_4x1_match & ~curved_6x1_match
                & ~slope_4x2_match)
    penalty_mm = residual + mismatch.astype(np.float32) * bricks.MM_PER_STUD
    return float(penalty_mm.mean() / bricks.MM_PER_STUD)


def _assembly_surface(a: Assembly) -> np.ndarray:
    """Cells that are occupied AND have ≥1 empty face neighbour, OR touch the
    grid edge — the outside skin of the assembly."""
    occ = a.occupied()
    if not occ.any():
        return np.zeros_like(occ)
    surf = np.zeros_like(occ)
    surf[:-1, :, :] |= occ[:-1, :, :] & ~occ[1:, :, :]
    surf[1:, :, :]  |= occ[1:, :, :]  & ~occ[:-1, :, :]
    surf[:, :-1, :] |= occ[:, :-1, :] & ~occ[:, 1:, :]
    surf[:, 1:, :]  |= occ[:, 1:, :]  & ~occ[:, :-1, :]
    surf[:, :, :-1] |= occ[:, :, :-1] & ~occ[:, :, 1:]
    surf[:, :, 1:]  |= occ[:, :, 1:]  & ~occ[:, :, :-1]
    surf[0, :, :]  |= occ[0, :, :]
    surf[-1, :, :] |= occ[-1, :, :]
    surf[:, 0, :]  |= occ[:, 0, :]
    surf[:, -1, :] |= occ[:, -1, :]
    surf[:, :, 0]  |= occ[:, :, 0]
    surf[:, :, -1] |= occ[:, :, -1]
    return surf


def evaluate(a: Assembly, target: Target, w: Optional[Weights] = None
              ) -> tuple[LossBreakdown, LossBreakdown]:
    """Return (raw, weighted) breakdown. Raw components on ~[0,1]; weighted is
    what actually enters `total`."""
    if w is None:
        w = Weights()
    a_occ = a.occupied()
    iou = _iou(a_occ, target.occupied)
    shape_volume = 1.0 - iou
    shape_surface = _surface_residual(a, target)
    color = _mean_lab_error(a.cell_color, target)
    diversity = _type_entropy(a.cell_type)
    coherence = _tv_adjacency(a.cell_type, a.cell_color)
    stability = _stability_components(a.cell_type)
    n_cells = max(1, int(target.occupied.sum()))
    count = _count_estimate(a.cell_type, a.cell_color) / n_cells
    raw = LossBreakdown(shape_volume=shape_volume, shape_surface=shape_surface,
                         color=color, diversity=diversity,
                         coherence=coherence, stability=stability, count=count)
    weighted = raw.weighted(w)
    return raw, weighted


# ---------------------------------------------------------------------------
# Cheap per-move delta
# ---------------------------------------------------------------------------

def _neighbors6(y: int, x: int, z: int, shape: tuple[int, int, int]):
    H, W, D = shape
    for dy, dx, dz in ((1, 0, 0), (-1, 0, 0),
                        (0, 1, 0), (0, -1, 0),
                        (0, 0, 1), (0, 0, -1)):
        ny, nx_, nz = y + dy, x + dx, z + dz
        if 0 <= ny < H and 0 <= nx_ < W and 0 <= nz < D:
            yield ny, nx_, nz


def delta_swap_cell(a: Assembly, target: Target, w: Weights,
                    y: int, x: int, z: int,
                    new_type: int, new_color: int) -> float:
    """Return dL from swapping a single cell to (new_type, new_color).
    Only shape/color/coherence terms have local support; the others
    (diversity, stability, count) are recomputed on demand — cheap enough
    at grid scale ~50k cells and skipped here.

    Diversity, stability, count contributions from ONE cell are negligible
    per-move, so we ignore them for delta and let the periodic full-eval
    catch drift.
    """
    old_type = int(a.cell_type[y, x, z])
    old_color = int(a.cell_color[y, x, z])
    if old_type == new_type and old_color == new_color:
        return 0.0

    old_occ = old_type != EMPTY
    new_occ = new_type != EMPTY

    dL = 0.0

    # --- shape: IoU-based; +1 to union or intersection at this cell only
    if bool(target.occupied[y, x, z]) != new_occ:
        # Now diverges from target ⇒ worse
        pass
    # Approximation: shape term change per cell ≈ ±1/N (union grows/shrinks
    # by 1). This is cheap and unbiased on average.
    N = max(1, int(target.occupied.sum()))
    if bool(target.occupied[y, x, z]):
        if old_occ and not new_occ:
            dL += w.shape * (1.0 / N)   # lost a matching cell
        elif not old_occ and new_occ:
            dL -= w.shape * (1.0 / N)   # gained a matching cell
    else:
        if not old_occ and new_occ:
            dL += w.shape * (1.0 / N)   # added a non-target cell
        elif old_occ and not new_occ:
            dL -= w.shape * (1.0 / N)   # removed a non-target cell

    # --- color: only if this cell is a target-surface cell
    if target.surface[y, x, z]:
        pal = colors.palette()
        rgb_pal = np.array([c.rgb for c in pal], dtype=np.uint8)
        from .target import _lab_from_rgb
        tgt_lab = target.lab[y, x, z]
        old_rgb = rgb_pal[old_color] if old_color >= 0 else np.array([0, 0, 0], dtype=np.uint8)
        new_rgb = rgb_pal[new_color] if new_color >= 0 else np.array([0, 0, 0], dtype=np.uint8)
        old_lab = _lab_from_rgb(old_rgb.astype(np.float32)[None])[0]
        new_lab = _lab_from_rgb(new_rgb.astype(np.float32)[None])[0]
        Nm = max(1, int(target.surface.sum()))
        old_err = float(np.linalg.norm(old_lab - tgt_lab)) / 40.0
        new_err = float(np.linalg.norm(new_lab - tgt_lab)) / 40.0
        dL += w.color * (new_err - old_err) / Nm

    # --- coherence: face-adjacent (type, color) discontinuity change
    # For each neighbor, count old-vs-new indicator flips.
    pairs = 0
    old_diffs = 0
    new_diffs = 0
    for ny, nx_, nz in _neighbors6(y, x, z, a.cell_type.shape):
        nt = int(a.cell_type[ny, nx_, nz])
        nc = int(a.cell_color[ny, nx_, nz])
        if nt == EMPTY or (old_type == EMPTY and new_type == EMPTY):
            continue
        pairs += 1
        if old_type != EMPTY and nt != EMPTY:
            if old_type != nt:
                old_diffs += 1
            if old_color != nc:
                old_diffs += 1
        if new_type != EMPTY and nt != EMPTY:
            if new_type != nt:
                new_diffs += 1
            if new_color != nc:
                new_diffs += 1
    if pairs > 0:
        # Global TV was diffs / (total_pairs * 2). This cell's contribution
        # scales as (new_diffs - old_diffs) / (total_pairs * 2). We don't
        # track total_pairs in delta — approximate as N_cells * 3 (avg 3 axes).
        Ntot = max(1, int(a.occupied().sum())) * 6
        dL += w.coherence * (new_diffs - old_diffs) / Ntot

    return dL
