"""SoftTrainer — Adam-driven gradient descent with temperature annealing.

Iterates:
    1. Compute SoftLoss(SoftAssembly, tau).
    2. loss.backward() — gradients flow into type_logits + color_logits.
    3. Adam step.
    4. Anneal tau (τ_start → τ_end geometrically).

At the end, discretise the soft assembly by argmax and emit Placements.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

from ..bricks import MM_PER_STUD, MM_PER_PLATE
from ..colors import palette
from ..pack import Placement
from ..templates.base import part_by_ldraw
from .soft_assembly import (SoftAssembly, N_TYPES, TYPE_NAMES,
                                EMPTY, PLATE, TILE,
                                SLOPE_XP, SLOPE_XM, SLOPE_ZP, SLOPE_ZM,
                                CHEESE_XP, CHEESE_XM, CHEESE_ZP, CHEESE_ZM)
from .soft_loss import SoftLoss, SoftLossBreakdown


def _majority_smooth_colors(hard_types: np.ndarray,
                              hard_colors: np.ndarray,
                              *, passes: int = 1) -> np.ndarray:
    """Majority-vote colour smoothing over 6-neighbour windows.

    Per pass: for each occupied cell tally the colour of each occupied
    face neighbour, then if any single colour is picked by ≥ 4 of the
    (up-to-6) neighbours and differs from the cell's own, adopt it.
    Multiple passes propagate the smoothing further.
    """
    from .soft_assembly import EMPTY
    occ = hard_types != EMPTY
    Y, X, Z = hard_colors.shape
    n_col = int(hard_colors.max()) + 2
    colors = hard_colors.copy()

    def _shift(arr: np.ndarray, axis: int, direction: int) -> np.ndarray:
        """Shift arr by `direction` (+1 or -1) along `axis`, filling the
        vacated slice with a value that acts as "no-neighbour" (colours
        get 0, occupancy gets False)."""
        rolled = np.roll(arr, shift=direction, axis=axis)
        clear = [slice(None)] * 3
        idx = 0 if direction > 0 else -1
        clear[axis] = idx
        rolled[tuple(clear)] = 0
        return rolled

    def _shift_occ(arr: np.ndarray, axis: int, direction: int) -> np.ndarray:
        rolled = np.roll(arr, shift=direction, axis=axis)
        clear = [slice(None)] * 3
        idx = 0 if direction > 0 else -1
        clear[axis] = idx
        rolled[tuple(clear)] = False
        return rolled

    # (Y, X, Z, N_COL) neighbour-colour histogram.  Each cell's histogram
    # sums 1 per occupied face-neighbour with that colour.
    for _ in range(max(1, passes)):
        hist = np.zeros((Y, X, Z, n_col), dtype=np.int16)
        for axis in range(3):
            for direction in (-1, +1):
                nb_c = _shift(colors, axis, direction)
                nb_o = _shift_occ(occ, axis, direction)
                # Where nb_o is True, add 1 at hist[..., nb_c].
                if not nb_o.any():
                    continue
                idx_flat = np.flatnonzero(nb_o.reshape(-1))
                col_at = nb_c.reshape(-1)[idx_flat]
                np.add.at(hist.reshape(-1, n_col), (idx_flat, col_at), 1)
        maj_col = hist.argmax(axis=-1)
        maj_cnt = hist.max(axis=-1)
        replace = occ & (maj_cnt >= 4) & (maj_col != colors)
        colors = np.where(replace, maj_col, colors)
    return colors


def _column_majority_colors(hard_types: np.ndarray,
                              hard_colors: np.ndarray) -> np.ndarray:
    """For each (x, z) column, snap each PLATE cell's colour to the
    dominant colour among its 3-plate vertical window (y-1, y, y+1).

    This trades a small amount of per-cell colour fidelity for massive
    part-count reduction: brick_tall + brick pre-passes can now fire
    on textured meshes where 3-plate columns had a mix of similar
    LDraw greys/browns.
    """
    from .soft_assembly import PLATE
    Y, X, Z = hard_colors.shape
    plate_mask = (hard_types == PLATE)
    n_col = int(hard_colors.max()) + 2
    # Per-y, per-colour count within a 3-plate window, using cumulative
    # sums over the y-axis for O(Y*X*Z*C) complexity but vectorized.
    # For each colour c, build an occupancy array (Y,X,Z) that is 1
    # where cell is PLATE and colour==c.  Sum over 3-plate window via
    # convolution (numpy roll).
    result = hard_colors.copy()
    # Tally colour counts per cell for its 3-plate window.
    hist = np.zeros((Y, X, Z, n_col), dtype=np.int16)
    for dy in (-1, 0, 1):
        if dy == 0:
            src_c = hard_colors
            src_v = plate_mask
        else:
            src_c = np.roll(hard_colors, -dy, axis=0)
            src_v = np.roll(plate_mask, -dy, axis=0)
            if dy < 0:  # shifted "up" by |dy|; bottom rows become invalid
                src_v[Y + dy:, :, :] = False
            else:       # shifted "down"; top rows invalid
                src_v[:dy, :, :] = False
        # Increment hist at src_c wherever src_v is True.
        idx = np.flatnonzero(src_v.reshape(-1))
        if idx.size:
            col_flat = src_c.reshape(-1)[idx]
            np.add.at(hist.reshape(-1, n_col), (idx, col_flat), 1)
    maj_col = hist.argmax(axis=-1)
    # Only rewrite PLATE cells whose window had >= 2 valid neighbours.
    total = hist.sum(axis=-1)
    replace = plate_mask & (total >= 2)
    result = np.where(replace, maj_col, result)
    return result


def _plane_majority_colors(hard_types: np.ndarray,
                              hard_colors: np.ndarray,
                              *, passes: int = 1) -> np.ndarray:
    """Horizontal (in-y-plane) 3×3 majority smoothing of PLATE cells.

    Analogous to `_column_majority_colors` but along (x, z) at each y.
    Multiple passes propagate colour blocks outward: pass 1 grows same-
    colour blocks by 1 cell in each direction, pass 2 by 2.

    Trades sub-cell colour fidelity for MILP-consolidation freedom —
    the biggest wins are on textured meshes with tight colour dithering.
    """
    from .soft_assembly import PLATE
    Y, X, Z = hard_colors.shape
    plate_mask = (hard_types == PLATE)
    n_col = int(hard_colors.max()) + 2
    colors = hard_colors.copy()
    for _ in range(max(1, passes)):
        hist = np.zeros((Y, X, Z, n_col), dtype=np.int16)
        # 8 in-plane neighbours + self.
        offsets = [(dx, dz) for dx in (-1, 0, 1) for dz in (-1, 0, 1)]
        for dx, dz in offsets:
            src_c = np.roll(np.roll(colors, -dx, axis=1), -dz, axis=2)
            src_v = np.roll(np.roll(plate_mask, -dx, axis=1), -dz, axis=2)
            if dx < 0:
                src_v[:, X + dx:, :] = False
            elif dx > 0:
                src_v[:, :dx, :] = False
            if dz < 0:
                src_v[:, :, Z + dz:] = False
            elif dz > 0:
                src_v[:, :, :dz] = False
            idx = np.flatnonzero(src_v.reshape(-1))
            if idx.size:
                col_flat = src_c.reshape(-1)[idx]
                np.add.at(hist.reshape(-1, n_col), (idx, col_flat), 1)
        maj_col = hist.argmax(axis=-1)
        maj_cnt = hist.max(axis=-1)
        # Adopt majority when it dominates ≥ 5 of the 9-cell window.
        replace = plate_mask & (maj_cnt >= 5)
        colors = np.where(replace, maj_col, colors)
    return colors


@dataclass
class SoftTrainerResult:
    placements: list[Placement]
    initial_loss: float = 0.0
    best_loss: float = 0.0
    n_steps: int = 0
    elapsed_s: float = 0.0
    history: list[float] = field(default_factory=list)


class SoftTrainer:
    def __init__(self, *,
                  soft_loss: SoftLoss,
                  lr: float = 0.5,
                  iters: int = 500,
                  tau_start: float = 1.0,
                  tau_end: float = 0.05,
                  on_progress: Optional[Callable[[int, SoftLossBreakdown], None]] = None,
                  progress_every: int = 50) -> None:
        assert _HAS_TORCH
        self.soft_loss = soft_loss
        self.lr = lr
        self.iters = iters
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.on_progress = on_progress
        self.progress_every = progress_every

    def run(self, a: SoftAssembly) -> SoftTrainerResult:
        opt = torch.optim.Adam(a.parameters(), lr=self.lr)
        tau = self.tau_start
        alpha = (self.tau_end / self.tau_start) ** (1.0 / max(1, self.iters - 1))

        _, init_L = self.soft_loss.evaluate(a, tau=tau)
        best_L = float(init_L.detach())
        history = [best_L]
        t0 = time.time()

        for step in range(self.iters):
            opt.zero_grad()
            brk, L = self.soft_loss.evaluate(a, tau=tau)
            L.backward()
            opt.step()
            if brk.total < best_L:
                best_L = brk.total
            history.append(brk.total)
            if (self.on_progress is not None
                    and step % max(1, self.progress_every) == 0):
                self.on_progress(step, brk)
            tau *= alpha

        placements = self._discretise(a)
        return SoftTrainerResult(
            placements=placements,
            initial_loss=history[0],
            best_loss=best_L,
            n_steps=self.iters,
            elapsed_s=time.time() - t0,
            history=history,
        )

    # ---- discretisation ------------------------------------------------

    def _discretise(self, a: SoftAssembly, *,
                       consolidate: bool = True) -> list[Placement]:
        """Turn the argmax of the soft assembly into real Placements.

        When `consolidate=True`, runs MILPPacker per (y-level, type, color)
        group so contiguous same-attribute cells become the biggest LEGO
        part that fits — same principle as the v1 export packer, applied
        here to the argmax grid.  When False, emits one Placement per
        non-empty cell (fastest, useful for debugging the raw soft state)."""
        hard_types = a.hard_types()
        hard_colors = a.hard_colors()
        # Morphological cleanup: replace isolated-colour outliers by the
        # majority colour among their 6-face neighbours.  Removes the
        # single-cell "sparkle" that appears on roof surfaces where the
        # mesh's projected colour was slightly off.  Only run on cells
        # whose original argmax is non-EMPTY.
        hard_colors = _majority_smooth_colors(hard_types, hard_colors,
                                                    passes=2)
        # Vertical column-majority smoothing.  Along each (x, z) column,
        # snap every non-empty cell's colour to the DOMINANT colour
        # inside its 3-plate window (y-1, y, y+1).  Enables the brick
        # pre-pass to actually fire on textured meshes — otherwise
        # each 3-plate column has a mix of colours from voxel dithering
        # and only 1×1 bricks can be used.
        hard_colors = _column_majority_colors(hard_types, hard_colors)
        # Horizontal in-plane majority smoothing.  Same idea but along
        # (x, z) at each y-layer, using a 3x3 window.  Enables MILP to
        # pack big 2×N and NxM footprints on textured surfaces where
        # adjacent voxels were dithering between two close LDraw
        # entries.  Two passes so a colour block can grow by 2 cells.
        hard_colors = _plane_majority_colors(hard_types, hard_colors,
                                                    passes=2)
        Y, X, Z = hard_types.shape
        plate_1x1 = part_by_ldraw("3024.dat")
        tile_1x1 = part_by_ldraw("3070b.dat")
        cheese = part_by_ldraw("54200.dat")
        slope_2x1 = part_by_ldraw("3040.dat")
        if plate_1x1 is None:
            return []
        if consolidate:
            return self._discretise_consolidated(a, hard_types, hard_colors)

        cheese_rot = {CHEESE_XP: 0, CHEESE_XM: 180,
                        CHEESE_ZP: 270, CHEESE_ZM: 90}
        slope_rot = {SLOPE_XP: 0, SLOPE_XM: 180,
                       SLOPE_ZP: 270, SLOPE_ZM: 90}
        out: list[Placement] = []
        for y in range(Y):
            for x in range(X):
                for z in range(Z):
                    t = int(hard_types[y, x, z])
                    if t == EMPTY:
                        continue
                    c = int(hard_colors[y, x, z])
                    if t == PLATE:
                        part = plate_1x1
                        rot = 0
                        w, d = 1, 1
                    elif t == TILE:
                        part = tile_1x1 or plate_1x1
                        rot = 0
                        w, d = 1, 1
                    elif t in cheese_rot and cheese is not None:
                        part = cheese
                        rot = cheese_rot[t]
                        w, d = 1, 1
                    elif t in slope_rot and slope_2x1 is not None:
                        # Slope is 2 studs wide — only emit when the +axis
                        # neighbour is the same type (block claim).
                        part = slope_2x1
                        rot = slope_rot[t]
                        # Use as 1x1 for the discretiser first-pass (the
                        # consolidator can promote to 2x1 in a follow-up).
                        part = plate_1x1
                        rot = 0
                        w, d = 1, 1
                    else:
                        part = plate_1x1
                        rot = 0
                        w, d = 1, 1
                    out.append(Placement(
                        part=part, color_idx=c,
                        x_stud=x, y_plate=y, z_stud=z,
                        w_world=w, d_world=d, rotation_deg=rot,
                    ))
        return out

    def _discretise_consolidated(self, a: SoftAssembly,
                                    hard_types: np.ndarray,
                                    hard_colors: np.ndarray
                                    ) -> list[Placement]:
        """Consolidated discretisation — runs MILPPacker per (y-level,
        type, colour) group so same-attribute contiguous cells become
        the biggest LEGO plate that fits.

        Brick pre-pass: before per-layer plate packing, look for
        vertically-contiguous PLATE cells whose columns match a 3-plate
        (BRICK) or 5-plate (BRICK_TALL) height.  Those columns are
        consolidated into single brick placements — 3× fewer parts,
        3× fewer horizontal seams on walls.
        """
        from ..bricks import catalog
        from .milp import MILPPacker
        Y, X, Z = hard_types.shape
        plate_footprints = sorted(
            {fp for p in catalog() if p.shape == "plate"
              for fp in p.orientations()},
            key=lambda wd: (-wd[0] * wd[1], -max(wd)))
        tile_footprints = sorted(
            {fp for p in catalog() if p.shape == "tile"
              for fp in p.orientations()},
            key=lambda wd: (-wd[0] * wd[1], -max(wd)))
        brick_footprints = sorted(
            {fp for p in catalog() if p.shape == "brick"
              for fp in p.orientations()},
            key=lambda wd: (-wd[0] * wd[1], -max(wd)))
        plate_lookup: dict[tuple[int, int], "object"] = {}
        for p in catalog():
            if p.shape == "plate":
                for w, d in p.orientations():
                    plate_lookup.setdefault((w, d), p)
        tile_lookup: dict[tuple[int, int], "object"] = {}
        for p in catalog():
            if p.shape == "tile":
                for w, d in p.orientations():
                    tile_lookup.setdefault((w, d), p)
        brick_lookup: dict[tuple[int, int], "object"] = {}
        for p in catalog():
            if p.shape == "brick":
                for w, d in p.orientations():
                    brick_lookup.setdefault((w, d), p)

        # BRICK PRE-PASSES.  Two rounds:
        #   Round A: brick_tall (15 plates = 5 bricks)  — 30360, 2454
        #   Round B: brick (3 plates)  — the standard 3-plate consolidation
        # A cell qualifies for consolidation when all N stacked layers
        # have the same PLATE type + same colour.  Greedy biggest-first
        # so tall columns collapse before the 3-plate pass sees them.
        consumed = np.zeros_like(hard_types, dtype=bool)
        out: list[Placement] = []

        brick_tall_2x2 = part_by_ldraw("30360.dat")   # 2x2x5
        brick_tall_1x2 = part_by_ldraw("2454.dat")    # 1x2x5

        def _try_tall(part, fw, fd, height):
            if part is None:
                return
            for rot, pw, pd in ((0, fw, fd), (90, fd, fw)) \
                    if fw != fd else ((0, fw, fd),):
                for y in range(Y - height + 1):
                    for x in range(X - pw + 1):
                        for z in range(Z - pd + 1):
                            sl = np.s_[y:y + height, x:x + pw, z:z + pd]
                            if consumed[sl].any():
                                continue
                            if not (hard_types[sl] == PLATE).all():
                                continue
                            c0 = int(hard_colors[y, x, z])
                            if not (hard_colors[sl] == c0).all():
                                continue
                            out.append(Placement(
                                part=part, color_idx=c0,
                                x_stud=x, y_plate=y, z_stud=z,
                                w_world=pw, d_world=pd,
                                rotation_deg=rot,
                            ))
                            consumed[sl] = True

        _try_tall(brick_tall_2x2, 2, 2, 15)
        _try_tall(brick_tall_1x2, 2, 1, 15)
        packer_brick = MILPPacker(parts=brick_footprints,
                                     time_limit_s=1.0, max_mask_area=200) \
                             if brick_footprints else None
        if packer_brick is not None and brick_lookup:
            for y in range(0, Y - 2, 3):
                cells_ok = ((hard_types[y] == PLATE)
                              & (hard_types[y + 1] == PLATE)
                              & (hard_types[y + 2] == PLATE)
                              & (hard_colors[y] == hard_colors[y + 1])
                              & (hard_colors[y + 1] == hard_colors[y + 2]))
                if not cells_ok.any():
                    continue
                for color_idx in np.unique(hard_colors[y][cells_ok]):
                    mask = cells_ok & (hard_colors[y] == color_idx)
                    if not mask.any():
                        continue
                    res = packer_brick.pack(mask)
                    for pl in res.placements:
                        fw, fd = brick_footprints[pl.part_index]
                        part = brick_lookup.get((fw, fd)) \
                                or brick_lookup.get((fd, fw))
                        if part is None:
                            continue
                        out.append(Placement(
                            part=part, color_idx=int(color_idx),
                            x_stud=pl.x, y_plate=y, z_stud=pl.z,
                            w_world=pl.w, d_world=pl.d,
                            rotation_deg=pl.rotation_deg,
                        ))
                        consumed[y:y+3, pl.x:pl.x+pl.w,
                                    pl.z:pl.z+pl.d] = True

        # Prepare cheese + slope 1x1 emission (no consolidation yet).
        cheese_p = part_by_ldraw("54200.dat")
        plate_1x1 = part_by_ldraw("3024.dat")
        slope_2x1 = part_by_ldraw("3040.dat") or part_by_ldraw("3040b.dat")
        slope_2x2 = part_by_ldraw("3039.dat")
        slope_4x1 = part_by_ldraw("61678.dat")   # curved slope, 4x1x3
        brick_1x1 = part_by_ldraw("3005.dat")    # same height as 2x1 slope
        cheese_rot = {CHEESE_XP: 0, CHEESE_XM: 180,
                        CHEESE_ZP: 270, CHEESE_ZM: 90}
        # 2x1 slope (part 3040) low-toward-<axis>.  LDraw rot convention here
        # matches cheese_rot above.
        slope_rot = {SLOPE_XP: 0, SLOPE_XM: 180,
                       SLOPE_ZP: 270, SLOPE_ZM: 90}
        packer_plate = MILPPacker(parts=plate_footprints,
                                     time_limit_s=1.0, max_mask_area=200)
        packer_tile = MILPPacker(parts=tile_footprints,
                                    time_limit_s=1.0, max_mask_area=200)
        for y in range(Y):
            plate_cells = (hard_types[y] == PLATE) & (~consumed[y])
            tile_cells = (hard_types[y] == TILE) & (~consumed[y])
            for color_idx in np.unique(hard_colors[y]):
                pmask = plate_cells & (hard_colors[y] == color_idx)
                if pmask.any() and plate_1x1 is not None:
                    result = packer_plate.pack(pmask)
                    for pl in result.placements:
                        fw, fd = plate_footprints[pl.part_index]
                        part = plate_lookup.get((fw, fd)) \
                                or plate_lookup.get((fd, fw), plate_1x1)
                        out.append(Placement(
                            part=part, color_idx=int(color_idx),
                            x_stud=pl.x, y_plate=y, z_stud=pl.z,
                            w_world=pl.w, d_world=pl.d,
                            rotation_deg=pl.rotation_deg,
                        ))
                tmask = tile_cells & (hard_colors[y] == color_idx)
                if tmask.any() and plate_1x1 is not None:
                    result = packer_tile.pack(tmask)
                    for pl in result.placements:
                        fw, fd = tile_footprints[pl.part_index]
                        part = tile_lookup.get((fw, fd)) \
                                or tile_lookup.get((fd, fw), plate_1x1)
                        out.append(Placement(
                            part=part, color_idx=int(color_idx),
                            x_stud=pl.x, y_plate=y, z_stud=pl.z,
                            w_world=pl.w, d_world=pl.d,
                            rotation_deg=pl.rotation_deg,
                        ))
        # Cheese cells (54200) emit 1x1 directly.
        # SLOPE_* cells consolidate by scanning MAXIMAL RUNS of the same
        # (type, color) along the perpendicular-to-pitch axis:
        #   SLOPE_XP / SLOPE_XM pitch along ±x, run along z
        #   SLOPE_ZP / SLOPE_ZM pitch along ±z, run along x
        # For a run of length N: emit ⌊N/2⌋ × 2x1 slopes (3040) and, when
        # N is odd, a single 1x1 BRICK (3005) at the FIRST cell — same
        # height as the 2x1 slope, so no sawtooth pit at the run's edge,
        # and placed toward the interior (hidden against the gable end).
        # Cheese fallback is used only when neither the 2x1 slope nor the
        # 1x1 brick is available in the catalog.
        slope_pair_axis = {SLOPE_XP: "z", SLOPE_XM: "z",
                             SLOPE_ZP: "x", SLOPE_ZM: "x"}
        # Merge the brick-pre-pass' `consumed` mask so slopes don't
        # re-emit into cells that were already claimed by bricks.
        claimed = consumed.copy()
        # First pass: emit cheese cells (unambiguous, no consolidation).
        for y in range(Y):
            for x in range(X):
                for z in range(Z):
                    if claimed[y, x, z]:
                        continue
                    t = int(hard_types[y, x, z])
                    if t in cheese_rot and cheese_p is not None:
                        c = int(hard_colors[y, x, z])
                        out.append(Placement(
                            part=cheese_p, color_idx=c,
                            x_stud=x, y_plate=y, z_stud=z,
                            w_world=1, d_world=1,
                            rotation_deg=cheese_rot[t],
                        ))
                        claimed[y, x, z] = True

        def _emit_slope_run(y: int, x0: int, z0: int, t: int, c: int,
                              run_axis: str, length: int) -> None:
            """Emit a run of `length` same-(t,c) SLOPE_* cells starting at
            (y, x0, z0) along `run_axis`.  Uses 2x1 slopes + 1x1 brick
            for the odd cell (placed at the start of the run)."""
            rot = slope_rot[t]
            i = 0
            if length % 2 == 1:
                # Odd cell → 1x1 brick at the start.
                if brick_1x1 is not None:
                    out.append(Placement(
                        part=brick_1x1, color_idx=c,
                        x_stud=x0, y_plate=y, z_stud=z0,
                        w_world=1, d_world=1, rotation_deg=0,
                    ))
                elif cheese_p is not None:
                    out.append(Placement(
                        part=cheese_p, color_idx=c,
                        x_stud=x0, y_plate=y, z_stud=z0,
                        w_world=1, d_world=1, rotation_deg=rot,
                    ))
                if run_axis == "z":
                    claimed[y, x0, z0] = True
                else:
                    claimed[y, x0, z0] = True
                i = 1
            # Remaining cells are even — emit 2x1 slopes.
            while i < length:
                if run_axis == "z":
                    z_here = z0 + i
                    if slope_2x1 is not None:
                        out.append(Placement(
                            part=slope_2x1, color_idx=c,
                            x_stud=x0, y_plate=y, z_stud=z_here,
                            w_world=1, d_world=2, rotation_deg=rot,
                        ))
                    else:
                        # Fallback: two 1x1 bricks.
                        for dz in (0, 1):
                            if brick_1x1 is not None:
                                out.append(Placement(
                                    part=brick_1x1, color_idx=c,
                                    x_stud=x0, y_plate=y, z_stud=z_here + dz,
                                    w_world=1, d_world=1, rotation_deg=0,
                                ))
                    claimed[y, x0, z_here] = True
                    claimed[y, x0, z_here + 1] = True
                else:  # run_axis == "x"
                    x_here = x0 + i
                    if slope_2x1 is not None:
                        out.append(Placement(
                            part=slope_2x1, color_idx=c,
                            x_stud=x_here, y_plate=y, z_stud=z0,
                            w_world=2, d_world=1, rotation_deg=rot,
                        ))
                    else:
                        for dx in (0, 1):
                            if brick_1x1 is not None:
                                out.append(Placement(
                                    part=brick_1x1, color_idx=c,
                                    x_stud=x_here + dx, y_plate=y, z_stud=z0,
                                    w_world=1, d_world=1, rotation_deg=0,
                                ))
                    claimed[y, x_here, z0] = True
                    claimed[y, x_here + 1, z0] = True
                i += 2

        # Second pass: SLOPE_* runs.
        for y in range(Y):
            for x in range(X):
                for z in range(Z):
                    if claimed[y, x, z]:
                        continue
                    t = int(hard_types[y, x, z])
                    if t not in slope_rot:
                        continue
                    c = int(hard_colors[y, x, z])
                    axis = slope_pair_axis[t]
                    # Scan max run along axis.
                    run = 1
                    if axis == "z":
                        z_end = z + 1
                        while (z_end < Z and not claimed[y, x, z_end]
                                and int(hard_types[y, x, z_end]) == t
                                and int(hard_colors[y, x, z_end]) == c):
                            run += 1
                            z_end += 1
                    else:  # axis == "x"
                        x_end = x + 1
                        while (x_end < X and not claimed[y, x_end, z]
                                and int(hard_types[y, x_end, z]) == t
                                and int(hard_colors[y, x_end, z]) == c):
                            run += 1
                            x_end += 1
                    _emit_slope_run(y, x, z, t, c, axis, run)
        return out
