"""Simulated annealing driver.

Given a Target and an initial Assembly, iterate:
    1. sample a proposal move
    2. apply it
    3. evaluate full loss (or fast approximate delta)
    4. accept if dL < 0 or with prob exp(-dL/T)
    5. otherwise revert
    6. cool T on a geometric schedule
    7. periodically re-evaluate full loss (delta drifts)
    8. track best assembly by full-eval loss

Exports:
    OptResult      — bookkeeping: best assembly, loss history, stats
    optimize(...)  — the main driver

Convert `result.best` to actual LDraw placements with
`assembly_to_placements(result.best, target)` from `assembly_export`.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .loss import Assembly, LossBreakdown, Weights, evaluate
from .moves import Move, apply, revert, sample_move
from .target import Target


@dataclass
class OptResult:
    best: Assembly
    best_loss: LossBreakdown
    best_raw: LossBreakdown
    history: list[float] = field(default_factory=list)
    n_accepts: int = 0
    n_proposals: int = 0
    elapsed_s: float = 0.0
    initial_loss: Optional[LossBreakdown] = None


def optimize(target: Target,
              *,
              weights: Optional[Weights] = None,
              iters: int = 20_000,
              T_start: float = 0.05,
              T_end: float = 0.0005,
              full_eval_every: int = 500,
              seed: int = 0,
              on_progress: Optional[Callable[[int, LossBreakdown], None]] = None,
              initial: Optional[Assembly] = None,
              ) -> OptResult:
    """Run SA. Returns the best assembly found + stats.

    Parameters
    ----------
    target : Target
        Precomputed from `build_target`.
    weights : Weights
        Loss weights (defaults to the tunable set in loss.py).
    iters : int
        Number of proposal iterations.
    T_start, T_end : float
        Geometric cooling from T_start to T_end.
    full_eval_every : int
        Recompute the full loss every N iters (delta drift correction).
    initial : Assembly | None
        Starting assembly. Defaults to `Assembly.from_target(target)`.
    """
    if weights is None:
        weights = Weights()
    rng = random.Random(seed)
    np.random.seed(seed)

    a = initial if initial is not None else Assembly.from_target(target)
    raw, weighted = evaluate(a, target, weights)
    result = OptResult(
        best=Assembly(cell_type=a.cell_type.copy(), cell_color=a.cell_color.copy()),
        best_loss=weighted,
        best_raw=raw,
        initial_loss=weighted,
    )
    result.history.append(weighted.total)

    # Cooling schedule (geometric).
    if iters > 1 and T_start > 0 and T_end > 0:
        alpha = (T_end / T_start) ** (1.0 / max(1, iters - 1))
    else:
        alpha = 1.0
    T = float(T_start)

    current_loss = weighted.total
    t_start = time.time()

    for step in range(iters):
        move = sample_move(a, target, rng)
        if move is None:
            T *= alpha
            continue

        apply(a, move)
        result.n_proposals += 1

        # Full-eval every N steps; between eval-points we use full re-eval too
        # (this optimizer is small enough that a full IoU/coherence pass on a
        # 40k-cell grid is only a few ms with numpy).
        raw, weighted = evaluate(a, target, weights)
        dL = weighted.total - current_loss

        accept = dL < 0 or rng.random() < math.exp(-dL / max(T, 1e-9))
        if accept:
            current_loss = weighted.total
            result.n_accepts += 1
            if weighted.total < result.best_loss.total:
                result.best.cell_type[:] = a.cell_type
                result.best.cell_color[:] = a.cell_color
                result.best_loss = weighted
                result.best_raw = raw
        else:
            revert(a, move)

        result.history.append(current_loss)
        T *= alpha

        if on_progress is not None and step % max(1, iters // 40) == 0:
            on_progress(step, weighted)

    result.elapsed_s = time.time() - t_start
    return result
