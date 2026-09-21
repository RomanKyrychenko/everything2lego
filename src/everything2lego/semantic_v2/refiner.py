"""Refiner — simulated-annealing loss-guided refinement of an Assembly.

Given an initial Assembly (typically from SemanticLegoPipeline), a
LossFunction, and a move pool, iteratively propose moves and accept those
that decrease the loss (or, with an SA temperature, occasionally accept
losing moves to escape local optima).

Keeps the best-observed assembly across the whole run.

Usage:
    refiner = Refiner(loss=loss, moves=build_move_pool(...), iters=5000)
    result = refiner.run(initial_placements)
    result.best   # list[Placement] with lowest total loss
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from ..pack import Placement
from .loss import LossFunction, LossBreakdown
from .moves import Move, sample_move


@dataclass
class RefinerResult:
    best: list[Placement]
    best_raw: LossBreakdown
    best_weighted: LossBreakdown
    initial_raw: LossBreakdown
    initial_weighted: LossBreakdown
    n_accepted: int = 0
    n_proposed: int = 0
    elapsed_s: float = 0.0
    history: list[float] = field(default_factory=list)


class Refiner:
    """SA driver — proposes moves from the pool, accepts by loss delta."""

    def __init__(self, *,
                  loss: LossFunction,
                  moves: list[tuple[Move, int]],
                  iters: int = 5000,
                  t_start: float = 0.02,
                  t_end: float = 0.0005,
                  seed: int = 0,
                  on_progress: Optional[Callable[[int, LossBreakdown], None]] = None,
                  progress_every: int = 200) -> None:
        self.loss = loss
        self.moves = moves
        self.iters = iters
        self.t_start = t_start
        self.t_end = t_end
        self.seed = seed
        self.on_progress = on_progress
        self.progress_every = progress_every

    def run(self, initial: list[Placement]) -> RefinerResult:
        rng = random.Random(self.seed)
        current = list(initial)
        init_raw, init_weighted = self.loss.evaluate(current)

        best = list(current)
        best_raw = init_raw
        best_weighted = init_weighted
        current_loss = init_weighted.total

        if self.iters > 1 and self.t_start > 0 and self.t_end > 0:
            alpha = (self.t_end / self.t_start) ** (1.0 / (self.iters - 1))
        else:
            alpha = 1.0
        T = float(self.t_start)

        history = [current_loss]
        t0 = time.time()
        n_accepted = 0
        n_proposed = 0

        for step in range(self.iters):
            move = sample_move(self.moves, rng)
            result = move.apply(current, rng)
            n_proposed += 1
            if result is None:
                T *= alpha
                continue

            candidate = result.new_placements
            _, weighted = self.loss.evaluate(candidate)
            dL = weighted.total - current_loss
            accept = dL < 0 or rng.random() < math.exp(-dL / max(T, 1e-9))
            if accept:
                current = candidate
                current_loss = weighted.total
                n_accepted += 1
                if weighted.total < best_weighted.total:
                    best = list(current)
                    best_weighted = weighted
                    best_raw, _ = self.loss.evaluate(current)

            history.append(current_loss)
            T *= alpha

            if (self.on_progress is not None
                    and step % max(1, self.progress_every) == 0):
                self.on_progress(step, weighted)

        return RefinerResult(
            best=best, best_raw=best_raw, best_weighted=best_weighted,
            initial_raw=init_raw, initial_weighted=init_weighted,
            n_accepted=n_accepted, n_proposed=n_proposed,
            elapsed_s=time.time() - t0, history=history,
        )
