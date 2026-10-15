"""Post-sampler support fixer: drop placements with no vertical supporter
below.

A `Placement` is supported iff `y_plate == 0` or some other placement
directly below (its `y_plate + height_units` == this one's `y_plate`)
has a non-zero footprint overlap in (x, z).  Removing an unsupported
placement can leave its own supporters unsupported, so the sweep
iterates until stable.

Simpler than `sampler.State._supported` which checks stud-to-antistud
mating — this function keeps side-held placements too (SNOT support is
hard to recover from the placement-level view alone), but drops the
cleanly-floating ones.  Opt-in via `sampler_demo --prune-unsupported`;
both the Python and the native engine's output passes through it.
"""

from __future__ import annotations

from collections import defaultdict

from .pack import Placement


def _overlap(a: Placement, b: Placement) -> bool:
    if a.x_stud + a.w_world <= b.x_stud or b.x_stud + b.w_world <= a.x_stud:
        return False
    if a.z_stud + a.d_world <= b.z_stud or b.z_stud + b.d_world <= a.z_stud:
        return False
    return True


def prune_unsupported(placements: list[Placement]) -> tuple[list[Placement], int]:
    """Return (kept, pruned_count) after iterating until no placement is
    unsupported.  Keeps a placement if it is on the ground, or if any
    stud-mating neighbour exists (another placement whose top is at this
    one's bottom plane, or whose bottom is at this one's top plane) with
    footprint overlap — matching sampler.State._supported's rule, modulo
    side (SNOT) support which can't be recovered from placements alone.
    """
    kept = list(placements)
    pruned = 0
    while True:
        by_top: dict[int, list[Placement]] = defaultdict(list)
        by_bot: dict[int, list[Placement]] = defaultdict(list)
        for p in kept:
            by_top[p.y_plate + p.part.height_units].append(p)
            by_bot[p.y_plate].append(p)
        new_kept = []
        for p in kept:
            if p.y_plate == 0:
                new_kept.append(p)
                continue
            below = by_top.get(p.y_plate, ())
            above = by_bot.get(p.y_plate + p.part.height_units, ())
            if any(q is not p and _overlap(p, q) for q in below) \
               or any(q is not p and _overlap(p, q) for q in above):
                new_kept.append(p)
            else:
                pruned += 1
        if len(new_kept) == len(kept):
            return kept, pruned
        kept = new_kept
