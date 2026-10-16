"""Post-sampler clean-up passes.  Each takes a list[Placement] and
returns (kept, pruned_count).  Opt-in from sampler_demo; both the
Python and the native engine's output passes through them.

- prune_unsupported: drops placements with no vertical supporter below.
- drop_isolated_clusters: drops small connected clusters (floating
  specks that survived the sampler and prune step).
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


def _adjacent(a: Placement, b: Placement) -> bool:
    """Any shared cell face (stud mating or side touch)."""
    if a.y_plate + a.part.height_units == b.y_plate or b.y_plate + b.part.height_units == a.y_plate:
        return _overlap(a, b)
    if a.y_plate == b.y_plate or (a.y_plate < b.y_plate + b.part.height_units
                                   and b.y_plate < a.y_plate + a.part.height_units):
        # same vertical span: touch if footprints share an edge
        ax0, ax1 = a.x_stud, a.x_stud + a.w_world
        az0, az1 = a.z_stud, a.z_stud + a.d_world
        bx0, bx1 = b.x_stud, b.x_stud + b.w_world
        bz0, bz1 = b.z_stud, b.z_stud + b.d_world
        touch_x = ax1 == bx0 or bx1 == ax0
        touch_z = az1 == bz0 or bz1 == az0
        z_overlap = az0 < bz1 and bz0 < az1
        x_overlap = ax0 < bx1 and bx0 < ax1
        return (touch_x and z_overlap) or (touch_z and x_overlap)
    return False


def drop_isolated_clusters(placements: list[Placement], min_cluster: int = 2
                            ) -> tuple[list[Placement], int]:
    """Drop connected clusters of fewer than `min_cluster` placements.
    Adjacency = stud-mating OR side-touch at a shared footprint edge.

    Keep `min_cluster` small (1-2) in builds with SNOT or non-standard
    joins: a 20-part SNOT-held engine block that only touches the main
    build via side connectors looks like an "isolated cluster" here, and
    this function doesn't see those joins — so an aggressive setting
    will drop legitimate structure."""
    n = len(placements)
    if n <= min_cluster:
        return list(placements), 0
    # bucket by y_plate for cheaper neighbour queries
    by_y: dict[int, list[int]] = defaultdict(list)
    for i, p in enumerate(placements):
        for y in range(p.y_plate, p.y_plate + p.part.height_units + 1):
            by_y[y].append(i)
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    for p_idx, p in enumerate(placements):
        for y in range(p.y_plate - 1, p.y_plate + p.part.height_units + 1):
            for q_idx in by_y.get(y, ()):
                if q_idx <= p_idx:
                    continue
                if _adjacent(p, placements[q_idx]):
                    union(p_idx, q_idx)
    sizes: dict[int, int] = defaultdict(int)
    for i in range(n):
        sizes[find(i)] += 1
    kept = [p for i, p in enumerate(placements) if sizes[find(i)] >= min_cluster]
    return kept, n - len(kept)


