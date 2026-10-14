"""How real sets use parts: statistics exported from the parts database
(tools/partsdb/export.py -> data/part_usage.json), for decisions made by
the numbers rather than by hand-written preference.

    costs = part_costs(["3037.dat", "3039.dat", "4445.dat"], role="roof")
    # {"3037.dat": 0.6, "3039.dat": 1.6, "4445.dat": 2.2}  (-log probability)

A run split into parts by the least total cost is the split real sets make
most likely for that role: each part costs -log of how often sets use it
there, among the parts that could go there.  Common parts are cheap, rare
ones dear, and fewer parts still win when the saving is real (every part
adds a cost).
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from importlib.resources import files


@lru_cache(maxsize=1)
def _usage() -> dict:
    try:
        return json.loads(files("everything2lego.data").joinpath("part_usage.json").read_text())
    except FileNotFoundError:
        return {"all": {}, "roles": {}, "joints": {}}


def part_costs(ldraws: list[str], role: str | None = None, *, min_evidence: int = 20,
               prior: float = 20.0) -> dict[str, float]:
    """-log P(part | role) over `ldraws`.  The role's counts from official
    models are sparse (a 1x16 brick never appears in their walls), so they
    are smoothed toward how often sets use each part at all: P = (n_role +
    prior * share_all) / (N_role + prior).  A part the role never shows
    then costs about what its overall rarity says, not the flat 1/N an
    add-one count would give it.  Without enough role evidence
    (`min_evidence`), the overall shares alone."""
    u = _usage()
    allc = u.get("all", {})
    g = [allc.get(ld, 0) + 1 for ld in ldraws]                  # overall, add-one
    gz = float(sum(g))
    share = [x / gz for x in g]
    counts = u.get("roles", {}).get(role or "", {})
    n = [counts.get(ld, 0) for ld in ldraws]
    if sum(n) < min_evidence:
        return {ld: -math.log(sh) for ld, sh in zip(ldraws, share)}
    total = sum(n) + prior
    return {ld: -math.log((k + prior * sh) / total) for ld, k, sh in zip(ldraws, n, share)}


def joint_ratio(male: str, female: str) -> float | None:
    """How often official models join `male`'s studs (or bars, pins...) to
    `female`, against chance (1.0); None when the pair has no record."""
    rec = _usage().get("joints", {}).get(male, {}).get(female)
    return rec[1] if rec else None


def joint_realism(placements, *, rare: float = 0.2) -> dict[str, float]:
    """How the stud joints of a build compare with official models: of the
    joints whose part pair the models record, the share real sets make no
    more than `rare` times as often as chance would ("unlike sets"), and
    the share of joints whose pair the models never show at all."""
    from .stability import bottom_cells, top_studs
    tops: dict[int, dict] = {}
    for i, p in enumerate(placements):
        tops.setdefault(p.y_plate + p.part.height_units, {}).update({s: i for s in top_studs(p)})
    seen: set[tuple[int, int]] = set()
    known = rare_n = unknown = 0
    for i, p in enumerate(placements):
        below = tops.get(p.y_plate, {})
        for s in bottom_cells(p):
            j = below.get(s)
            if j is None or (j, i) in seen:
                continue
            seen.add((j, i))
            r = joint_ratio(placements[j].part.ldraw, p.part.ldraw)   # j's studs into i
            if r is None:
                unknown += 1
            else:
                known += 1
                rare_n += r <= rare
    total = known + unknown
    return {"joints": total,
            "unlike_sets_pct": round(100 * rare_n / known, 1) if known else 0.0,
            "unrecorded_pct": round(100 * unknown / total, 1) if total else 0.0}


UNRECORDED_JOINT = 1.0   # cost of a joint the official models never show (about -log 0.37)


def joint_cost(male: str, female: str) -> float:
    """-log of how often official models make this joint against chance,
    floored at 0 (a joint as common as chance or more costs nothing);
    UNRECORDED_JOINT when the pair has no record."""
    r = joint_ratio(male, female)
    if r is None:
        return UNRECORDED_JOINT
    return max(0.0, -math.log(max(r, 1e-3)))


def landing_tax(ldraws: list[str], role: str | None = None) -> dict[str, float]:
    """For each part, the joint cost it can expect from what lands on it:
    sum over the parts `ldraws` (what goes on top, as often as real sets
    use them for `role`) of P(q) * joint_cost(part, q).  A part real sets
    rarely build on — a 1x16 brick in a wall — is dear before anything
    sits on it, not only after."""
    costs = part_costs(ldraws, role)
    probs = {q: math.exp(-c) for q, c in costs.items()}
    z = sum(probs.values()) or 1.0
    return {p: sum(probs[q] / z * joint_cost(p, q) for q in ldraws) for p in ldraws}
