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


def part_costs(ldraws: list[str], role: str | None = None, *, min_evidence: int = 20) -> dict[str, float]:
    """-log P(part | role) over `ldraws` (add-one smoothed).  Uses the role's
    counts when the official models show the role with these parts at least
    `min_evidence` times, else how often sets use each part at all."""
    u = _usage()
    counts = u.get("roles", {}).get(role or "", {})
    n = [counts.get(ld, 0) for ld in ldraws]
    if sum(n) < min_evidence:
        n = [u.get("all", {}).get(ld, 0) for ld in ldraws]
    total = sum(n) + len(ldraws)
    return {ld: -math.log((k + 1) / total) for ld, k in zip(ldraws, n)}


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
