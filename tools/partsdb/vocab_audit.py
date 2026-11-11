"""Which parts real building sets use that the generator's vocabulary lacks, and the filter that drops each:
no geometry (no LDraw file), category (not in shapes.CATEGORIES, or NON_SYSTEM), sets (< MIN_SETS), shape (shapes.shape
refuses it: off the stud grid, or too big), colours (none on record).  Ranked by the share of building sets using it.

    .venv-arm/bin/python tools/partsdb/vocab_audit.py [--kind building] [--top 60]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "set_study")]
from build import DB, NON_SYSTEM  # noqa: E402
from shapes import CATEGORIES, MIN_SETS  # noqa: E402
from vocabulary import OUT  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="building"); ap.add_argument("--top", type=int, default=60)
    ap.add_argument("--min-share", type=float, default=0.05, help="parts used by at least this share of the kind's sets")
    a = ap.parse_args()
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    have = {p["part_num"] for p in json.loads(OUT.read_text())["parts"]}
    shaped = {r[0] for r in con.execute("SELECT part_num FROM part_shape")}
    geo = {r[0] for r in con.execute("SELECT part_num FROM geometry WHERE ldraw IS NOT NULL")}
    col = {r[0] for r in con.execute("SELECT part_num FROM part_colour WHERE ldraw_code IS NOT NULL")}
    rows = con.execute("""SELECT k.part_num, p.name, p.category, p.n_sets, k.set_share FROM part_kind k
                          JOIN part p ON p.part_num = k.part_num WHERE k.kind = ? AND k.set_share >= ?
                          ORDER BY k.set_share DESC""", (a.kind, a.min_share)).fetchall()
    why = {}
    for pn, name, cat, n, ss in rows:
        if pn in have:
            continue
        why[pn] = ("category" if cat in NON_SYSTEM or cat not in CATEGORIES else "sets" if (n or 0) < MIN_SETS
                   else "no geometry" if pn not in geo else "shape" if pn not in shaped
                   else "colours" if pn not in col else "merged")      # same geometry as a kept mould
    print(f"{len(rows)} parts used by >= {a.min_share:.0%} of {a.kind} sets; in the vocabulary {len(rows) - len(why)}; "
          "missing by filter: " + ", ".join(f"{k} {v}" for k, v in Counter(why.values()).most_common()))
    cats = Counter((c, why[pn]) for pn, _, c, _, _ in rows if pn in why and why[pn] != "merged")
    print("missing by category and filter: " + "; ".join(f"{c} / {w} {n}" for (c, w), n in cats.most_common(15)))
    shown = 0
    for pn, name, cat, n, ss in rows:
        if pn in why and why[pn] != "merged" and shown < a.top:
            print(f"  {ss:5.1%}  {why[pn]:11s} {cat[:26]:26s} {pn:10s} {name[:60]}"); shown += 1


if __name__ == "__main__":
    main()
