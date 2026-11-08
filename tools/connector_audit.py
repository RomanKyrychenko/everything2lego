"""Compare the builder's top studs / bottom holes (vocabulary.json) with BrickNet's connector labels, part by part,
on the stud grid; and the parts database's own rule (tools/partsdb/shapes.py, from the LDCad shadow library) with them.

    .venv-arm/bin/python tools/connector_audit.py [--db] [--show 40]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "partsdb"))
from vocabulary import DB, OUT, bricknet_grid, bricknet_labels  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", action="store_true", help="audit the parts database's studs/holes instead of vocabulary.json")
    ap.add_argument("--show", type=int, default=40)
    a = ap.parse_args()
    labels, aliases = bricknet_labels()
    parts = json.loads(OUT.read_text())["parts"]
    if a.db:
        con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
        own = {pn: (json.loads(t), json.loads(b)) for pn, t, b in
               con.execute("SELECT part_num, studs_top, holes_bottom FROM part_shape")}
        parts = [{**p, "studs_top": own[p["part_num"]][0], "holes_bottom": own[p["part_num"]][1]}
                 for p in parts if p["part_num"] in own]
    tally, rows = Counter(), []
    for p in parts:
        st = p["ldraw"].lower().removesuffix(".dat")
        lab = labels.get(aliases.get(st, st) + ".dat")
        if lab is None:
            tally["not in BrickNet"] += 1
            continue
        top, bot, off = bricknet_grid(p["size"], p["origin"], lab)
        top, bot = set(map(tuple, top)), set(map(tuple, bot))
        vt, vb = set(map(tuple, p["studs_top"])), set(map(tuple, p["holes_bottom"]))
        diff = {"extra top": vt - top, "missing top": top - vt, "extra bottom": vb - bot, "missing bottom": bot - vb}
        bad = [k for k, v in diff.items() if v]
        tally["agree" if not bad else "disagree"] += 1
        tally.update(bad)
        if bad:
            rows.append((p["n_sets"], st, p["name"][:40], {k: len(v) for k, v in diff.items() if v}, off))
    print(dict(tally))
    for r in sorted(rows, reverse=True)[:a.show]:
        print(f"{r[0]:5d} {r[1]:9s} {r[2]:40s} {r[3]}  off-grid in BrickNet {r[4]}")


if __name__ == "__main__":
    main()
