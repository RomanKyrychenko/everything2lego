"""Export the generator's vocabulary from the parts database: every part
that sits on the build grid (part_shape: bricks, plates, tiles, slopes,
curves, rounds, wedges, ...) that LEGO makes now, with

  shape      how full each cell of its box is ([y from the bottom][x][z],
             its own frame), and where in it: `bits`, 4x4x4 samples per
             cell as a 64-bit mask (shapes.pack); and where its box
             centre is from its origin
  studs      the cells with a stud on top / a hole underneath, and the
             studs and holes on its sides with the face they point out of
  token      its statistics token (features.py: the official models'
             neighbourhood patterns are counted in these)
  colours    the LDraw colours it came in (2+ sets since 2010), with counts
  kinds      what it is for: per kind of object (building, road_vehicle,
             creature, ...) the share of that kind's sets using it and its
             lift (part_kind; kinds.py)

    python tools/partsdb/vocabulary.py    # -> src/everything2lego/data/vocabulary.json

Parts of the same geometry, connections and token (a mould and its
successor) are one entry: the most used.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "set_study"))
from build import DB, NON_SYSTEM  # noqa: E402
from features import token  # noqa: E402

OUT = REPO / "src/everything2lego/data/vocabulary.json"
MIN_SETS = 30
MIN_COLOUR_SETS = 2
MIN_KIND_SETS = 3


def main() -> None:
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    sh = pd.read_sql("""
        SELECT s.*, p.name, p.category, p.n_sets_recent, g.w_studs, g.d_studs, g.h_plates
        FROM part_shape s JOIN part p ON p.part_num = s.part_num
        JOIN (SELECT part_num, w_studs, d_studs, h_plates FROM geometry GROUP BY part_num) g
             ON g.part_num = s.part_num
        WHERE p.n_sets_recent >= ?""", con, params=(MIN_SETS,))
    sh = sh[~sh.category.isin(NON_SYSTEM)]
    pc = pd.read_sql("""SELECT part_num, CAST(ldraw_code AS INT) AS code, n_sets_current AS n
                        FROM part_colour WHERE ldraw_code IS NOT NULL AND n_sets_current >= ?""",
                     con, params=(MIN_COLOUR_SETS,))
    colours = {pn: dict(zip(g.code.astype(str), g.n.astype(int))) for pn, g in pc.groupby("part_num")}
    pk = pd.read_sql("SELECT part_num, kind, n_sets, set_share, lift FROM part_kind WHERE n_sets >= ?",
                     con, params=(MIN_KIND_SETS,))
    kinds = {pn: {k: {"sets": int(n), "set_share": round(float(ss), 4), "lift": round(float(l), 3)}
                  for k, n, ss, l in zip(g.kind, g.n_sets, g.set_share, g.lift)}
             for pn, g in pk.groupby("part_num")}
    out = []
    for r in sh.itertuples(index=False):
        if not colours.get(r.part_num):
            continue
        tok = token(r.category, r.name, r.w_studs or 0, r.d_studs or 0, r.h_plates or 0)
        out.append({"ldraw": r.ldraw, "part_num": r.part_num, "name": r.name, "category": r.category,
                    "token": tok, "size": [int(r.w), int(r.d), int(r.h)],
                    "origin": [r.origin_x, r.origin_y, r.origin_z], "fill": round(float(r.fill), 3),
                    "occ": json.loads(r.occ), "bits": json.loads(r.bits), "studs_top": json.loads(r.studs_top),
                    "holes_bottom": json.loads(r.holes_bottom), "sides": json.loads(r.sides),
                    "n_sets": int(r.n_sets_recent), "colours": colours[r.part_num],
                    "kinds": kinds.get(r.part_num, {})})
    best = {}
    for p in sorted(out, key=lambda p: -p["n_sets"]):
        key = hashlib.md5(json.dumps([p["token"], p["occ"], p["studs_top"], p["holes_bottom"],
                                      p["sides"]]).encode()).hexdigest()
        best.setdefault(key, p)
    vocab = sorted(best.values(), key=lambda p: (p["category"], p["token"], -p["n_sets"]))
    OUT.write_text(json.dumps({"source": "tools/partsdb/vocabulary.py", "min_sets": MIN_SETS,
                               "parts": vocab}, separators=(",", ":")))
    cats = pd.Series([p["category"] for p in vocab]).value_counts()
    print(f"vocabulary: {len(vocab)} parts -> {OUT} ({OUT.stat().st_size // 1024} KB)")
    print(cats.to_string())


if __name__ == "__main__":
    main()
