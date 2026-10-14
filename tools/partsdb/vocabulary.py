"""Export the generator's vocabulary from the parts database: every plain
box part (brick, plate, tile of any size) LEGO currently makes, with its
real geometry, its statistics token (features.py, so the sampler's
patterns are the official models' patterns) and the colours it comes in.

    python tools/partsdb/vocabulary.py    # -> src/everything2lego/data/vocabulary.json

A part qualifies when its name is a plain size ("Brick 1 x 2", "Plate 2 x 2",
"Tile 1 x 4 with Groove", "Brick 1 x 2 x 5") and it came in at least
MIN_SETS sets since 2010; a colour qualifies when the part came in it in
at least MIN_COLOUR_SETS such sets.
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "set_study"))
from build import DB  # noqa: E402
from features import token  # noqa: E402

OUT = REPO / "src/everything2lego/data/vocabulary.json"
MIN_SETS = 20
MIN_COLOUR_SETS = 2
PLAIN = re.compile(r"^(Brick|Plate|Tile) (\d+) x (\d+)( x (\d+))?( with Groove)?$")
SHAPE = {"Brick": "brick", "Plate": "plate", "Tile": "tile"}


def main() -> None:
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    parts = pd.read_sql("""
        SELECT p.part_num, p.name, p.category, p.n_sets_recent, g.ldraw, g.w_studs, g.d_studs, g.h_plates
        FROM part p JOIN (SELECT part_num, MIN(ldraw) AS ldraw, w_studs, d_studs, h_plates
                          FROM geometry GROUP BY part_num) g ON g.part_num = p.part_num
        WHERE p.category IN ('Bricks', 'Plates', 'Tiles') AND p.n_sets_recent >= ?""",
                        con, params=(MIN_SETS,))
    pc = pd.read_sql("""SELECT part_num, CAST(ldraw_code AS INT) AS code, n_sets_current AS n
                        FROM part_colour WHERE ldraw_code IS NOT NULL AND n_sets_current >= ?""",
                     con, params=(MIN_COLOUR_SETS,))
    colours = {pn: dict(zip(g.code.astype(str), g.n.astype(int))) for pn, g in pc.groupby("part_num")}
    out = []
    for r in parts.itertuples(index=False):
        m = PLAIN.match(r.name or "")
        if not m or not r.ldraw:
            continue
        w, d, h = int(round(r.w_studs)), int(round(r.d_studs)), int(round(r.h_plates))
        if {w, d} != {int(m.group(2)), int(m.group(3))} or h < 1:
            continue                                   # name and geometry disagree
        if not colours.get(r.part_num):
            continue
        out.append({"ldraw": r.ldraw, "part_num": r.part_num, "name": r.name,
                    "shape": SHAPE[m.group(1)], "footprint": [max(w, d), min(w, d)],
                    "height_units": h, "token": token(r.category, r.name, w, d, h),
                    "n_sets": int(r.n_sets_recent), "colours": colours[r.part_num]})
    # one part per (shape, size, height): the most used
    best = {}
    for p in sorted(out, key=lambda p: -p["n_sets"]):
        best.setdefault((p["shape"], *p["footprint"], p["height_units"]), p)
    vocab = sorted(best.values(), key=lambda p: (p["shape"], p["height_units"], p["footprint"]))
    OUT.write_text(json.dumps({"source": "tools/partsdb/vocabulary.py", "min_sets": MIN_SETS,
                               "parts": vocab}, indent=0))
    print(f"vocabulary: {len(vocab)} parts -> {OUT}")
    for p in vocab:
        print(f"  {p['token']:>14} {p['ldraw']:>11} {p['name']:<28} sets {p['n_sets']:>5}  colours {len(p['colours'])}")


if __name__ == "__main__":
    main()
