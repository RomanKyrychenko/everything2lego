"""Wheel assemblies: which tyre goes on which rim, as the official models
put them (a tyre placed with its rim's own transform), and each pair's
shape as one grid part — the rim's connectors (its hub's pin or axle
hole), the pair's measured shape centred in the smallest grid box holding
it (a wheel's diameter is no whole number of studs).

    python tools/partsdb/wheels.py      # -> $LEGO_DATA/analysis/wheel_pairs.json

shapes.py adds the pairs to part_shape (companion = the tyre).
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "set_study"))
from build import DATA, DB, OMR, _ld_to_part  # noqa: E402

OUT = DATA / "analysis" / "wheel_pairs.json"
MIN_MODELS = 5
_CTX: dict = {}


def _init():
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    ldmap = _ld_to_part(con)
    cat = dict(con.execute("SELECT part_num, name FROM part WHERE category = 'Wheels and Tyres'"))
    _CTX["kind"] = {}
    for ld, pn in ldmap.items():
        name = cat.get(pn)
        if name:
            _CTX["kind"][ld] = "tyre" if name.startswith("Tyre") else "rim" if name.startswith("Wheel") else None


def pairs_in(f: Path) -> Counter:
    from ldraw_read import flatten
    try:
        parts = flatten(f)
    except Exception:
        return Counter()
    kind = _CTX["kind"]
    rims = [p for p in parts if kind.get(p.name) == "rim"]
    tyres = [p for p in parts if kind.get(p.name) == "tyre"]
    out = Counter()
    for r in rims:
        for t in tyres:
            if np.allclose(r.M, t.M, atol=0.5):              # the tyre on this rim, as drawn
                out[(r.name, t.name)] += 1
    return out


def main() -> None:
    files = sorted(OMR.glob("*.mpd")) + sorted(OMR.glob("*.ldr"))
    seen, models = Counter(), Counter()
    with Pool(max(1, (os.cpu_count() or 2) - 1), initializer=_init) as pool:
        for got in pool.imap_unordered(pairs_in, files, chunksize=4):
            seen.update(got)
            models.update(got.keys())
    pairs = [{"rim": r, "tyre": t, "models": models[(r, t)], "count": n}
             for (r, t), n in seen.most_common() if models[(r, t)] >= MIN_MODELS]
    OUT.write_text(json.dumps(pairs, indent=1))
    print(f"wheel pairs: {len(pairs)} seen in {MIN_MODELS}+ models -> {OUT}")
    for p in pairs[:15]:
        print(f"  {p['rim']:>14} + {p['tyre']:<14} {p['models']} models")


if __name__ == "__main__":
    main()
