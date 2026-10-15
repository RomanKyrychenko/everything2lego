"""Hinge pairs: two parts whose finger connectors interlock in the same
place of an official model.  A hinge joint has fingers meshed at a
shared axis; we count a pair when a finger of part A and a finger of
part B land at the same world position in a model.

    python tools/partsdb/hinges.py      # -> $LEGO_DATA/analysis/hinge_pairs.json

Writes a list of {a, b, models, count} for part pairs seen in
MIN_MODELS+ distinct models.  Later: shapes.py can lift these to
composite parts at a grid-aligned angle for the sampler to place.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "set_study"))
from build import DATA, DB, OMR, _ld_to_part  # noqa: E402

OUT = DATA / "analysis" / "hinge_pairs.json"
MIN_MODELS = 5
TOL = 1.0                           # LDU; interlocking finger centres coincide to <~0.5
_CTX: dict = {}


def _init():
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    ldmap = _ld_to_part(con)
    rows = con.execute("SELECT part_num, x, y, z FROM connector WHERE kind='finger'").fetchall()
    fingers: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for pn, x, y, z in rows:
        fingers[pn].append((float(x), float(y), float(z)))
    _CTX["ld_to_pn"] = ldmap
    _CTX["fingers"] = fingers


def pairs_in(f: Path) -> Counter:
    from ldraw_read import flatten
    try:
        parts = flatten(f)
    except Exception:
        return Counter()
    ldm, fingers = _CTX["ld_to_pn"], _CTX["fingers"]
    inst: list[tuple[str, np.ndarray]] = []
    for p in parts:
        pn = ldm.get(p.name)
        if pn is None:
            continue
        fp = fingers.get(pn)
        if not fp:
            continue
        W = (p.M @ np.c_[np.array(fp, float), np.ones(len(fp))].T).T[:, :3]
        inst.append((pn, W))
    out: Counter = Counter()
    for i, (a, wa) in enumerate(inst):
        for j in range(i + 1, len(inst)):
            b, wb = inst[j]
            d = ((wa[:, None, :] - wb[None, :, :]) ** 2).sum(-1).min()
            if d < TOL ** 2:
                out[tuple(sorted((a, b)))] += 1
    return out


def main() -> None:
    files = sorted(OMR.glob("*.mpd")) + sorted(OMR.glob("*.ldr"))
    seen, models = Counter(), Counter()
    with Pool(max(1, (os.cpu_count() or 2) - 1), initializer=_init) as pool:
        for got in pool.imap_unordered(pairs_in, files, chunksize=4):
            seen.update(got)
            models.update(got.keys())
    pairs = [{"a": a, "b": b, "models": models[(a, b)], "count": n}
             for (a, b), n in seen.most_common() if models[(a, b)] >= MIN_MODELS]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(pairs, indent=1))
    print(f"hinge pairs: {len(pairs)} seen in {MIN_MODELS}+ models -> {OUT}")
    for p in pairs[:15]:
        print(f"  {p['a']:>14} + {p['b']:<14} {p['models']} models")


if __name__ == "__main__":
    main()
