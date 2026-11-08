"""Fillers: parts the stud grid refuses (shapes.shape: a window's glass, a door, a shutter -- held by a frame's clips,
not studs) as the official models put them in a grid part: placed unturned relative to it, the centre of its
bounding box strictly inside the part's box (an origin sits on a box's edge: the glass of the window below would pass).  Each (host, filler, offset) seen in MIN_MODELS+ models, with the filler's most used colour there.

    python tools/partsdb/fillers.py      # -> $LEGO_DATA/analysis/filler_pairs.json

shapes.py adds each host with its most used filler to part_shape (companion = the filler, at that offset), as
wheels.py's pairs: one grid part, the pair's shape, the host's connectors.
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
sys.path[:0] = [str(HERE), str(HERE.parent / "set_study")]
from build import DATA, DB, OMR, _ld_to_part  # noqa: E402
from shapes import CATEGORIES, MIN_SETS, triangles  # noqa: E402

OUT = DATA / "analysis" / "filler_pairs.json"
MIN_MODELS = 5
_CTX: dict = {}


def _init():
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    pn_ld = {}
    for ld, pn in _ld_to_part(con).items():
        pn_ld.setdefault(pn, []).append(ld.lower())
    box = {}
    for pn, ld, x0, y0, z0, w, d, h in con.execute("SELECT part_num, ldraw, x0, y0, z0, w, d, h FROM part_shape "
                                                   "WHERE companion IS NULL"):
        box[ld.lower()] = np.array([x0, y0, z0, x0 + 20 * w, y0 + 8 * h, z0 + 20 * d], float)
    shaped = {r[0] for r in con.execute("SELECT part_num FROM part_shape")}
    q = f"SELECT part_num FROM part WHERE category IN ({','.join('?' * len(CATEGORIES))}) AND n_sets >= ?"
    refused = {ld for (pn,) in con.execute(q, (*CATEGORIES, MIN_SETS)) if pn not in shaped for ld in pn_ld.get(pn, [])}
    centre = {}
    for ld in refused:
        try:
            P = triangles(ld).reshape(-1, 3); centre[ld] = (P.min(0) + P.max(0)) / 2
        except Exception:                                # noqa: BLE001
            pass
    _CTX.update(box=box, centre=centre)


def pairs_in(f: Path) -> Counter:
    from ldraw_read import flatten
    try:
        parts = flatten(f)
    except Exception:                                    # noqa: BLE001
        return Counter()
    box, centre = _CTX["box"], _CTX["centre"]
    hosts = [p for p in parts if p.name in box]
    out = Counter()
    if not hosts:
        return out
    HM = np.stack([p.M for p in hosts])
    for p in (p for p in parts if p.name in centre):
        rel = np.linalg.solve(HM, np.broadcast_to(p.M, HM.shape))      # the filler in each host's frame
        for i in np.flatnonzero(np.abs(rel[:, :3, :3] - np.eye(3)).max((1, 2)) < 1e-3):
            t, b = rel[i, :3, 3], box[hosts[i].name]
            c = t + centre[p.name]
            if (c > b[:3] + 0.5).all() and (c < b[3:] - 0.5).all():
                out[(hosts[i].name, p.name, tuple(np.round(t).astype(int).tolist()), p.colour)] += 1
    return out


def main() -> None:
    files = sorted(OMR.glob("*.mpd")) + sorted(OMR.glob("*.ldr"))
    seen, models = Counter(), Counter()
    with Pool(max(1, (os.cpu_count() or 2) - 2), initializer=_init) as pool:
        for got in pool.imap_unordered(pairs_in, files, chunksize=4):
            seen.update(got)
            models.update({k[:3] for k in got})
    colour = {}
    for (h, f, t, c), n in seen.most_common():           # most placed first: each pair's colour
        colour.setdefault((h, f, t), c)
    pairs = [{"host": h, "filler": f, "at": list(t), "colour": colour[(h, f, t)], "models": n}
             for (h, f, t), n in models.most_common() if n >= MIN_MODELS]
    OUT.write_text(json.dumps(pairs, indent=1))
    print(f"filler pairs: {len(pairs)} seen in {MIN_MODELS}+ models -> {OUT}")
    for p in pairs[:25]:
        print(f"  {p['host']:>14} + {p['filler']:<14} at {p['at']} colour {p['colour']}: {p['models']} models")


if __name__ == "__main__":
    main()
