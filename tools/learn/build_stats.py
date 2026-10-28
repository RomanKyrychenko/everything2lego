"""Readable statistics of a build next to real building sets, ranked by how far outside the real range it falls.

Each statistic is something you can see in a render: the mix of bricks / plates / tiles / slopes / round / 1x1
parts, how long the bricks are, how many studded tops are left bare, how often a brick sits exactly on a brick of
the same size (stacked joints instead of a bonded wall), how many parts are mounted sideways, colours per 100 parts.

    .venv-arm/bin/python tools/learn/build_stats.py build.ldr [--kind building --n 150]
"""
from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "set_study")]
DATA = Path("/Volumes/PRO-G40/lego_data")
_cat: dict = {}; _bb: dict = {}


def category(name: str) -> str:
    """Rebrickable category of an LDraw part ('' when unknown)."""
    if name not in _cat:
        db = sqlite3.connect(str(DATA / "parts.sqlite"))
        r = db.execute("select p.category from ldraw_name l join part p on p.part_num = l.part_num where l.ldraw = ?", (name,)).fetchone()
        _cat[name] = (r[0] or "") if r else ""
    return _cat[name]


def stats(path: Path) -> dict | None:
    from ldraw_read import flatten, part_bbox
    from realism import _parts_with_grid
    ps = flatten(path)
    for q in ps:
        if q.name not in _bb:
            try: _bb[q.name] = part_bbox(q.name)
            except Exception: _bb[q.name] = None                                  # noqa: BLE001
    it = _parts_with_grid(ps, _bb)
    if len(it) < 60: return None
    lo = np.array([x[3] for x in it]); hi = np.array([x[4] for x in it]); ext = hi - lo
    w = np.maximum(ext[:, 0], ext[:, 2]); dd = np.minimum(ext[:, 0], ext[:, 2]); h = ext[:, 1]
    cats = [category(x[0]) for x in it]; low = [c.lower() for c in cats]
    is_tile = np.array(["tile" in c for c in low]); is_slope = np.array(["slope" in c for c in low])
    is_round = np.array(["round" in c or "cone" in c or "dish" in c for c in low])
    is_brick = np.array([c.startswith("bricks") and "slope" not in c for c in low]) & (h > 2.5)
    is_plate = np.array(["plates" in c for c in low]) & ~is_tile
    up = np.array([x[2][1, 1] > 0.99 for x in it])
    one = (w <= 1.01) & (dd <= 1.01)
    # studs left bare: a studded, upright part with no part's box covering the cells right above its top
    occ = defaultdict(int); base = lo.min(0)
    for k in range(len(it)):
        for y in range(int(round(lo[k, 1] - base[1])), int(round(hi[k, 1] - base[1]))):
            for x in range(int(np.floor(lo[k, 0] - base[0])), int(np.ceil(hi[k, 0] - base[0]))):
                for z in range(int(np.floor(lo[k, 2] - base[2])), int(np.ceil(hi[k, 2] - base[2]))):
                    occ[(y, x, z)] += 1
    studded = (is_brick | is_plate) & up; bare = 0
    for k in np.flatnonzero(studded):
        y = int(round(hi[k, 1] - base[1]))
        cells = [(y, x, z) for x in range(int(np.floor(lo[k, 0] - base[0])), int(np.ceil(hi[k, 0] - base[0])))
                 for z in range(int(np.floor(lo[k, 2] - base[2])), int(np.ceil(hi[k, 2] - base[2])))]
        bare += sum(occ[c] == 0 for c in cells) / max(1, len(cells)) > 0.5
    # stacked joints: a brick right on top of a brick with the same footprint, edges aligned
    key = defaultdict(list)
    for k in np.flatnonzero(is_brick): key[(round(lo[k, 0], 1), round(lo[k, 2], 1), round(hi[k, 0], 1), round(hi[k, 2], 1))].append(k)
    stacked = sum(1 for ks in key.values() for a in ks for b in ks if a != b and abs(lo[b, 1] - hi[a, 1]) < 0.3)
    nb = max(1, is_brick.sum())
    return {"bricks": is_brick.mean(), "plates": is_plate.mean(), "tiles": is_tile.mean(), "slopes": is_slope.mean(),
            "round": is_round.mean(), "1x1": one.mean(), "brick_len": float(w[is_brick].mean()) if is_brick.any() else 0.0,
            "brick_len>=4": float((w[is_brick] >= 3.99).mean()) if is_brick.any() else 0.0,
            "bare_studs": bare / max(1, studded.sum()), "stacked_bricks": stacked / nb, "sideways": 1 - up.mean(),
            "colours/100": 100 * len({x[1] for x in it}) / len(it), "parts": len(it)}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("builds", type=Path, nargs="+")
    ap.add_argument("--kind", default="building"); ap.add_argument("--n", type=int, default=150); ap.add_argument("--min-parts", type=int, default=200)
    a = ap.parse_args()
    kinds = {r["id"]: r["kind"] for r in csv.DictReader(open(DATA / "analysis" / "set_kinds.csv"))}
    files = [f for f in sorted((DATA / "omr" / "files").glob("*.mpd")) if kinds.get(f.stem.split("_")[0]) == a.kind]
    real = []
    for f in files:
        try: s = stats(f)
        except Exception: continue                                                 # noqa: BLE001
        if s and s["parts"] >= a.min_parts: real.append(s)
        if len(real) >= a.n: break
    print(f"real {a.kind} sets: {len(real)} (>= {a.min_parts} parts)")
    ours = [(p, stats(p)) for p in a.builds]
    keys = [k for k in real[0] if k != "parts"]
    rows = []
    for k in keys:
        v = np.array([r[k] for r in real]); p10, med, p90 = np.percentile(v, [10, 50, 90])
        gap = [(s[k] - med) / max(1e-9, (p90 - p10) / 2) for _, s in ours]               # in half-ranges of the real spread
        rows.append((max(abs(g) for g in gap), k, p10, med, p90, [s[k] for _, s in ours], gap))
    names = [p.parent.name or p.stem for p, _ in ours]
    print(f"{'statistic':15s} {'real p10':>8} {'median':>7} {'p90':>7}  " + "  ".join(f"{n[:14]:>14}" for n in names))
    for _, k, p10, med, p90, vals, gap in sorted(rows, reverse=True):
        print(f"{k:15s} {p10:8.3f} {med:7.3f} {p90:7.3f}  " + "  ".join(f"{v:8.3f} ({g:+4.1f})" for v, g in zip(vals, gap)))
    print("(+/-x.x = distance from the real median in half real-ranges; |x| > 1 is outside the p10-p90 band)")


if __name__ == "__main__":
    main()
