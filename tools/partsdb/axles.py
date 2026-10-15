"""Axle pairs: a Technic axle rod threaded through another part's axle
hole, as official models use it.  An axle rod has an axle:M connector
along its length and an axis; a hole part has an axlehole:F connector
at the hole centre.  We count a pair when the hole's world position
lies on the rod's world axis within the rod's length.

    python tools/partsdb/axles.py       # -> $LEGO_DATA/analysis/axle_pairs.json

Writes a list of {axle, hole, models, count} for pairs seen in
MIN_MODELS+ distinct models.  Each entry is `axle` (the rod's part_num)
and `hole` (the holed part's part_num).  Later: shapes.py can lift
these to axled assemblies for the sampler.

shortcut: the test uses the rod's connector position + axis to build a
line segment of length `axle_len` (from the rod's bbox along its
axis); when the hole F position is within RADIAL LDU of that line and
inside the rod's length along it, we count a pair.  Rods with no
axle:M connector in the DB are skipped; add them via
tools/partsdb/connectors.py if needed.
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

OUT = DATA / "analysis" / "axle_pairs.json"
MIN_MODELS = 5
RADIAL = 3.0                        # LDU; axle rod radius ~2.5
_CTX: dict = {}


def _init():
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    ldmap = _ld_to_part(con)
    axle_rows = con.execute(
        "SELECT c.part_num, c.x, c.y, c.z, c.fx, c.fy, c.fz, g.x_min, g.x_max, g.y_min, g.y_max, g.z_min, g.z_max "
        "FROM connector c JOIN geometry g USING(part_num) "
        "WHERE c.kind='axle' AND c.gender='M'"
    ).fetchall()
    axles: dict[str, dict] = {}
    for pn, x, y, z, fx, fy, fz, x0, x1, y0, y1, z0, z1 in axle_rows:
        axis = np.array([fx, fy, fz], float)
        n = float(np.linalg.norm(axis))
        if n < 1e-6:
            continue
        axis /= n
        # rod length along its axis = projection span of the bbox corners
        corners = np.array([[x0, y0, z0], [x0, y0, z1], [x0, y1, z0], [x0, y1, z1],
                            [x1, y0, z0], [x1, y0, z1], [x1, y1, z0], [x1, y1, z1]], float)
        proj = corners @ axis
        axles[pn] = {"origin": np.array([x, y, z], float), "axis": axis,
                     "span": (float(proj.min()), float(proj.max()))}
    hole_rows = con.execute(
        "SELECT part_num, x, y, z FROM connector WHERE kind='axlehole' AND gender='F'"
    ).fetchall()
    holes: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for pn, x, y, z in hole_rows:
        holes[pn].append((float(x), float(y), float(z)))
    _CTX["ld_to_pn"] = ldmap
    _CTX["axles"] = axles
    _CTX["holes"] = holes


def _apply(M: np.ndarray, p: np.ndarray) -> np.ndarray:
    return (M @ np.array([p[0], p[1], p[2], 1.0]))[:3]


def _apply_dir(M: np.ndarray, d: np.ndarray) -> np.ndarray:
    v = M[:3, :3] @ d
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-6 else v


def pairs_in(f: Path) -> Counter:
    from ldraw_read import flatten
    try:
        parts = flatten(f)
    except Exception:
        return Counter()
    ldm, axles, holes = _CTX["ld_to_pn"], _CTX["axles"], _CTX["holes"]
    rods: list[tuple[str, np.ndarray, np.ndarray, tuple[float, float]]] = []
    hol: list[tuple[str, np.ndarray]] = []
    for p in parts:
        pn = ldm.get(p.name)
        if pn is None:
            continue
        if pn in axles:
            a = axles[pn]
            rods.append((pn, _apply(p.M, a["origin"]), _apply_dir(p.M, a["axis"]), a["span"]))
        if pn in holes:
            for h in holes[pn]:
                hol.append((pn, _apply(p.M, np.array(h, float))))
    out: Counter = Counter()
    for rpn, ro, rd, (lo, hi) in rods:
        for hpn, hw in hol:
            if rpn == hpn:
                continue
            v = hw - ro
            t = float(v @ rd)
            radial = float(np.linalg.norm(v - t * rd))
            if radial < RADIAL and lo - 2.0 <= t <= hi + 2.0:
                out[(rpn, hpn)] += 1
    return out


def main() -> None:
    files = sorted(OMR.glob("*.mpd")) + sorted(OMR.glob("*.ldr"))
    seen, models = Counter(), Counter()
    with Pool(max(1, (os.cpu_count() or 2) - 1), initializer=_init) as pool:
        for got in pool.imap_unordered(pairs_in, files, chunksize=4):
            seen.update(got)
            models.update(got.keys())
    pairs = [{"axle": a, "hole": h, "models": models[(a, h)], "count": n}
             for (a, h), n in seen.most_common() if models[(a, h)] >= MIN_MODELS]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(pairs, indent=1))
    print(f"axle pairs: {len(pairs)} seen in {MIN_MODELS}+ models -> {OUT}")
    for p in pairs[:15]:
        print(f"  axle {p['axle']:>8} + hole {p['hole']:<12} {p['models']} models")


if __name__ == "__main__":
    main()
