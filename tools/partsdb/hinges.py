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


def _pose_key(M_rel: np.ndarray) -> tuple | None:
    """Quantise M_rel to the sampler's grid: translation to (dy_plate,
    dx_stud, dz_stud), rotation to a yaw in {0, 90, 180, 270} around +Y.
    Returns None for off-grid rotations (true hinge angles like 45°)."""
    R, t = M_rel[:3, :3], M_rel[:3, 3]
    if abs(R[0, 1]) > 0.1 or abs(R[1, 0]) > 0.1 or abs(R[1, 2]) > 0.1 or abs(R[2, 1]) > 0.1:
        return None                                                  # tilted around X or Z
    if abs(R[1, 1] - 1.0) > 0.1:
        return None
    # yaw = atan2(R[0,2], R[0,0]); choose the nearest 90°
    import math as _m
    yaw = int(round(_m.degrees(_m.atan2(R[0, 2], R[0, 0])) / 90.0)) * 90 % 360
    # LDraw: 20 LDU/stud (x, z), 8 LDU/plate (y, y down)
    dx = int(round(t[0] / 20.0))
    dy = int(round(-t[1] / 8.0))                                     # flip because LDraw y points down
    dz = int(round(t[2] / 20.0))
    if abs(t[0] - 20 * dx) > 1.0 or abs(-t[1] - 8 * dy) > 1.0 or abs(t[2] - 20 * dz) > 1.0:
        return None                                                  # off the plate/stud grid
    return (yaw, dy, dx, dz)


def pairs_in(f: Path) -> dict:
    """{(a, b, yaw, dy, dx, dz): count} — grid-aligned relative poses only.
    `a` and `b` are part numbers with a < b (sorted); the pose places B's
    origin relative to A's."""
    from ldraw_read import flatten
    try:
        parts = flatten(f)
    except Exception:
        return {}
    ldm, fingers = _CTX["ld_to_pn"], _CTX["fingers"]
    inst: list[tuple[str, np.ndarray, np.ndarray]] = []
    for p in parts:
        pn = ldm.get(p.name)
        if pn is None:
            continue
        fp = fingers.get(pn)
        if not fp:
            continue
        W = (p.M @ np.c_[np.array(fp, float), np.ones(len(fp))].T).T[:, :3]
        inst.append((pn, W, p.M))
    out: dict = {}
    for i, (a, wa, Ma) in enumerate(inst):
        for j in range(i + 1, len(inst)):
            b, wb, Mb = inst[j]
            d = ((wa[:, None, :] - wb[None, :, :]) ** 2).sum(-1).min()
            if d >= TOL ** 2:
                continue
            # relative pose of the second part in the first part's frame;
            # swap so (part_num_smaller, part_num_larger) is the key order
            if a <= b:
                M_rel = np.linalg.inv(Ma) @ Mb
                pair = (a, b)
            else:
                M_rel = np.linalg.inv(Mb) @ Ma
                pair = (b, a)
            key = _pose_key(M_rel)
            if key is None:
                continue
            full_key = pair + key
            out[full_key] = out.get(full_key, 0) + 1
    return out


def main() -> None:
    files = sorted(OMR.glob("*.mpd")) + sorted(OMR.glob("*.ldr"))
    seen: Counter = Counter()                                        # (a, b, yaw, dy, dx, dz) -> count
    pair_models: dict = {}                                           # (a, b) -> set of model files that use any pose
    with Pool(max(1, (os.cpu_count() or 2) - 1), initializer=_init) as pool:
        for idx, got in enumerate(pool.imap_unordered(pairs_in, files, chunksize=4)):
            if not got:
                continue
            seen.update(got)
            for key in got:
                a, b = key[0], key[1]
                pair_models.setdefault((a, b), set()).add(idx)
    # for each (a, b), pick the most common grid pose
    best: dict = {}
    for key, n in seen.most_common():
        a, b = key[0], key[1]
        if (a, b) in best:
            continue
        if len(pair_models.get((a, b), ())) < MIN_MODELS:
            continue
        yaw, dy, dx, dz = key[2], key[3], key[4], key[5]
        best[(a, b)] = {"a": a, "b": b, "yaw": yaw, "dy": dy, "dx": dx, "dz": dz,
                         "models": len(pair_models[(a, b)]), "count": n}
    pairs = sorted(best.values(), key=lambda p: -p["models"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(pairs, indent=1))
    print(f"hinge pairs: {len(pairs)} grid-aligned in {MIN_MODELS}+ models -> {OUT}")
    for p in pairs[:15]:
        print(f"  {p['a']:>8} + {p['b']:<8}  yaw={p['yaw']:>3}  dy={p['dy']:+d}p dx={p['dx']:+d}s dz={p['dz']:+d}s  {p['models']} models")


if __name__ == "__main__":
    main()
