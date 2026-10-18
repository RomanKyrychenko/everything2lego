"""SNOT pairs: two parts whose side-facing stud and antistud (studhole)
meet in the same place of an official model.  A SNOT connection has a
stud of part A at world position P with normal N (horizontal direction)
mating with an antistud of part B at the same P with normal -N.

    python tools/partsdb/snot_pairs.py   # -> $LEGO_DATA/analysis/snot_pairs.json

Writes a list of {a, b, yaw, dy, dx, dz, count, models} for part pairs
seen in MIN_MODELS+ distinct OMR sets.  Later: shapes.py can lift these
to composite SNOT units for the sampler to place in flat-wall regions.

This mirrors hinges.py (fingers → fingers) but uses stud (M) and antistud (F)
with directional matching; only side-facing connectors are considered (we
already catch the plain top-stud case via normal placements).
"""
from __future__ import annotations

import json
import math
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

OUT = DATA / "analysis" / "snot_pairs.json"
MIN_MODELS = 3                       # looser than hinges (5): SNOT patterns are rarer in OMR
POS_TOL = 1.0                        # LDU
DIR_TOL = 0.1                        # cosine between opposite normals
_CTX: dict = {}


def _is_side(f: tuple[float, float, float]) -> bool:
    """Normal direction is horizontal (not plain top/bottom studs)."""
    fx, fy, fz = f
    return abs(fy) < 0.5 and (abs(fx) > 0.5 or abs(fz) > 0.5)


def _init():
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    ldmap = _ld_to_part(con)
    studs: dict[str, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    anti: dict[str, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    rows = con.execute("SELECT part_num, kind, x, y, z, fx, fy, fz FROM connector "
                        "WHERE kind IN ('stud', 'antistud')").fetchall()
    for pn, kind, x, y, z, fx, fy, fz in rows:
        if not _is_side((fx, fy, fz)):
            continue
        p = np.array([x, y, z], float)
        n = np.array([fx, fy, fz], float)
        (studs if kind == "stud" else anti)[pn].append((p, n))
    _CTX["ld_to_pn"] = ldmap
    _CTX["studs"] = studs
    _CTX["anti"] = anti


AXES = {
    (1, 0, 0): "x+", (-1, 0, 0): "x-",
    (0, 1, 0): "y+", (0, -1, 0): "y-",
    (0, 0, 1): "z+", (0, 0, -1): "z-",
}


def _rot_key(R: np.ndarray) -> str | None:
    """Classify a rotation into one of the 24 cube symmetries by looking at
    where the +X and +Y basis vectors land.  Each entry is a 2-letter code
    (e.g. 'x+y+' means identity, 'x+z-' means 90 deg around +X)."""
    def ax(v):
        for a, label in AXES.items():
            if abs(v[0] - a[0]) < 0.15 and abs(v[1] - a[1]) < 0.15 and abs(v[2] - a[2]) < 0.15:
                return label
        return None
    rx = ax(R[:, 0])
    ry = ax(R[:, 1])
    if rx is None or ry is None:
        return None
    return rx + ry


def _pose_key(M_rel: np.ndarray) -> tuple | None:
    """Quantise M_rel: 24-way rotation code (any cube symmetry) + grid-aligned
    translation.  Mirrors hinges._pose_key but allows SNOT-style rotations
    (90 deg around X or Z, not just around Y)."""
    R, t = M_rel[:3, :3], M_rel[:3, 3]
    rot = _rot_key(R)
    if rot is None:
        return None
    dx = int(round(t[0] / 20.0))
    dy = int(round(-t[1] / 8.0))
    dz = int(round(t[2] / 20.0))
    if abs(t[0] - 20 * dx) > 1.0 or abs(-t[1] - 8 * dy) > 1.0 or abs(t[2] - 20 * dz) > 1.0:
        return None
    return (rot, dy, dx, dz)


def _transform(points_normals, M):
    """Apply 4x4 M to each (p, n): rotate normal, translate+rotate point."""
    R = M[:3, :3]
    T = M[:3, 3]
    out = []
    for p, n in points_normals:
        pw = R @ p + T
        nw = R @ n
        out.append((pw, nw))
    return out


def pairs_in(f: Path) -> dict:
    """{(a, b, yaw, dy, dx, dz): count} — grid-aligned relative poses where
    a side-stud of a meets a side-antistud of b at the same world position
    with opposite normals.  `a < b` by sort order."""
    from ldraw_read import flatten
    try:
        parts = flatten(f)
    except Exception:
        return {}
    ldm, studs, anti = _CTX["ld_to_pn"], _CTX["studs"], _CTX["anti"]
    # Each part instance: (pn, world studs, world antistuds, M)
    inst = []
    for p in parts:
        pn = ldm.get(p.name)
        if pn is None:
            continue
        if not studs.get(pn) and not anti.get(pn):
            continue
        ws = _transform(studs.get(pn, []), p.M)
        wa = _transform(anti.get(pn, []), p.M)
        inst.append((pn, ws, wa, p.M))
    out: dict = {}
    for i, (a, ws_a, wa_a, Ma) in enumerate(inst):
        for j in range(len(inst)):
            if i == j:
                continue
            b, ws_b, wa_b, Mb = inst[j]
            # Stud of A meets antistud of B at same point with opposite normal
            match = False
            for (pa, na) in ws_a:
                for (pb, nb) in wa_b:
                    if np.linalg.norm(pa - pb) < POS_TOL and np.dot(na, nb) < -1 + DIR_TOL:
                        match = True
                        break
                if match:
                    break
            if not match:
                continue
            # Canonical order (part_num smaller first)
            if a <= b:
                M_rel = np.linalg.inv(Ma) @ Mb
                pair = (a, b)
            else:
                M_rel = np.linalg.inv(Mb) @ Ma
                pair = (b, a)
            key = _pose_key(M_rel)
            if key is None:
                continue
            full = pair + key
            out[full] = out.get(full, 0) + 1
    return out


def main() -> None:
    files = sorted(OMR.glob("*.mpd")) + sorted(OMR.glob("*.ldr"))
    seen: Counter = Counter()
    pair_models: dict = {}
    with Pool(max(1, (os.cpu_count() or 2) - 1), initializer=_init) as pool:
        for idx, got in enumerate(pool.imap_unordered(pairs_in, files, chunksize=4)):
            if not got:
                continue
            seen.update(got)
            for key in got:
                a, b = key[0], key[1]
                pair_models.setdefault((a, b), set()).add(idx)
    best: dict = {}
    for key, n in seen.most_common():
        a, b = key[0], key[1]
        if (a, b) in best:
            continue
        if len(pair_models.get((a, b), ())) < MIN_MODELS:
            continue
        rot, dy, dx, dz = key[2], key[3], key[4], key[5]
        best[(a, b)] = {"a": a, "b": b, "rot": rot, "dy": dy, "dx": dx, "dz": dz,
                         "models": len(pair_models[(a, b)]), "count": n}
    pairs = sorted(best.values(), key=lambda p: -p["models"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(pairs, indent=1))
    print(f"snot pairs: {len(pairs)} grid-aligned in {MIN_MODELS}+ models -> {OUT}")
    for p in pairs[:15]:
        print(f"  {p['a']:>8} + {p['b']:<8}  rot={p['rot']}  dy={p['dy']:+d}p dx={p['dx']:+d}s dz={p['dz']:+d}s  {p['models']} models")


if __name__ == "__main__":
    main()
