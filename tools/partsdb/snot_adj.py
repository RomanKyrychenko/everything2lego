"""SNOT pairs via physical adjacency: in each OMR model, find pairs
of adjacent parts where at least one placement has a non-yaw rotation
(studs not up).  The pair's relative pose captures the SNOT template.

    python tools/partsdb/snot_adj.py   # -> $LEGO_DATA/analysis/snot_adj.json

Different from snot_pairs.py (which looked for connector coincidences,
which don't exist for most SNOT mating): this one directly records
"regular brick with a sideways plate next to it" from the placement
geometry.

Each record carries B's full relative rotation in A's frame — one of the
24 cube symmetries, encoded as a 2-letter axis code `rx+ry+` naming where
B's +X and +Y basis vectors point in A's frame.  Grid offsets `(dx, dy,
dz)` are also expressed in A's frame (A's horizontal yaw backed out), so
the template is reusable regardless of how A is placed in the model.
face_a/face_b and world-frame dyaw are kept redundantly for compatibility
with earlier readers; new code should prefer `rot` + A-frame `(dx,dy,dz)`.
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
from neighbours import neighbour_pairs, model_parts, part_classes, STUD, PLATE  # noqa: E402

OUT = DATA / "analysis" / "snot_adj.json"
MIN_MODELS = 3

AXES = {
    (1, 0, 0): "x+", (-1, 0, 0): "x-",
    (0, 1, 0): "y+", (0, -1, 0): "y-",
    (0, 0, 1): "z+", (0, 0, -1): "z-",
}
_CTX: dict = {}


def _init():
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    _CTX["ldmap"] = _ld_to_part(con)
    _CTX["classes"] = part_classes(con)


def _ax(v: np.ndarray) -> str | None:
    for a, label in AXES.items():
        if abs(v[0] - a[0]) < 0.15 and abs(v[1] - a[1]) < 0.15 and abs(v[2] - a[2]) < 0.15:
            return label
    return None


def _rot_key(R: np.ndarray) -> str | None:
    """Classify a 3x3 rotation into one of 24 cube symmetries by where its
    +X and +Y basis vectors land in world.  Returns None for off-axis rotations."""
    rx = _ax(R[:, 0])
    ry = _ax(R[:, 1])
    if rx is None or ry is None:
        return None
    return rx + ry


def _is_snot(R: np.ndarray) -> bool:
    """Y basis vector not aligned with world ±Y means studs are not up/down -> SNOT."""
    ry = _ax(R[:, 1])
    return ry not in ("y+", "y-")


def _part_rotations(f: Path, ldmap: dict) -> list[np.ndarray] | None:
    """3x3 rotation matrix per part in the same order as model_parts.
    Returns None on parse failure or empty model.  Mirrors model_parts'
    filter (part_bbox not None) and its to_model_frame() call, so indices
    stay in lock-step with the (part_num, class, lo, hi, yaw, face) list."""
    sys.path.insert(0, str(HERE.parent / "set_study"))
    try:
        from ldraw_read import flatten, part_bbox             # noqa: E402
        from patterns import to_model_frame                   # noqa: E402
    except Exception:
        return None
    try:
        parts = [p for p in flatten(f) if part_bbox(p.name) is not None]
    except Exception:
        return None
    if not parts:
        return None
    parts = to_model_frame(parts)
    rots = []
    for p in parts:
        R = p.M[:3, :3].copy()
        n = np.linalg.norm(R, axis=0)
        n[n < 1e-9] = 1.0
        rots.append(R / n)
    return rots


def pairs_in(f: Path) -> dict:
    """{(part_a, part_b, face_a, face_b, rot, dyaw_world, dy_p, dx_s_a, dz_s_a): count}
    where at least one placement is SNOT (face=='S') and the two parts are
    physically adjacent.

    `rot` is B's rotation in A's local frame, encoded as `(rx, ry)` naming
    where B's +X and +Y basis vectors point (`x+`, `x-`, `y+`, `y-`, `z+`,
    `z-`).  Dropped if either basis isn't axis-aligned (rare off-grid
    placements).  `(dx_s_a, dz_s_a)` are B-minus-A horizontal offsets with
    A's yaw backed out (stud units); `dy_p` is in plate units (y invariant
    under yaw).  `dyaw_world` is the world-frame yaw diff kept for
    backward compatibility; new readers should use `rot`."""
    try:
        items = model_parts(f, _CTX["ldmap"], _CTX["classes"])
    except Exception:
        return {}
    if not items:
        return {}
    rots = _part_rotations(f, _CTX["ldmap"])
    if rots is None or len(rots) != len(items):
        return {}                                             # re-parse mismatch; skip model
    pairs = {(i, j) if i < j else (j, i) for i, j in neighbour_pairs(items)}
    out: dict = {}
    for i, j in pairs:
        it_i = items[i]; it_j = items[j]
        face_i = it_i[5]; face_j = it_j[5]
        if face_i == "U" and face_j == "U":
            continue                                          # both studs-up: regular placement
        # Canonical order by part_num
        pn_i, pn_j = it_i[0], it_j[0]
        if pn_i <= pn_j:
            a, b, A, B, Ra, Rb = pn_i, pn_j, it_i, it_j, rots[i], rots[j]
        else:
            a, b, A, B, Ra, Rb = pn_j, pn_i, it_j, it_i, rots[j], rots[i]
        rot = _rot_key(Ra.T @ Rb)                             # B in A's frame
        if rot is None:
            continue                                          # off-axis: drop
        # Horizontal offset with A's yaw backed out
        wdx = B[2][0] - A[2][0]
        wdz = B[2][2] - A[2][2]
        dy = int(round(B[2][1] - A[2][1]))
        yaw_a = A[4] % 4
        for _ in range(yaw_a):                                # inverse of a +90 yaw
            wdx, wdz = -wdz, wdx
        dx = int(round(wdx))
        dz = int(round(wdz))
        dyaw = (B[4] - A[4]) % 4
        key = (a, b, A[5], B[5], rot, dyaw, dy, dx, dz)
        out[key] = out.get(key, 0) + 1
    return out


def main() -> None:
    files = sorted(OMR.glob("*.mpd")) + sorted(OMR.glob("*.ldr"))
    seen: Counter = Counter()
    pair_models: dict = {}
    # SNOT_ADJ_WORKERS=1 forces serial (fits ~400 MB), 0 or unset -> ncpu-1 Pool.
    nw = int(os.environ.get("SNOT_ADJ_WORKERS", "0"))
    if nw == 1:
        _init()
        for idx, f in enumerate(files):
            got = pairs_in(f)
            if not got:
                continue
            seen.update(got)
            for key in got:
                a, b = key[0], key[1]
                pair_models.setdefault((a, b), set()).add(idx)
            if idx % 100 == 0:
                print(f"  {idx+1}/{len(files)}  pairs so far: {len(seen)}", flush=True)
    else:
        with Pool(nw if nw > 1 else max(1, (os.cpu_count() or 2) - 1), initializer=_init) as pool:
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
        face_a, face_b, rot, dyaw, dy, dx, dz = key[2], key[3], key[4], key[5], key[6], key[7], key[8]
        best[(a, b)] = {"a": a, "b": b, "face_a": face_a, "face_b": face_b,
                         "rot": rot, "dyaw": int(dyaw), "dy": dy, "dx": dx, "dz": dz,
                         "models": len(pair_models[(a, b)]), "count": n}
    pairs = sorted(best.values(), key=lambda p: -p["models"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(pairs, indent=1))
    print(f"snot adjacency pairs: {len(pairs)} in {MIN_MODELS}+ models -> {OUT}")
    for p in pairs[:25]:
        print(f"  {p['a']:>8} ({p['face_a']}) + {p['b']:<8} ({p['face_b']})  rot={p['rot']}  "
              f"dy={p['dy']:+d}p dx={p['dx']:+d}s dz={p['dz']:+d}s  {p['models']} models")


if __name__ == "__main__":
    main()
