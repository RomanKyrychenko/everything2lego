"""Extend the GNN dataset with more degraded samples.  Reuses the existing
graphs.pkl (produced by realism_gnn.py build) and appends to the 'deg'
bucket — avoids rerunning the slow ours / ours_from_omr sampler phases.

Degradations: replace, colour, position (same as realism.py), plus:
- crop: random contiguous subgraph of 50-90% of parts
- combined: colour + position shuffle applied together

    python tools/learn/realism_add_deg.py --n 2000

Writes the updated graphs.pkl in place."""
from __future__ import annotations

import argparse
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "tools" / "set_study"))

DATA = Path(os.environ.get("LEGO_DATA", "/Volumes/PRO-G40/lego_data"))
GRAPHS_PKL = DATA / "analysis" / "realism_gnn" / "graphs.pkl"
OMR = DATA / "omr" / "files"

from realism import _degrade, _building_omr_ids
from realism_gnn import build_graph


def _write_tmp_ldr(parts, path):
    lines = ["0 degraded"]
    for p in parts:
        M = p.M
        x, y, z = M[0, 3], M[1, 3], M[2, 3]
        a, b, c = M[0, 0], M[0, 1], M[0, 2]
        d, e, f = M[1, 0], M[1, 1], M[1, 2]
        g, h, i = M[2, 0], M[2, 1], M[2, 2]
        lines.append(f"1 {p.colour} {x} {y} {z} {a} {b} {c} {d} {e} {f} {g} {h} {i} {p.name}")
    path.write_text("\n".join(lines) + "\n")


def _crop(parts, rng, keep_frac=None):
    """Drop a random fraction of parts (keeps spatial scatter / adjacency intact
    for the kept subset).  Different from the GNN augmentation crop which keeps
    a BFS-connected subgraph; this one is random-drop so adjacency breaks more."""
    if keep_frac is None:
        keep_frac = rng.uniform(0.5, 0.9)
    N = len(parts)
    k = min(N, max(10, int(N * keep_frac)))
    if k < 2:
        return parts
    keep = rng.sample(range(N), k)
    return [parts[i] for i in sorted(keep)]


def _combined(parts, rng):
    """colour shuffle + position shuffle."""
    parts = _degrade(parts, "colour", rng)
    parts = _degrade(parts, "position", rng)
    return parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000, help="target number of degraded samples to add")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import tempfile
    from ldraw_read import flatten

    with GRAPHS_PKL.open("rb") as f:
        bucket = pickle.load(f)
    print(f"existing: {dict((k, len(v)) for k, v in bucket.items())}", flush=True)

    omr_files = [f for f in sorted(OMR.glob("*.mpd"))]
    rng = random.Random(args.seed)
    rng.shuffle(omr_files)
    omr_files = omr_files[:min(len(omr_files), args.n // 2)]
    print(f"will degrade from {len(omr_files)} source OMR files", flush=True)

    # Pre-load parts (avoid repeated flatten)
    pre: list = []
    for i, f in enumerate(omr_files):
        try:
            parts = flatten(f)
            if parts:
                pre.append(parts)
        except Exception:
            continue
        if (i + 1) % 50 == 0:
            print(f"  loaded {i+1}/{len(omr_files)}", flush=True)
    print(f"loaded {len(pre)} parts lists", flush=True)

    kinds = ["replace", "colour", "position", "crop", "combined"]
    added = 0
    attempts = 0
    while added < args.n and attempts < args.n * 3:
        parts = rng.choice(pre)
        kind = kinds[attempts % len(kinds)]
        attempts += 1
        if kind == "crop":
            dp = _crop(parts, rng)
        elif kind == "combined":
            dp = _combined(parts, rng)
        else:
            dp = _degrade(parts, kind, rng)
        if not dp:
            continue
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ldr", delete=False) as fh:
            tmp = Path(fh.name)
        _write_tmp_ldr(dp, tmp)
        try:
            g = build_graph(tmp)
        except Exception:
            g = None
        finally:
            tmp.unlink(missing_ok=True)
        if g is not None:
            bucket["deg"].append(g)
            added += 1
            if added % 100 == 0:
                print(f"  added {added}/{args.n}", flush=True)

    with GRAPHS_PKL.open("wb") as f:
        pickle.dump(bucket, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"final: {dict((k, len(v)) for k, v in bucket.items())}  (+{added} deg)", flush=True)


if __name__ == "__main__":
    main()
