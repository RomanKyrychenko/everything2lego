"""Generate N varied ours builds and append them to the GNN dataset's
'ours' bucket.  Each build samples random hyperparams (snot-rate,
snot-boost, tile-boost, style) and uses one of the available meshes
(or voxelised OMR silhouettes) as the target.

    python tools/learn/realism_gen_varied.py --n 500

Writes to the existing graphs.pkl in place.  Caches generated .ldr
files under $LEGO_DATA/analysis/realism_gnn/_varied_scratch so a
rerun with the same seed is idempotent."""
from __future__ import annotations

import argparse
import os
import pickle
import random
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "tools" / "set_study"))

DATA = Path(os.environ.get("LEGO_DATA", "/Volumes/PRO-G40/lego_data"))
OUT = DATA / "analysis" / "realism_gnn"
GRAPHS_PKL = OUT / "graphs.pkl"
SCRATCH = OUT / "_varied_scratch"

from realism_gnn import build_graph


def _run_one(mesh: Path, seed: int, snot_rate: float, snot_boost: float,
             tile_boost: float, style: float, iters: int) -> Path | None:
    tag = f"{mesh.stem}_s{seed}_sr{snot_rate:.2f}_sb{snot_boost:.1f}_tb{tile_boost:.1f}_st{style:.2f}"
    sess = SCRATCH / tag
    ldr = sess / "model.ldr"
    if ldr.exists():
        return ldr
    sess.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(ROOT / "tools" / "sampler_demo.py"),
           str(mesh), str(sess), "--iters", str(iters), "--seed", str(seed),
           "--snot-decorate" if snot_rate > 0 else "--no-snot-decorate",
           "--snot-rate", str(snot_rate),
           "--snot-boost", str(snot_boost),
           "--tile-boost", str(tile_boost),
           "--style", str(style)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if r.returncode != 0 or not ldr.exists():
        return None
    return ldr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--iters", type=int, default=15000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    meshes = sorted((ROOT / "examples").glob("*.glb"))
    if not meshes:
        raise SystemExit("no meshes in examples/")

    rng = random.Random(args.seed)
    added = 0
    attempts = 0
    print(f"generating up to {args.n} varied ours from {len(meshes)} meshes", flush=True)

    new_graphs = []
    while added < args.n and attempts < args.n * 2:
        mesh = rng.choice(meshes)
        seed = args.seed * 10000 + attempts
        snot_rate = rng.choice([0.0, 0.0, 0.2, 0.4, 0.6, 0.8])     # weighted toward "no SNOT" for variety
        snot_boost = rng.choice([0.0, 0.0, 2.0, 5.0])
        tile_boost = rng.choice([0.0, 0.0, 2.0, 5.0])
        style = rng.choice([0.5, 1.0, 1.0, 1.5, 2.0])
        attempts += 1
        ldr = _run_one(mesh, seed, snot_rate, snot_boost, tile_boost, style, args.iters)
        if ldr is None:
            continue
        g = build_graph(ldr)
        if g is None:
            continue
        new_graphs.append(g)
        added += 1
        if added % 20 == 0:
            print(f"  added {added}/{args.n}", flush=True)

    # Append to existing graphs.pkl
    with GRAPHS_PKL.open("rb") as f:
        bucket = pickle.load(f)
    before = len(bucket.get("ours", []))
    bucket.setdefault("ours", []).extend(new_graphs)
    after = len(bucket["ours"])
    with GRAPHS_PKL.open("wb") as f:
        pickle.dump(bucket, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"ours: {before} -> {after}  (+{added} varied)", flush=True)
    print(f"final: {dict((k, len(v)) for k, v in bucket.items())}", flush=True)


if __name__ == "__main__":
    main()
