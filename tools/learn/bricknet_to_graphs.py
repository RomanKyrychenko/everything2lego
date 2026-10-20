"""Convert BrickNet .npz graphs (via bricknet.load_graphs + graph_to_ldr)
into our GNN's 5-tuple format and append to graphs.pkl's 'real' class.

    python tools/learn/bricknet_to_graphs.py --split val
    python tools/learn/bricknet_to_graphs.py --split sft --max 10000
    python tools/learn/bricknet_to_graphs.py --split pt --max 50000

Writes to the existing graphs.pkl in place.  Uses tempfile → build_graph
→ delete, so parsing overhead dominates (no sampler calls)."""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "tools" / "set_study"))

DATA = Path(os.environ.get("LEGO_DATA", "/Volumes/PRO-G40/lego_data"))
BRICKNET_DIR = DATA / "bricknet"
GRAPHS_PKL = DATA / "analysis" / "realism_gnn" / "graphs.pkl"

from realism_gnn import build_graph, _part_id_map


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "sft", "pt"], required=True)
    ap.add_argument("--max", type=int, default=0, help="cap graphs (0 = all)")
    args = ap.parse_args()

    import bricknet
    npz = BRICKNET_DIR / f"{args.split}.npz"
    if not npz.exists():
        xz = BRICKNET_DIR / f"{args.split}.npz.xz"
        if xz.exists():
            print(f"decompressing {xz}...", flush=True)
            import subprocess
            subprocess.run(["unxz", "-k", str(xz)], check=True)
    print(f"loading {npz}...", flush=True)
    graphs = bricknet.load_graphs(str(npz))
    total = len(graphs)
    cap = args.max if args.max > 0 else total
    print(f"loaded {total} graphs; converting {cap}", flush=True)

    part_map = _part_id_map()
    converted = []
    skipped = 0
    for i in range(cap):
        g_bn = graphs[i]
        try:
            text = bricknet.graph_to_ldr(g_bn)
        except Exception:
            skipped += 1
            continue
        if not text or "1 " not in text:
            skipped += 1
            continue
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ldr", delete=False,
                                           dir=os.environ.get("TMPDIR")) as fh:
            fh.write(text)
            tmp = Path(fh.name)
        try:
            g_ours = build_graph(tmp, part_id_map=part_map)
        except Exception:
            g_ours = None
        finally:
            tmp.unlink(missing_ok=True)
        if g_ours is None:
            skipped += 1
            continue
        converted.append(g_ours)
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{cap} converted={len(converted)} skipped={skipped}", flush=True)

    print(f"done: converted={len(converted)} skipped={skipped}", flush=True)

    with GRAPHS_PKL.open("rb") as f:
        bucket = pickle.load(f)
    before = len(bucket.get("real", []))
    bucket.setdefault("real", []).extend(converted)
    after = len(bucket["real"])
    with GRAPHS_PKL.open("wb") as f:
        pickle.dump(bucket, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"real: {before} -> {after}  (+{len(converted)} from bricknet {args.split})", flush=True)
    print(f"final: {dict((k, len(v)) for k, v in bucket.items())}", flush=True)


if __name__ == "__main__":
    main()
