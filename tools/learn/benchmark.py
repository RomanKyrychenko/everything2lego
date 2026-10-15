"""The sampler's scoreboard: rebuild the shapes of 50 held-out official
models (never seen by any learning step) and score each build.

    python tools/learn/benchmark.py TAG [--iters 30000] [--n 50] [--no-net] [--hand]
        -> $LEGO_DATA/analysis/benchmark/TAG.json, and one summary line

Scores (means over the models; lower is better except where noted):
  volume   wrong samples per target cell
  outline  wrong outline pixels (in cell faces, three views) per cells^(2/3)
  local    mean -log P(B, where | A) per touching pair
  kl/paths KL of pair / path patterns to the reference models
  dens     the pair-density statistic (0: a real set's pairs per cell)
  loose    share of parts no stud holds
  parts    parts in our build / parts in the official model
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "tools"))
import common as C  # noqa: E402
from sampler_demo import load_models  # noqa: E402

from everything2lego import sampler  # noqa: E402

OUT = C.F.DATA / "analysis" / "benchmark"


def models_for_benchmark(rows: list[dict], n: int) -> list[dict]:
    """A fixed, kind-spread draw of held-out models of moderate size."""
    pool = sorted((r for r in rows if r["held_out"] and 300 <= r["n_parts"] <= 2500
                   and r.get("object_kind") in sampler.known_kinds() and r["object_kind"] != "scene"),
                  key=lambda r: r["model"])
    return random.Random(0).sample(pool, min(n, len(pool)))


_G: dict = {}


def _build(row):
    got = C.target(row["model"], row["object_kind"])
    if got is None:
        return None
    target, (pns, *_) = got
    w = sampler.Weights() if _G["hand"] else sampler.Weights.learned(target.kind)
    ps, info = sampler.sample(target, _G["train"], iters=_G["iters"], seed=0,
                              weights=w, net=not _G["no_net"])
    cells = float(target.want.sum())
    return {"model": row["model"], "kind": row["object_kind"], "cells": cells,
            "volume": info["fidelity"] / cells, "outline": info["outline"] / max(1.0, cells ** (2 / 3)),
            "local": info["local_per_pair"], "kl": info["kl"], "paths": info["kl_paths"],
            "dens": info["dens"], "loose": info["unsupported"] / max(1, len(ps)),
            "parts": len(ps) / max(1, len(pns))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--no-net", action="store_true")
    ap.add_argument("--hand", action="store_true", help="hand-set weights, not the learned ones")
    ap.add_argument("--quick", action="store_true", help="10 models x 10k iters: ~15s on 8 cores")
    a = ap.parse_args()
    if a.quick:
        a.n = min(a.n, 10)
        a.iters = min(a.iters, 10000)
    C.F._init()
    import pickle
    rows = pickle.load(open(C.F.OUT, "rb"))
    import pandas as pd
    k = pd.read_csv(C.F.DATA / "analysis" / "set_kinds.csv").drop_duplicates("id", keep="last")
    kind = dict(zip(k.id, k.kind))
    for r in rows:
        r["object_kind"] = kind.get(r["model"]) or kind.get(r["model"].split("_")[0])
    _G.update(train=load_models(), iters=a.iters, no_net=a.no_net, hand=a.hand)
    chosen = models_for_benchmark(rows, a.n)
    t0 = time.time()
    with mp.get_context("fork").Pool(max(1, (os.cpu_count() or 2) - 2)) as workers:
        res = [r for r in workers.imap_unordered(_build, chosen) if r is not None]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{a.tag}.json").write_text(json.dumps(res, indent=1))
    keys = ("volume", "outline", "local", "kl", "paths", "dens", "loose", "parts")
    summary = {k: float(np.mean([r[k] for r in res])) for k in keys}
    print(f"{a.tag} ({len(res)} models, {time.time() - t0:.0f} s): "
          + "  ".join(f"{k} {v:.3f}" for k, v in summary.items()))


if __name__ == "__main__":
    main()
