"""Negatives for the proposal network: what the sampler builds where real
sets build something else.  The sampler (with the current network) builds
the shapes of official training models; its parts are replayed in fill
order into (context, token) samples, like the official ones; each token is
weighted by how much more our builds use it than real sets do on the same
shapes (1 - 1/ratio, 0 where we use it no more).  proposal_net.py train
then lowers those tokens' probability in those contexts (unlikelihood
training) as it raises the official ones'.

    python tools/learn/negatives.py [--models 120] [--iters 20000]
        -> $LEGO_DATA/analysis/proposal_neg.npz
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "tools"))
import common as C  # noqa: E402
import proposal_net as P  # noqa: E402
from sampler_demo import load_models  # noqa: E402

from everything2lego import sampler  # noqa: E402
from everything2lego.sampler import SUB, popcount  # noqa: E402

OUT = C.F.DATA / "analysis" / "proposal_neg.npz"


_G: dict = {}


def _build(job):
    """One official model's shape, built by the sampler, replayed into samples."""
    n, row = job
    got = C.target(row["model"], row["object_kind"])
    if got is None:
        return None
    target, (_, toks, *_rest) = got
    others = [m for m in _G["models"] if m["model"] != row["model"]]
    ps, _ = sampler.sample(target, others, iters=_G["iters"], seed=n)
    i0 = np.array([(p.y_plate, p.x_stud, p.z_stud) for p in ps])
    i1 = i0 + np.array([(p.part.height_units, p.w_world, p.d_world) for p in ps])
    our_toks = [_G["token_of"][p.part.ldraw] for p in ps]
    X, Y = P.replay(target.fill, our_toks, i0, i1, row["object_kind"])
    return row["model"], X, Y, our_toks, toks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=int, default=120)
    ap.add_argument("--iters", type=int, default=20000)
    a = ap.parse_args()
    C.F._init()
    models = load_models()
    pool = sorted((r for r in models if 300 <= r["n_parts"] <= 2500 and r.get("object_kind") in sampler.known_kinds()
                   and r["object_kind"] != "scene"), key=lambda r: r["model"])
    token_of = {v.part.ldraw: v.token for v in sampler.vocabulary()}
    _G.update(models=models, token_of=token_of, iters=a.iters)
    rows = list(enumerate(random.Random(1).sample(pool, min(a.models, len(pool)))))
    Xs, Ys, ours, theirs = [], [], Counter(), Counter()
    # models are independent: one per core; forked workers share what is loaded
    with mp.get_context("fork").Pool(max(1, (os.cpu_count() or 2) - 2)) as workers:
        for got in workers.imap_unordered(_build, rows):
            if got is None:
                continue
            name, X, Y, our_toks, toks = got
            Xs.append(X)
            Ys.append(Y)
            ours.update(our_toks)
            theirs.update(t for t in toks if t in token_of.values())
            print(f"  {len(Xs)}: {name} {len(our_toks)} parts", flush=True)
    no, nt = sum(ours.values()), sum(theirs.values())
    weight = np.zeros(len(P.TOKENS) + 1, np.float32)
    for i, t in enumerate(P.TOKENS):
        if ours[t]:
            ratio = (ours[t] / no) / ((theirs[t] + 1) / nt)
            weight[i] = max(0.0, 1.0 - 1.0 / ratio)
    X, Y = np.concatenate(Xs), np.concatenate(Ys)
    np.savez(OUT, X=X, Y=Y, W=weight[Y.astype(int)])
    worst = sorted(((weight[i], t) for i, t in enumerate(P.TOKENS)), reverse=True)[:10]
    print(f"negatives: {len(Y)} samples, {(weight[Y.astype(int)] > 0).mean():.0%} weighted -> {OUT}")
    print("  most over-used:", ", ".join(f"{t} {w:.2f}" for w, t in worst))


if __name__ == "__main__":
    main()
