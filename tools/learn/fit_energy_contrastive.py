"""Fit the sampler's energy weights by contrastive learning instead of
moment matching: train a logistic classifier on (official, sampled) builds
and interpret the discriminator weights as the energy weights that best
separate real sets from sampler builds.

    python tools/learn/fit_energy_contrastive.py [--targets 32] [--iters 30000]
        -> src/everything2lego/data/energy_weights_contrastive.json

Writes to a SEPARATE file so the existing moment-matched weights aren't
overwritten.  To use the learned weights, replace or override `weights`
in `energy_weights.json`.

Compared to fit_energy.py's moment matching:
- MM asks "match the mean statistics of real sets"; each weight moves by
  the mean deviation between our builds and official models.
- Contrastive asks "which linear combination best discriminates real from
  our builds"; weights are the log-odds gradient of P(real | features).

Both are MLE-of-energy approximations when `energy = w . features`.  MM
converges when sampled and real feature means match; contrastive converges
to the Bayes-optimal linear separator -- which may put more weight on
features that discriminate most, not just those with largest mean gap.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "tools" / "partsdb"))
sys.path.insert(0, str(REPO / "tools" / "set_study"))
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))

import common as C  # noqa: E402
import features as F  # noqa: E402
from sampler_demo import load_models  # noqa: E402
from everything2lego import sampler  # noqa: E402
from fit_energy import FITTED, SCALE, DEGRADE, statistics, official_statistics, outline  # noqa: E402

OUT = REPO / "src/everything2lego/data/energy_weights_contrastive.json"

_G: dict = {}


def _collect(job):
    """One build job: return (ours, official) feature dicts."""
    row, rnd_i, w = job
    got = C.target(row["model"], row["object_kind"], degrade=DEGRADE, seed=rnd_i)
    if got is None or got[0].want.sum() > 400_000:
        return None
    target, (_, _, _, _, true_bits) = got
    others = _G["others"][row["model"]]
    refs = sampler.nearest_models(others, target.shape_descriptor(), 50, target.kind)
    stats = sampler.Stats(refs)
    stats.degree = _G["degree"]
    ps, info = sampler.sample(target, iters=_G["iters"], seed=rnd_i, weights=w, stats=stats)
    cells = float(target.want.sum())
    return {"model": row["model"], "kind": row["object_kind"],
            "ours": statistics(info, len(ps), cells),
            "official": official_statistics(row, stats, float(target.fill.sum()), _G["vocab_tokens"],
                                             outline(true_bits, target.bits))}


def fit_logistic(X: np.ndarray, y: np.ndarray, epochs: int = 500, lr: float = 0.1,
                   l2: float = 1e-3) -> tuple[np.ndarray, float]:
    """Fit logistic regression P(y=1 | x) = sigmoid(w.x + b).  Returns (w, b)."""
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(epochs):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))
        grad_w = X.T @ (p - y) / n + l2 * w
        grad_b = (p - y).mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=4, help="rounds of sampling to collect features")
    ap.add_argument("--targets", type=int, default=32, help="targets per round")
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kind", default=None)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--lr", type=float, default=0.1)
    a = ap.parse_args()
    F._init()
    models = load_models()
    rnd = random.Random(a.seed)
    pool = [r for r in models if 300 <= r["n_parts"] <= 2500 and r.get("object_kind") in sampler.known_kinds()
            and r["object_kind"] not in ("scene",) and (a.kind is None or r["object_kind"] == a.kind)]
    vocab_tokens = {v.token for v in sampler.vocabulary()}
    _G.update(others={m["model"]: [x for x in models if x["model"] != m["model"]] for m in pool},
              degree=sampler.token_degree(models), vocab_tokens=vocab_tokens, iters=a.iters)
    # Collect features from several rounds starting from the current best weights
    w0 = sampler.Weights() if a.kind is None else sampler.Weights.learned(a.kind)
    rows = []
    for rnd_i in range(a.rounds):
        jobs = [(row, rnd_i * 100 + i, w0) for i, row in enumerate(rnd.sample(pool, min(a.targets, len(pool))))]
        with mp.get_context("fork").Pool(max(1, (os.cpu_count() or 2) - 2)) as pool_:
            for got in pool_.imap_unordered(_collect, jobs):
                if got is None:
                    continue
                rows.append(got)
        print(f"collected {len(rows)} pairs after round {rnd_i + 1}", flush=True)
    # Build feature matrix: positive = official, negative = ours
    X = []
    y = []
    for r in rows:
        X.append([r["official"][k] for k in FITTED])
        y.append(1)
        X.append([r["ours"][k] for k in FITTED])
        y.append(0)
    X = np.array(X, np.float64)
    y = np.array(y, np.float64)
    # Normalize per-feature then fit
    mu = X.mean(0)
    sigma = X.std(0) + 1e-6
    Xn = (X - mu) / sigma
    w_logistic, b = fit_logistic(Xn, y, epochs=a.epochs, lr=a.lr)
    # Energy weight interpretation: features high in "ours" (fake, y=0) should get HIGHER energy.
    # Logistic learned sigmoid(w.x+b)=P(real).  w_k > 0 means feature k is higher in real.
    # So energy weight should be -w_logistic (penalize features more common in fake).
    w_scale = w_logistic / sigma                                        # un-normalize
    w_energy = -w_scale                                                 # flip sign
    # Clamp: keep positive and in a reasonable range (same as fit_energy's lo bounds)
    lo = {"sil": 2.0, "loc": 0.3, "glob": 0.3, "glob2": 0.3, "glob3": 0.0,
          "dens": 1.0, "cnt": 1.0, "sup": 20.0, "side": 2.0}
    out_weights = {k: float(max(lo[k], min(100.0, w_energy[i]))) if w_energy[i] > 0 else lo[k]
                     for i, k in enumerate(FITTED)}
    # Compare to current (moment-matched)
    print("\n  key        contrastive raw     clamped     moment-matched")
    cur = sampler.Weights() if a.kind is None else sampler.Weights.learned(a.kind)
    for i, k in enumerate(FITTED):
        print(f"  {k:5s}      {w_energy[i]:+.3f}             {out_weights[k]:.3f}        {getattr(cur, k):.3f}")
    # Save
    d = json.loads(OUT.read_text()) if OUT.exists() else {}
    d["_source"] = "tools/learn/fit_energy_contrastive.py (logistic discriminator on real vs sampler builds)"
    d["_n_pairs"] = len(rows)
    if a.kind is None:
        d["weights"] = out_weights
    else:
        d.setdefault("by_kind", {})[a.kind] = out_weights
    OUT.write_text(json.dumps(d, indent=1))
    print(f"\n-> {OUT}  (n_pairs={len(rows)})")


if __name__ == "__main__":
    main()
