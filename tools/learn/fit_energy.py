"""Fit the sampler's style weights to the official models, by moment
matching — the maximum-likelihood rule for an energy that is a weighted
sum of statistics: at the optimum, builds sampled under the energy have
the statistics real sets have.

Each round the sampler (with the learned proposals) builds the shapes of
official training models — every model's own shape (its parts' measured
shapes) as the target, degraded as a reconstruction from a photo would be
(common.target: surface cells flipped, part-full cells redrawn), the model
itself left out of its references — and each weight moves by

    log w_k  +=  rate * (f_k(our build) - f_k(the official model)) / scale_k

summed over the targets: raised where our builds have more of what the
term penalises than real sets have, lowered where less (a build more
"typical" than real sets is monotonous).  The outline weight is fitted
the same way: our build's outline should be as far from the degraded
target's as the real model's is — follow the form, not its noise.  The
volume weight stays: it sets the scale; colour has no data here (official
models' colours are not targets).  The official models' statistics count
only the parts the sampler can use (vocabulary tokens).

    python tools/learn/fit_energy.py [--kind building] [--rounds 8] [--targets 16] [--iters 30000]
        -> src/everything2lego/data/energy_weights.json ("weights", or "by_kind"[kind])
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import json
import math
import random
import sys
from collections import Counter
from dataclasses import asdict
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
from everything2lego.style import kl  # noqa: E402

OUT = REPO / "src/everything2lego/data/energy_weights.json"
FITTED = ("sil", "loc", "glob", "glob2", "glob3", "dens", "cnt", "sup", "side")
SCALE = {"sil": 0.05, "loc": 1.0, "glob": 0.5, "glob2": 0.5, "glob3": 0.5,
         "dens": 0.5, "cnt": 0.5, "sup": 0.02, "side": 0.02}
DEGRADE = 0.15


def outline(bits_a: np.ndarray, bits_b: np.ndarray) -> float:
    """Outline pixels (three axis views) where two shapes differ, in cell faces."""
    from everything2lego.sampler import unpack
    fa, fb = unpack(bits_a), unpack(bits_b)
    return sum(float((fa.any(axis=a) != fb.any(axis=a)).sum()) for a in (0, 1, 2)) / sampler.SUB ** 2


def statistics(info: dict, parts: int, cells: float) -> dict[str, float]:
    """The energy's statistics of a build (sampler info)."""
    return {"sil": info["outline"] / cells ** (2 / 3), "loc": info["local_per_pair"],
            "glob": info["kl"], "glob2": info["kl_paths"], "glob3": info.get("kl_paths4", 0.0),
            "dens": info["dens"], "cnt": info["cnt"], "sup": info["unsupported"] / max(1, parts),
            "side": info["bare_side_connectors"] / max(1, parts)}


def official_statistics(row: dict, stats: sampler.Stats, cells: float, vocab_tokens: set,
                        sil: float) -> dict[str, float]:
    """The same statistics of the official model itself, over its parts the
    sampler can use; its parts are all held and no side stud is bare."""
    pats = Counter({k: v for k, v in row["patterns"].items() if k[0] in vocab_tokens and k[1] in vocab_tokens})
    paths = Counter({k: v for k, v in row.get("paths", {}).items() if k[0] in vocab_tokens and k[1] in vocab_tokens})
    paths4 = Counter({k: v for k, v in row.get("paths4", {}).items() if k[0] in vocab_tokens and k[1] in vocab_tokens})
    n = sum(pats.values()) or 1
    loc = sum(c * stats.nll(*k) for k, c in pats.items()) / n
    r = math.log(sum(row["patterns"].values()) / (stats.density * cells))
    rp = math.log(row["n_parts"] / (stats.part_density * cells))
    glob3 = kl(paths4, stats.ref3) if stats.ref3 is not None and paths4 else 0.0
    return {"sil": sil / cells ** (2 / 3), "loc": loc, "glob": kl(pats, stats.ref), "glob2": kl(paths, stats.ref2),
            "glob3": glob3,
            "dens": r * r / (2 * stats.density_sd ** 2), "cnt": rp * rp / (2 * stats.part_density_sd ** 2),
            "sup": 0.0, "side": 0.0}


_G: dict = {}


def _build(job):
    """One degraded official target, built by the sampler: our statistics and the model's."""
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
    return {"model": row["model"], "kind": row["object_kind"], "ours": statistics(info, len(ps), cells),
            "official": official_statistics(row, stats, float(target.fill.sum()), _G["vocab_tokens"],
                                            outline(true_bits, target.bits))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--targets", type=int, default=16)
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--rate", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kind", default=None, help="fit for one kind of object (else all kinds pooled)")
    a = ap.parse_args()
    F._init()
    models = load_models()
    rnd = random.Random(a.seed)
    pool = [r for r in models if 300 <= r["n_parts"] <= 2500 and r.get("object_kind") in sampler.known_kinds()
            and r["object_kind"] not in ("scene",) and (a.kind is None or r["object_kind"] == a.kind)]
    vocab_tokens = {v.token for v in sampler.vocabulary()}
    # one filtered list per model, shared via fork; degree depends only on all models
    _G.update(others={m["model"]: [x for x in models if x["model"] != m["model"]] for m in pool},
              degree=sampler.token_degree(models), vocab_tokens=vocab_tokens, iters=a.iters)
    w = sampler.Weights() if a.kind is None else sampler.Weights.learned()   # a kind starts from the pooled fit
    # Seed glob3 above zero so log(w) in the moment-match update doesn't blow up;
    # the fit will then move it per the gradient.  0.3 is the hand-tuned starter.
    if w.glob3 <= 0.0:
        w.glob3 = 0.3
    log = []
    for rnd_i in range(a.rounds):
        diff = {k: 0.0 for k in FITTED}
        rows = []
        jobs = [(row, rnd_i, w) for row in rnd.sample(pool, min(a.targets, len(pool)))]
        # the round's targets are independent: one per core, forked workers share what is loaded
        with mp.get_context("fork").Pool(max(1, (os.cpu_count() or 2) - 2)) as workers:
            for got in workers.imap_unordered(_build, jobs):
                if got is None:
                    continue
                for k in FITTED:
                    diff[k] += (got["ours"][k] - got["official"][k]) / SCALE[k]
                rows.append(got)
        n = max(1, len(rows))
        for k in FITTED:
            g = max(-1.0, min(1.0, a.rate * diff[k] / n))           # at most a factor e per round
            # content and style must each carry real weight: the fit can scale them
            # relative to each other but neither drops to the floor
            lo = {"sil": 2.0, "loc": 0.3, "glob": 0.3, "glob2": 0.3, "glob3": 0.0,
                  "dens": 1.0, "cnt": 1.0, "sup": 20.0, "side": 2.0}[k]
            setattr(w, k, float(min(100.0, max(lo, getattr(w, k) * math.exp(g)))))
        mean = {k: (np.mean([r["ours"][k] for r in rows]), np.mean([r["official"][k] for r in rows])) for k in FITTED}
        print(f"round {rnd_i + 1}: " + "  ".join(f"{k} {o:.3f}/{t:.3f}->w {getattr(w, k):.3g}"
                                                  for k, (o, t) in mean.items()), flush=True)
        log.append({"round": rnd_i + 1, "weights": asdict(w), "targets": rows})
    d = json.loads(OUT.read_text()) if OUT.exists() else {}
    d["_source"] = "tools/learn/fit_energy.py (moment matching on official models)"
    fitted = {k: getattr(w, k) for k in FITTED}
    if a.kind is None:
        d["weights"] = fitted
    else:
        d.setdefault("by_kind", {})[a.kind] = fitted
    OUT.write_text(json.dumps(d, indent=1))
    (F.DATA / "analysis" / f"fit_energy_log_{a.kind or 'all'}.json").write_text(json.dumps(log, indent=1, default=float))
    print("->", OUT)


if __name__ == "__main__":
    main()
