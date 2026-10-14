"""Fit the sampler's style weights to the official models, by moment
matching — the maximum-likelihood rule for an energy that is a weighted
sum of statistics: at the optimum, builds sampled under the energy have
the statistics real sets have.

Each round the sampler (with the learned proposals) builds the shapes of
official training models — every model's own shape as the target, the
model itself left out of its references — and each style weight moves by

    log w_k  +=  rate * (f_k(our build) - f_k(the official model)) / scale_k

summed over the targets: raised where our builds have more of what the
term penalises than real sets have, lowered where less (a build more
"typical" than real sets is monotonous).  The form weights (volume,
outline, colour) stay: they set the scale.  The official models'
statistics count only the parts the sampler can use (vocabulary tokens).

    python tools/learn/fit_energy.py [--rounds 8] [--targets 16] [--iters 30000]
        -> src/everything2lego/data/energy_weights.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "tools" / "partsdb"))
sys.path.insert(0, str(REPO / "tools" / "set_study"))
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "src"))
import features as F  # noqa: E402
from neighbours import model_parts  # noqa: E402
from sampler_demo import load_models  # noqa: E402

from everything2lego import sampler  # noqa: E402
from everything2lego.style import kl  # noqa: E402

OUT = REPO / "src/everything2lego/data/energy_weights.json"
FITTED = ("loc", "glob", "glob2", "dens", "sup", "side")
SCALE = {"loc": 1.0, "glob": 0.5, "glob2": 0.5, "dens": 0.5, "sup": 0.02, "side": 0.02}


def official_target(name: str, kind: str | None) -> sampler.Target | None:
    """The official model's own shape: its parts' grid boxes."""
    f = F.OMR / f"{name}.mpd"
    if not f.exists():
        f = F.OMR / f"{name}.ldr"
    items = model_parts(f, F._CTX["ldmap"], F._CTX["classes"])
    if not items:
        return None
    i0 = np.array([np.floor(it[2] + 0.25) for it in items], int)[:, [1, 0, 2]]
    i1 = np.array([np.maximum(np.ceil(it[3] - 0.25), np.floor(it[2] + 0.25) + 1) for it in items], int)[:, [1, 0, 2]]
    base = i0.min(0)
    i0, i1 = i0 - base, i1 - base
    want = np.zeros(tuple(i1.max(0)), bool)
    for a, b in zip(i0, i1):
        want[a[0]:b[0], a[1]:b[1], a[2]:b[2]] = True
    visible = want & ~binary_erosion(want)
    return sampler.Target(want, np.full(want.shape, -1), visible, kind=kind)


def statistics(info: dict, parts: int) -> dict[str, float]:
    """The energy's style statistics of a build (sampler info)."""
    return {"loc": info["local_per_pair"], "glob": info["kl"], "glob2": info["kl_paths"],
            "dens": info["dens"], "sup": info["unsupported"] / max(1, parts),
            "side": info["bare_side_connectors"] / max(1, parts)}


def official_statistics(row: dict, stats: sampler.Stats, cells: float, vocab_tokens: set) -> dict[str, float]:
    """The same statistics of the official model itself, over its parts the
    sampler can use; its parts are all held and no side stud is bare."""
    pats = Counter({k: v for k, v in row["patterns"].items() if k[0] in vocab_tokens and k[1] in vocab_tokens})
    paths = Counter({k: v for k, v in row.get("paths", {}).items() if k[0] in vocab_tokens and k[1] in vocab_tokens})
    n = sum(pats.values()) or 1
    loc = sum(c * stats.nll(*k) for k, c in pats.items()) / n
    r = math.log(sum(row["patterns"].values()) / (stats.density * cells))
    return {"loc": loc, "glob": kl(pats, stats.ref), "glob2": kl(paths, stats.ref2),
            "dens": r * r / (2 * stats.density_sd ** 2), "sup": 0.0, "side": 0.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--targets", type=int, default=16)
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--rate", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    F._init()
    models = load_models()
    rnd = random.Random(a.seed)
    pool = [r for r in models if 300 <= r["n_parts"] <= 2500 and r.get("object_kind") in sampler.known_kinds()
            and r["object_kind"] not in ("scene",)]
    vocab_tokens = {v.token for v in sampler.vocabulary()}
    w = sampler.Weights()
    log = []
    for rnd_i in range(a.rounds):
        diff = {k: 0.0 for k in FITTED}
        rows = []
        for row in rnd.sample(pool, a.targets):
            target = official_target(row["model"], row["object_kind"])
            if target is None or target.want.sum() > 400_000:
                continue
            others = [m for m in models if m["model"] != row["model"]]
            refs = sampler.nearest_models(others, target.shape_descriptor(), 50, target.kind)
            stats = sampler.Stats(refs)
            ps, info = sampler.sample(target, others, iters=a.iters, seed=rnd_i, weights=w)
            ours = statistics(info, len(ps))
            theirs = official_statistics(row, stats, float(target.fill.sum()), vocab_tokens)
            for k in FITTED:
                diff[k] += (ours[k] - theirs[k]) / SCALE[k]
            rows.append({"model": row["model"], "kind": row["object_kind"], "ours": ours, "official": theirs})
        n = max(1, len(rows))
        for k in FITTED:
            g = max(-1.0, min(1.0, a.rate * diff[k] / n))           # at most a factor e per round
            setattr(w, k, float(min(100.0, max(0.01, getattr(w, k) * math.exp(g)))))
        mean = {k: (np.mean([r["ours"][k] for r in rows]), np.mean([r["official"][k] for r in rows])) for k in FITTED}
        print(f"round {rnd_i + 1}: " + "  ".join(f"{k} {o:.3f}/{t:.3f}->w {getattr(w, k):.3g}"
                                                  for k, (o, t) in mean.items()), flush=True)
        log.append({"round": rnd_i + 1, "weights": asdict(w), "targets": rows})
    OUT.write_text(json.dumps({"_source": "tools/learn/fit_energy.py (moment matching on official models)",
                               "weights": {k: getattr(w, k) for k in FITTED}}, indent=1))
    (F.DATA / "analysis" / "fit_energy_log.json").write_text(json.dumps(log, indent=1, default=float))
    print("->", OUT)


if __name__ == "__main__":
    main()
