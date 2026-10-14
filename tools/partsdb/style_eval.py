"""Validate the style measure (everything2lego.style): KL of a model's
neighbourhood-pattern distribution from that of the official models most
similar in shape.  A useful measure ranks held-out official models better
(lower) than shuffled copies of themselves and than our builds.

    python tools/partsdb/style_eval.py [our.ldr ...]
"""
from __future__ import annotations

import pickle
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
import features as F  # noqa: E402
from likeness import shuffled  # noqa: E402
from neighbours import model_parts, neighbour_pairs  # noqa: E402

from everything2lego.style import Reference, kl  # noqa: E402

K = 50


class ShapeIndex:
    def __init__(self, rows):
        self.rows = rows
        X = np.array([r["shape"] for r in rows])
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        self.Z = (X - self.mu) / self.sd

    def nearest(self, shape, k=K):
        z = (np.asarray(shape) - self.mu) / self.sd
        d = ((self.Z - z) ** 2).sum(1)
        return [self.rows[i] for i in np.argsort(d)[:k]]


def model_features(path: Path, shuffle: bool = False):
    items = model_parts(path, F._CTX["ldmap"], F._CTX["classes"])
    if shuffle:
        items = shuffled(items)
    toks = [F._CTX["token"].get(it[0], "other0x0h0") for it in items]
    pats = Counter()
    for i, j in neighbour_pairs(items):
        pats[(toks[i], toks[j], F.vertical(items[i], items[j]))] += 1
    lo = np.min([it[2] for it in items], axis=0)
    hi = np.max([it[3] for it in items], axis=0)
    vol = sum(float(np.prod(np.maximum(it[3] - it[2], 0))) for it in items)
    return {"patterns": pats, "shape": F.shape_descriptor(np.maximum(hi - lo, 1e-3), vol),
            "n_parts": len(items)}


def main() -> None:
    F._init()
    rows = pickle.load(open(F.OUT, "rb"))
    train = [r for r in rows if not r["held_out"]]
    held = [r for r in rows if r["held_out"] and r["n_parts"] >= 100]
    idx = ShapeIndex(train)
    rnd = random.Random(0)
    res = []
    for r in held:
        ref = Reference.from_models(idx.nearest(r["shape"]))
        rand = Reference.from_models(rnd.sample(train, K))
        f = F.OMR / f"{r['model']}.mpd"
        if not f.exists():
            f = F.OMR / f"{r['model']}.ldr"
        try:
            sh = model_features(f, shuffle=True)
        except Exception:
            continue
        res.append((kl(r["patterns"], ref), kl(r["patterns"], rand), kl(sh["patterns"], ref), r["n_parts"]))
    a = np.array(res)
    print(f"held-out official models ({len(a)}, >= 100 parts): KL to shape-nearest reference")
    print(f"  official           median {np.median(a[:, 0]):.3f}  (10-90%: {np.percentile(a[:, 0], 10):.3f}-{np.percentile(a[:, 0], 90):.3f})")
    print(f"  vs random reference median {np.median(a[:, 1]):.3f}   (shape-matched lower in {100 * (a[:, 0] < a[:, 1]).mean():.0f}%)")
    print(f"  shuffled copy      median {np.median(a[:, 2]):.3f}   (official lower in {100 * (a[:, 0] < a[:, 2]).mean():.0f}%)")
    big = a[a[:, 3] >= 800]
    if len(big):
        print(f"  official, >= 800 parts ({len(big)}): median {np.median(big[:, 0]):.3f}")
    for p in sys.argv[1:]:
        m = model_features(Path(p))
        ref = Reference.from_models(idx.nearest(m["shape"]))
        print(f"  ours {Path(p).parent.name} ({m['n_parts']} parts): {kl(m['patterns'], ref):.3f}")


if __name__ == "__main__":
    main()
