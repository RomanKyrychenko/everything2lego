"""Train the sampler's proposal network (everything2lego.proposals) on the
official models: every part, in fill order, is a sample — its context
(the object's cells around it, what is built before it, the kind of
object) and its token.

    python tools/learn/proposal_net.py dataset   # -> $LEGO_DATA/analysis/proposal_ds.npz
    python tools/learn/proposal_net.py train     # -> src/everything2lego/data/proposal_net.npz

The held-out models (neighbours.held_out) are the validation set; the
report compares the network with the frequency of each token (what the
sampler's prior alone knows).
"""
from __future__ import annotations

import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "tools" / "partsdb"))
sys.path.insert(0, str(REPO / "tools" / "set_study"))
sys.path.insert(0, str(REPO / "src"))
import features as F  # noqa: E402
from neighbours import held_out, model_parts  # noqa: E402

from everything2lego.proposals import features as context, n_inputs  # noqa: E402
from everything2lego.sampler import known_kinds, vocabulary  # noqa: E402
from everything2lego.style import kind_of  # noqa: E402

DS = F.DATA / "analysis" / "proposal_ds.npz"
OUT = REPO / "src/everything2lego/data/proposal_net.npz"
KINDS_CSV = F.DATA / "analysis" / "set_kinds.csv"
MAX_PER_MODEL = 400
MAX_CELLS = 3_000_000

TOKENS = sorted({v.token for v in vocabulary()})
KINDS = sorted({kind_of(t) for t in TOKENS})
OBJECTS = sorted(known_kinds())


def _object_kinds() -> dict[str, str]:
    k = pd.read_csv(KINDS_CSV).drop_duplicates("id", keep="last")
    return dict(zip(k.id, k.kind))


def model_samples(args):
    """(X, y) of one model: its parts in fill order, at most MAX_PER_MODEL."""
    f, obj = args
    try:
        items = model_parts(f, F._CTX["ldmap"], F._CTX["classes"])
    except Exception:
        return None
    if len(items) < 20:
        return None
    toks = [F._CTX["token"].get(it[0], "other0x0h0") for it in items]
    i0 = np.array([np.floor(it[2] + 0.25) for it in items], int)[:, [1, 0, 2]]       # (y, x, z)
    i1 = np.array([np.maximum(np.ceil(it[3] - 0.25), np.floor(it[2] + 0.25) + 1) for it in items], int)[:, [1, 0, 2]]
    base = i0.min(0)
    i0, i1 = i0 - base, i1 - base
    shape = tuple(i1.max(0))
    if np.prod(shape) > MAX_CELLS:
        return None
    target = np.zeros(shape, bool)
    for a, b in zip(i0, i1):
        target[a[0]:b[0], a[1]:b[1], a[2]:b[2]] = True
    placed = np.full(shape, -1, np.int64)
    kind_idx = {k: i for i, k in enumerate(KINDS)}
    tok_idx = {t: i for i, t in enumerate(TOKENS)}
    order = np.lexsort((i0[:, 2], i0[:, 1], i0[:, 0]))
    rnd = np.random.default_rng(abs(hash(f.stem)) % 2 ** 32)
    keep = set(rnd.choice(len(order), min(len(order), MAX_PER_MODEL), replace=False).tolist())
    oi = OBJECTS.index(obj) if obj in OBJECTS else -1
    X, Y = [], []
    for r, i in enumerate(order):
        a, b = i0[i], i1[i]
        if r in keep:
            X.append(context(target, placed, int(a[0]), int(a[1]), int(a[2]), oi, len(KINDS), len(OBJECTS)))
            Y.append(tok_idx.get(toks[i], len(TOKENS)))
        placed[a[0]:b[0], a[1]:b[1], a[2]:b[2]] = kind_idx.get(kind_of(toks[i]), len(KINDS))
    return f.stem, np.array(X, np.float16), np.array(Y, np.int16)


def dataset() -> None:
    objects = _object_kinds()
    files = sorted(F.OMR.glob("*.mpd")) + sorted(F.OMR.glob("*.ldr"))
    args = [(f, objects.get(f.stem) or objects.get(f.stem.split("_")[0])) for f in files]
    Xs, Ys, val = [], [], []
    with Pool(max(1, (os.cpu_count() or 2) - 1), initializer=F._init, maxtasksperchild=50) as pool:
        for k, got in enumerate(pool.imap_unordered(model_samples, args, chunksize=2)):
            if got and len(got[2]):
                Xs.append(got[1])
                Ys.append(got[2])
                val.append(np.full(len(got[2]), held_out(got[0])))
            if k % 200 == 0:
                print(f"  {k}/{len(args)}", flush=True)
    X, Y, V = np.concatenate(Xs), np.concatenate(Ys), np.concatenate(val)
    assert X.shape[1] == n_inputs(len(KINDS), len(OBJECTS))
    np.savez(DS, X=X, Y=Y, val=V)
    print(f"dataset: {len(Y)} samples ({V.sum()} held out), {X.shape[1]} inputs, "
          f"{(Y < len(TOKENS)).mean():.1%} in the vocabulary -> {DS}")


def train(hidden: int = 128, epochs: int = 6, batch: int = 512, lr: float = 1e-3, seed: int = 0) -> None:
    d = np.load(DS)
    X, Y, V = d["X"].astype(np.float32), d["Y"].astype(np.int64), d["val"]
    C = len(TOKENS) + 1
    Xt, Yt, Xv, Yv = X[~V], Y[~V], X[V], Y[V]
    mu, sd = Xt.mean(0), Xt.std(0) + 1e-3
    rnd = np.random.default_rng(seed)
    P = {"W1": rnd.normal(0, np.sqrt(2 / X.shape[1]), (X.shape[1], hidden)).astype(np.float32),
         "b1": np.zeros(hidden, np.float32),
         "W2": rnd.normal(0, np.sqrt(1 / hidden), (hidden, C)).astype(np.float32),
         "b2": np.zeros(C, np.float32)}
    m = {k: np.zeros_like(v) for k, v in P.items()}
    s = {k: np.zeros_like(v) for k, v in P.items()}
    step = 0

    def forward(x):
        h = np.maximum(0, ((x - mu) / sd) @ P["W1"] + P["b1"])
        z = h @ P["W2"] + P["b2"]
        z -= z.max(1, keepdims=True)
        lp = z - np.log(np.exp(z).sum(1, keepdims=True))
        return h, lp

    def evaluate(x, y):
        lps = np.concatenate([forward(x[i:i + 8192])[1] for i in range(0, len(x), 8192)])
        top5 = np.argsort(-lps, 1)[:, :5]
        return -lps[np.arange(len(y)), y].mean(), (top5[:, 0] == y).mean(), (top5 == y[:, None]).any(1).mean()

    freq = np.bincount(Yt, minlength=C) + 1.0
    lf = np.log(freq / freq.sum())
    print(f"frequency baseline: val nll {-lf[Yv].mean():.3f}, top-1 {(Yv == lf.argmax()).mean():.3f}")
    for ep in range(epochs):
        perm = rnd.permutation(len(Yt))
        for i in range(0, len(perm), batch):
            j = perm[i:i + batch]
            x, y = Xt[j], Yt[j]
            xn = (x - mu) / sd
            h, lp = forward(x)
            g = np.exp(lp)
            g[np.arange(len(y)), y] -= 1
            g /= len(y)
            grads = {"W2": h.T @ g, "b2": g.sum(0)}
            dh = (g @ P["W2"].T) * (h > 0)
            grads["W1"], grads["b1"] = xn.T @ dh, dh.sum(0)
            step += 1
            for k in P:                                   # Adam
                m[k] = 0.9 * m[k] + 0.1 * grads[k]
                s[k] = 0.999 * s[k] + 0.001 * grads[k] ** 2
                P[k] -= lr * (m[k] / (1 - 0.9 ** step)) / (np.sqrt(s[k] / (1 - 0.999 ** step)) + 1e-8)
        nll, t1, t5 = evaluate(Xv, Yv)
        print(f"epoch {ep + 1}: val nll {nll:.3f}, top-1 {t1:.3f}, top-5 {t5:.3f}", flush=True)
    np.savez(OUT, **P, mu=mu, sd=sd, tokens=np.array(TOKENS), kinds=np.array(KINDS), objects=np.array(OBJECTS))
    print(f"-> {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    {"dataset": dataset, "train": train}[sys.argv[1]]()
