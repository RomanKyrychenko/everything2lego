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
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import features as F  # noqa: E402
from neighbours import held_out  # noqa: E402

from everything2lego.proposals import features as context, n_inputs, static_context  # noqa: E402
from everything2lego.sampler import SUB, popcount  # noqa: E402
from everything2lego.sampler import known_kinds, vocabulary  # noqa: E402
from everything2lego.style import kind_of  # noqa: E402

DS = F.DATA / "analysis" / "proposal_ds.npz"
NEG = F.DATA / "analysis" / "proposal_neg.npz"
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


def replay(fill: np.ndarray, toks: list, i0: np.ndarray, i1: np.ndarray, obj, keep=None):
    """(X, y) of a build, its parts replayed in fill order: each part's
    context when it was placed, and its token.  `keep`: the ranks to take."""
    static = static_context(fill)
    placed = np.full(fill.shape, -1, np.int64)
    kind_idx = {k: i for i, k in enumerate(KINDS)}
    tok_idx = {t: i for i, t in enumerate(TOKENS)}
    order = np.lexsort((i0[:, 2], i0[:, 1], i0[:, 0]))
    oi = OBJECTS.index(obj) if obj in OBJECTS else -1
    X, Y = [], []
    for r, i in enumerate(order):
        a, b = i0[i], i1[i]
        if keep is None or r in keep:
            X.append(context(fill, placed, static, int(a[0]), int(a[1]), int(a[2]), oi, len(KINDS), len(OBJECTS)))
            Y.append(tok_idx.get(toks[i], len(TOKENS)))
        placed[a[0]:b[0], a[1]:b[1], a[2]:b[2]] = kind_idx.get(kind_of(toks[i]), len(KINDS))
    return np.array(X, np.float16), np.array(Y, np.int16)


def model_samples(args):
    """(name, X, y) of one official model: its parts in fill order, at most
    MAX_PER_MODEL, against its real shape (common.model)."""
    f, obj = args
    try:
        got = C.model(f.stem)
    except Exception:
        return None
    if got is None or len(got[0]) < 20 or got[4].size > MAX_CELLS:
        return None
    _, toks, i0, i1, bits = got
    rnd = np.random.default_rng(abs(hash(f.stem)) % 2 ** 32)
    keep = set(rnd.choice(len(toks), min(len(toks), MAX_PER_MODEL), replace=False).tolist())
    X, Y = replay(popcount(bits) / SUB ** 3, toks, i0, i1, obj, keep)
    return f.stem, X, Y


def dataset() -> None:
    objects = _object_kinds()
    files = sorted(F.OMR.glob("*.mpd")) + sorted(F.OMR.glob("*.ldr"))
    args = [(f, objects.get(f.stem) or objects.get(f.stem.split("_")[0])) for f in files]
    Xs, Ys, val, mids = [], [], [], []
    with Pool(max(1, (os.cpu_count() or 2) - 1), initializer=F._init, maxtasksperchild=50) as pool:
        for k, got in enumerate(pool.imap_unordered(model_samples, args, chunksize=2)):
            if got and len(got[2]):
                Xs.append(got[1])
                Ys.append(got[2])
                val.append(np.full(len(got[2]), held_out(got[0])))
                mids.append(np.full(len(got[2]), k, np.int32))        # per-sample model id for sequence/composition losses
            if k % 200 == 0:
                print(f"  {k}/{len(args)}", flush=True)
    X, Y, V = np.concatenate(Xs), np.concatenate(Ys), np.concatenate(val)
    M = np.concatenate(mids)
    assert X.shape[1] == n_inputs(len(KINDS), len(OBJECTS))
    np.savez(DS, X=X, Y=Y, val=V, model_id=M)
    print(f"dataset: {len(Y)} samples ({V.sum()} held out), {X.shape[1]} inputs, "
          f"{(Y < len(TOKENS)).mean():.1%} in the vocabulary -> {DS}")


def train(hidden: int = 128, epochs: int = 6, batch: int = 512, lr: float = 1e-3, seed: int = 0,
          lam: float = 0.5, resume: bool = False, freq_reweight: bool = False,
          composition: float = 0.0, by_model: bool = False) -> None:
    """Maximum likelihood on the official placements, plus — when
    negatives.py has run — unlikelihood on the sampler's over-used choices.
    `resume=True`: start from the previously saved network.
    `freq_reweight=True`: scale each sample's loss by 1/sqrt(token freq).
    `composition=w > 0`: auxiliary KL between the net's mean predicted
        distribution over a batch and the batch's empirical label
        distribution — a whole-batch signal that pulls the net toward
        the right overall composition, not just per-cell correctness.
    `by_model=True`: batch samples by model (one model's placements per
        step) so the composition loss sees a real set's composition
        rather than a random cell mix.  Needs model_id in the dataset."""
    neg = np.load(NEG) if NEG.exists() else None
    d = np.load(DS)
    X, Y, V = d["X"].astype(np.float32), d["Y"].astype(np.int64), d["val"]
    M = d["model_id"].astype(np.int64) if "model_id" in d.files else np.zeros(len(Y), np.int64)
    C = len(TOKENS) + 1
    Xt, Yt, Xv, Yv = X[~V], Y[~V], X[V], Y[V]
    Mt = M[~V]
    mu, sd = Xt.mean(0), Xt.std(0) + 1e-3
    rnd = np.random.default_rng(seed)
    if resume and OUT.exists():
        prev = np.load(OUT)
        P = {"W1": prev["W1"].copy(), "b1": prev["b1"].copy(),
             "W2": prev["W2"].copy(), "b2": prev["b2"].copy()}
        mu, sd = prev["mu"], prev["sd"]              # keep normalisation stable
        print(f"resuming from {OUT.name}: W1 {P['W1'].shape}, W2 {P['W2'].shape}", flush=True)
    else:
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
    if freq_reweight:
        token_w = 1.0 / np.sqrt(freq)
        token_w /= token_w.mean()                                    # normalise so overall scale matches
        print(f"freq-reweight on: weight range {token_w.min():.2f}..{token_w.max():.2f}")
    else:
        token_w = np.ones(C, dtype=np.float32)
    def _model_batches():
        """Yield (indices,) batches grouped by model_id — each batch is a
        single model's placements (so composition loss sees a real set)."""
        order = rnd.permutation(np.unique(Mt))
        buf: list[int] = []
        for m in order:
            idxs = np.flatnonzero(Mt == m)
            if len(idxs) == 0:
                continue
            if len(buf) + len(idxs) > batch and buf:
                yield np.array(buf, np.int64); buf = []
            buf.extend(idxs.tolist())
            if len(buf) >= batch:
                yield np.array(buf[:batch], np.int64); buf = buf[batch:]
        if buf:
            yield np.array(buf, np.int64)

    for ep in range(epochs):
        if by_model:
            batches = list(_model_batches())
        else:
            perm = rnd.permutation(len(Yt))
            batches = [perm[i:i + batch] for i in range(0, len(perm), batch)]
        for j in batches:
            x, y = Xt[j], Yt[j]
            w = token_w[y].astype(np.float32)
            xn = (x - mu) / sd
            h, lp = forward(x)
            g = np.exp(lp)
            g[np.arange(len(y)), y] -= 1
            g *= w[:, None]
            g /= max(1e-6, w.sum())
            # composition loss: pull the batch-mean predicted distribution toward
            # the empirical label distribution.  Grad on logits = w * (q_mean - q_true)
            # where q_mean = exp(lp).mean(0), q_true = empirical_one_hot(y).mean(0).
            if composition > 0:
                p_mean = np.exp(lp).mean(0)
                q_emp = np.bincount(y, minlength=C) / max(1, len(y))
                comp_grad = (p_mean - q_emp)                         # (C,)
                # distribute over samples: each sample contributes 1/N to p_mean
                g_comp = np.broadcast_to(comp_grad[None, :], lp.shape).copy()
                g_comp = g_comp * np.exp(lp) / max(1, len(y))         # chain through softmax
                g = g + composition * g_comp
            if neg is not None:                            # and push the over-used choices down
                jn = rnd.integers(0, len(neg["Y"]), batch)
                xq, yq, wq = neg["X"][jn].astype(np.float32), neg["Y"][jn].astype(np.int64), neg["W"][jn]
                hq, lq = forward(xq)
                pq = np.exp(lq)
                py = pq[np.arange(len(yq)), yq]
                gq = -pq * (wq * py / np.maximum(1e-6, 1 - py))[:, None]
                gq[np.arange(len(yq)), yq] += wq * py / np.maximum(1e-6, 1 - py)
                gq *= lam / len(yq)
                h, g = np.concatenate([h, hq]), np.concatenate([g, gq])
                xn = np.concatenate([xn, (xq - mu) / sd])
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
    def _f(flag, default=0.0):
        """Parse --flag=VALUE from argv, else default."""
        for a in sys.argv:
            if a.startswith(f"--{flag}="):
                return float(a.split("=", 1)[1])
        return default
    {"dataset": dataset,
     "train": lambda: train(resume="--resume" in sys.argv,
                             freq_reweight="--freq-reweight" in sys.argv,
                             composition=_f("composition", 0.0),
                             by_model="--by-model" in sys.argv)}[sys.argv[1]]()
