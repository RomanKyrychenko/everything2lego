"""Masked part model (BERT-style): guess a part's token from the parts all around it and the object's fill there.

The sampler's proposal net (everything2lego.proposals) sees only what is built before a cell in fill order, as part
kinds; this model sees every neighbouring part (token, offset, size) on all sides, as retile / re-solve need when they
refill a hole inside a build.  Trained on the official models; half the samples also empty a random hole around the
part, as those moves do.

    python tools/learn/mlm_parts.py dataset   # -> $LEGO_DATA/analysis/mlm_parts/ds.npz (both models' inputs, same parts)
    python tools/learn/mlm_parts.py train     # -> .../mlm_parts/model.pt, held-out top-k against the proposal net
"""
from __future__ import annotations

import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / "tools" / "partsdb"), str(REPO / "tools" / "set_study"), str(REPO / "src"), str(HERE)]
import common as C  # noqa: E402
import features as F  # noqa: E402
from neighbours import held_out  # noqa: E402

from everything2lego.proposals import WX, WY, WZ, features as net_features, load as load_net, static_context  # noqa: E402
from everything2lego.sampler import SUB, popcount  # noqa: E402
from everything2lego.style import kind_of  # noqa: E402

OUT = F.DATA / "analysis" / "mlm_parts"
MAX_PER_MODEL = 400
MAX_CELLS = 3_000_000
N_CTX = 64                                   # neighbours kept, nearest first
R = np.array([6, 4, 4])                      # neighbourhood half-size (plates, studs, studs) around the part's first cell
NET = load_net()
TOK = {t: i for i, t in enumerate(NET.tokens)}          # the proposal net's tokens; len(TOK) = "other"
KIND = {k: i for i, k in enumerate(NET.kinds)}


def _object_kinds() -> dict:
    import pandas as pd
    k = pd.read_csv(F.DATA / "analysis" / "set_kinds.csv").drop_duplicates("id", keep="last")
    return dict(zip(k.id, k.kind))


def model_samples(args):
    """One official model's targets: label, the proposal net's input (its fill-order context), the fill window and
    the nearest N_CTX neighbouring parts (token, offset of their first cell, size)."""
    f, obj = args
    try:
        got = C.model(f.stem)
    except Exception:                                                          # noqa: BLE001
        return None
    if got is None or len(got[1]) < 20 or got[4].size > MAX_CELLS:
        return None
    _, toks, i0, i1, bits = got
    fill = popcount(bits) / SUB ** 3
    tid = np.array([TOK.get(t, len(TOK)) for t in toks])
    rnd = np.random.default_rng(abs(hash(f.stem)) % 2 ** 32)
    keep = set(rnd.choice(len(toks), min(len(toks), MAX_PER_MODEL), replace=False).tolist())
    static = static_context(fill)
    placed = np.full(fill.shape, -1, np.int64)
    oi = NET.objects.index(obj) if obj in NET.objects else -1
    pad = np.pad(fill, [(-lo, hi) for lo, hi in (WY, WX, WZ)])             # so the window at a starts at pad[a]
    out = []
    for i in np.lexsort((i0[:, 2], i0[:, 1], i0[:, 0])):                     # fill order, as the net was trained
        a = i0[i]
        if i in keep:
            xb = net_features(fill, placed, static, int(a[0]), int(a[1]), int(a[2]), oi, len(NET.kinds), len(NET.objects))
            win = pad[a[0]:a[0] + WY[1] - WY[0], a[1]:a[1] + WX[1] - WX[0], a[2]:a[2] + WZ[1] - WZ[0]]
            off = i0 - a
            near = np.where((np.abs(off) <= R).all(1) & (np.arange(len(toks)) != i))[0]
            near = near[np.argsort(np.abs(off[near]).sum(1))][:N_CTX]
            ctx = np.full((N_CTX, 7), -1, np.int16)
            ctx[:len(near), 0] = tid[near]
            ctx[:len(near), 1:4] = off[near]
            ctx[:len(near), 4:7] = np.minimum(i1[near] - i0[near], 32)
            out.append((tid[i], xb.astype(np.float16), (win * 64).round().astype(np.uint8).ravel(), ctx))
        b = i1[i]
        placed[a[0]:b[0], a[1]:b[1], a[2]:b[2]] = KIND.get(kind_of(toks[i]), len(KIND))
    return f.stem, out


def dataset() -> None:
    objects = _object_kinds()
    files = sorted(F.OMR.glob("*.mpd")) + sorted(F.OMR.glob("*.ldr"))
    args = [(f, objects.get(f.stem) or objects.get(f.stem.split("_")[0])) for f in files]
    Y, XB, W, CTX, V, M = [], [], [], [], [], []
    t0 = time.time()
    with Pool(max(1, (os.cpu_count() or 2) - 2), initializer=F._init, maxtasksperchild=50) as pool:
        for k, got in enumerate(pool.imap_unordered(model_samples, args, chunksize=2)):
            if got and got[1]:
                name, rows = got
                for y, xb, w, c in rows:
                    Y.append(y); XB.append(xb); W.append(w); CTX.append(c)
                V += [held_out(name)] * len(rows); M += [k] * len(rows)
            if k % 200 == 0:
                print(f"  {k}/{len(args)} models, {len(Y)} samples, {time.time() - t0:.0f}s", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "ds.npz", Y=np.array(Y, np.int16), XB=np.stack(XB), W=np.stack(W), CTX=np.stack(CTX),
             val=np.array(V), model_id=np.array(M, np.int32))
    print(f"dataset: {len(Y)} samples ({sum(V)} held out) -> {OUT / 'ds.npz'}")


def _torch():
    import torch
    import torch.nn as nn

    class MLM(nn.Module):
        """[target, fill window, neighbours...] -> transformer -> the target's token (+ "other")."""

        def __init__(self, n_tok: int, d=128, layers=4, heads=4):
            super().__init__()
            self.n_tok = n_tok
            self.tok = nn.Embedding(n_tok + 2, d)                              # + other, + MASK
            self.geo = nn.Linear(6, d)
            self.fill = nn.Linear((WY[1] - WY[0]) * (WX[1] - WX[0]) * (WZ[1] - WZ[0]), d)
            self.kind = nn.Embedding(3, d)                                     # target, fill, neighbour
            self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, 2 * d, 0.1, batch_first=True, norm_first=True), layers)
            self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, n_tok + 1))

        def forward(self, win, ctx, drop):
            """win (B, 180) fill 0..1; ctx (B, N, 7) int [token, dy, dx, dz, h, w, d], token -1 = none; drop (B, N) bool:
            neighbours hidden as MASK (their place and size kept)."""
            B, N, _ = ctx.shape
            tok = ctx[..., 0].clamp(min=0)
            tok = torch.where(drop, torch.full_like(tok, self.n_tok + 1), tok)
            geo = torch.cat([ctx[..., 1:4].float() / torch.tensor(R, dtype=torch.float32, device=ctx.device), ctx[..., 4:7].float() / 8], -1)
            h_ctx = self.tok(tok) + self.geo(geo) + self.kind.weight[2]
            h_t = (self.tok.weight[self.n_tok + 1] + self.kind.weight[0]).expand(B, 1, -1)
            h_f = (self.fill(win) + self.kind.weight[1]).unsqueeze(1)
            pad = torch.cat([torch.zeros(B, 2, dtype=torch.bool, device=ctx.device), ctx[..., 0] < 0], 1)
            h = self.enc(torch.cat([h_t, h_f, h_ctx], 1), src_key_padding_mask=pad)
            return self.head(h[:, 0])
    return torch, MLM


def _hole(ctx, rng, p=0.5):
    """Empty a random box holding the target's cell in half the samples (retile / re-solve refill a hole)."""
    ctx = ctx.copy()
    B = len(ctx)
    on = rng.random(B) < p
    half = rng.integers(0, [4, 3, 3], size=(B, 3))
    shift = (rng.random((B, 3)) * 2 - 1) * half
    lo, hi = np.floor(shift - half), np.ceil(shift + half)
    inside = ((ctx[..., 1:4] >= lo[:, None]) & (ctx[..., 1:4] <= hi[:, None])).all(-1) & on[:, None]
    ctx[inside] = -1
    return ctx


def train(epochs: int = 10, batch: int = 512, lr: float = 1e-3) -> None:
    torch, MLM = _torch()
    torch.manual_seed(0)
    d = np.load(OUT / "ds.npz")
    Y, W, CTX, val = d["Y"].astype(np.int64), d["W"], d["CTX"], d["val"].astype(bool)
    tr, va = np.where(~val)[0], np.where(val)[0]
    dev = "cpu"                                           # MPS hangs in tensor copies on macOS 27
    m = MLM(len(TOK)).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr, weight_decay=0.01)
    steps = epochs * (len(tr) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps)
    rng = np.random.default_rng(0)
    print(f"{len(tr)} train, {len(va)} held out, {sum(p.numel() for p in m.parameters()) / 1e6:.2f}M parameters, {dev}", flush=True)

    def tensors(ix, hole_p):
        c = _hole(CTX[ix], rng, hole_p) if hole_p > 0 else CTX[ix]
        drop = (rng.random(c.shape[:2]) < 0.15) & (c[..., 0] >= 0) if m.training else np.zeros(c.shape[:2], bool)
        return (torch.from_numpy(W[ix].astype(np.float32) / 64).to(dev), torch.from_numpy(c.astype(np.int64)).to(dev),
                torch.from_numpy(drop).to(dev), torch.from_numpy(Y[ix]).to(dev))

    def evaluate(hole_p):
        m.eval(); ranks = []
        er = np.random.default_rng(1)
        with torch.no_grad():
            for k in range(0, len(va), 2048):
                ix = va[k:k + 2048]
                c = _hole(CTX[ix], er, hole_p) if hole_p > 0 else CTX[ix]
                z = m(torch.from_numpy(W[ix].astype(np.float32) / 64).to(dev), torch.from_numpy(c.astype(np.int64)).to(dev),
                      torch.zeros(c.shape[:2], dtype=torch.bool, device=dev)).float().cpu().numpy()
                ranks.append((z > z[np.arange(len(ix)), Y[ix]][:, None]).sum(1))
        m.train(); r = np.concatenate(ranks)
        return {k: float((r < k).mean()) for k in (1, 5, 20)}

    lp = np.concatenate([NET.logprobs(d["XB"][va[k:k + 50000]].astype(np.float32)) for k in range(0, len(va), 50000)])
    r = (lp > lp[np.arange(len(va)), Y[va]][:, None]).sum(1)
    base = {k: float((r < k).mean()) for k in (1, 5, 20)}
    print(f"proposal net (held out): top1 {base[1]:.3f} top5 {base[5]:.3f} top20 {base[20]:.3f}", flush=True)
    best, t0 = -1.0, time.time()
    for ep in range(epochs):
        perm = rng.permutation(tr)
        for s in range(len(tr) // batch):
            w, c, dr, y = tensors(perm[s * batch:(s + 1) * batch], 0.5)
            loss = torch.nn.functional.cross_entropy(m(w, c, dr), y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            if s % 200 == 0:
                print(f"  ep {ep} step {s} loss {loss.item():.3f} {time.time() - t0:.0f}s", flush=True)
        full, holed = evaluate(0.0), evaluate(1.0)
        print(f"ep {ep}: masked model top1 {full[1]:.3f} top5 {full[5]:.3f} top20 {full[20]:.3f}; "
              f"in a hole top1 {holed[1]:.3f} top5 {holed[5]:.3f} top20 {holed[20]:.3f}  {time.time() - t0:.0f}s", flush=True)
        if holed[5] > best:
            best = holed[5]
            torch.save({"state_dict": m.state_dict(), "tokens": NET.tokens, "epoch": ep, "full": full, "hole": holed, "base": base},
                       OUT / "model.pt")


if __name__ == "__main__":
    {"dataset": dataset, "train": train}[sys.argv[1]]()
