"""Infill model: which builder option (part + rotation) goes at a cell, from the parts around it and the target's shape.

The masked part model (mlm_parts.py) guessed shape classes from a stale snapshot; this one guesses the builder's own
options, sees each neighbour's option and the target as 2x2x2 sub-cell fills, and is trained on holes that are partly
refilled, as the builder's infill move fills a cleared window part by part.  A small student (2 layers, width 64,
24 neighbours) learns the teacher's answers and is exported for the Rust forward (native/sampler).

    python tools/learn/infill_model.py dataset   # -> $LEGO_DATA/analysis/infill_model/ds.npz
    python tools/learn/infill_model.py teacher   # -> .../teacher.pt
    python tools/learn/infill_model.py student   # -> .../student.pt
    python tools/learn/infill_model.py export    # -> .../student.npz (flat weights for Rust)
"""
from __future__ import annotations

import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from mlm_parts import NET, N_CTX, R, WX, WY, WZ, _object_kinds  # noqa: E402  (also sets sys.path)
import common as C  # noqa: E402
import features as F  # noqa: E402
from neighbours import held_out  # noqa: E402

from everything2lego.sampler import vocabulary  # noqa: E402

OUT = F.DATA / "analysis" / "infill_model"
MAX_PER_MODEL = 400
MAX_CELLS = 3_000_000
N_STUDENT = 24
WIN = (WY[1] - WY[0]) * (WX[1] - WX[0]) * (WZ[1] - WZ[0])
TOK = {t: i for i, t in enumerate(NET.tokens)}
_ROT = {0: np.diag([1, 1, 1]), 90: np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),
        180: np.diag([-1, 1, -1]), 270: np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])}   # io_ldraw._ROT


def labels() -> tuple[list, dict, dict]:
    """The builder's distinct (ldraw, rotation) options, and per ldraw part its rotations and yaw offset."""
    rots, yaw = {}, {}
    for v in vocabulary():
        rots.setdefault(v.part.ldraw, set()).update(o.rot for o in v.orients)
        yaw[v.part.ldraw] = v.part.yaw
    lab = sorted((p, r) for p, rs in rots.items() for r in rs)
    return lab, rots, yaw


LAB, ROTS, YAW = labels()
LID = {k: i for i, k in enumerate(LAB)}                  # len(LAB) = other


def label_of(name: str, M: np.ndarray) -> int:
    """The option a placed part is: its ldraw and the builder's rotation (a symmetric part's equivalent one)."""
    rs = ROTS.get(name)
    if rs is None:
        return len(LAB)
    R3 = np.rint(M[:3, :3])
    r = next((k for k, m in _ROT.items() if (np.abs(R3 - m) < 0.5).all()), None)
    if r is None:                                        # sideways / tilted
        return len(LAB)
    r = (r - YAW[name]) % 360
    if r not in rs:
        r = r % 180 if (r % 180) in rs else min(rs)
    return LID.get((name, r), len(LAB))


def octants(bits: np.ndarray) -> np.ndarray:
    """Per cell the fill of its 8 octants (0..8 samples each) from the 4x4x4 sample masks (sampler.pack order)."""
    b = np.unpackbits(bits.astype("<u8").view(np.uint8).reshape(bits.shape + (8,)), axis=-1, bitorder="little")
    b = b.reshape(bits.shape + (4, 4, 4))                # bit = (sy * 4 + sx) * 4 + sz
    return b.reshape(bits.shape + (2, 2, 2, 2, 2, 2)).sum((-5, -3, -1)).reshape(bits.shape + (8,)).astype(np.uint8)


def model_samples(args):
    f, obj = args
    try:
        got = C.model(f.stem)
        parts = C.frame_parts(f.stem)
    except Exception:                                                          # noqa: BLE001
        return None
    if got is None or len(got[1]) < 20 or got[4].size > MAX_CELLS or len(parts) != len(got[1]):
        return None
    _, toks, i0, i1, bits = got
    lab = np.array([label_of(p.name, p.M) for p in parts])
    tok = np.array([TOK.get(t, len(TOK)) for t in toks])
    octs = np.pad(octants(bits), [(-lo, hi) for lo, hi in (WY, WX, WZ)] + [(0, 0)])
    rnd = np.random.default_rng(abs(hash(f.stem)) % 2 ** 32)
    out = []
    for i in rnd.choice(len(toks), min(len(toks), MAX_PER_MODEL), replace=False):
        a = i0[i]
        win = octs[a[0]:a[0] + WY[1] - WY[0], a[1]:a[1] + WX[1] - WX[0], a[2]:a[2] + WZ[1] - WZ[0]].reshape(-1)
        off = i0 - a
        near = np.where((np.abs(off) <= R).all(1) & (np.arange(len(toks)) != i))[0]
        near = near[np.argsort(np.abs(off[near]).sum(1), kind="stable")][:N_CTX]
        ctx = np.full((N_CTX, 8), -1, np.int16)
        ctx[:len(near), 0] = lab[near]
        ctx[:len(near), 1] = tok[near]
        ctx[:len(near), 2:5] = off[near]
        ctx[:len(near), 5:8] = np.minimum(i1[near] - i0[near], 32)
        out.append((lab[i], win, ctx))
    return f.stem, out


def dataset() -> None:
    objects = _object_kinds()
    files = sorted(F.OMR.glob("*.mpd")) + sorted(F.OMR.glob("*.ldr"))
    args = [(f, objects.get(f.stem) or objects.get(f.stem.split("_")[0])) for f in files]
    Y, W, CTX, V = [], [], [], []
    t0 = time.time()
    with Pool(max(1, (os.cpu_count() or 2) - 2), initializer=F._init, maxtasksperchild=50) as pool:
        for k, got in enumerate(pool.imap_unordered(model_samples, args, chunksize=2)):
            if got and got[1]:
                for y, w, c in got[1]:
                    Y.append(y); W.append(w); CTX.append(c)
                V += [held_out(got[0])] * len(got[1])
            if k % 200 == 0:
                print(f"  {k}/{len(args)} models, {len(Y)} samples, {time.time() - t0:.0f}s", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    Y = np.array(Y, np.int16)
    np.savez(OUT / "ds.npz", Y=Y, W=np.stack(W), CTX=np.stack(CTX), val=np.array(V))
    print(f"dataset: {len(Y)} samples ({sum(V)} held out), {len(LAB)} options, other {np.mean(Y == len(LAB)):.1%} -> {OUT}")


def _torch():
    import torch
    import torch.nn as nn

    class Infill(nn.Module):
        """[target, target shape, neighbours...] -> pre-norm transformer -> the target's option (+ other)."""

        def __init__(self, d=128, layers=4, heads=4):
            super().__init__()
            self.d, self.layers_n, self.heads = d, layers, heads
            self.lab = nn.Embedding(len(LAB) + 2, d)                          # + other, + MASK (the target)
            self.tok = nn.Embedding(len(TOK) + 1, d)
            self.geo = nn.Linear(6, d)
            self.shape = nn.Linear(WIN * 8, d)
            self.kind = nn.Embedding(3, d)                                     # target, shape, neighbour
            self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, 2 * d, 0.1, batch_first=True, norm_first=True), layers,
                                             enable_nested_tensor=False)
            self.norm = nn.LayerNorm(d)
            self.head = nn.Linear(d, len(LAB) + 1)

        def forward(self, win, ctx):
            """win (B, WIN*8) octant fills 0..1; ctx (B, N, 8) [option, class, dy, dx, dz, h, w, d], option -1 = none."""
            B = ctx.shape[0]
            geo = torch.cat([ctx[..., 2:5].float() / torch.tensor(R, dtype=torch.float32, device=ctx.device), ctx[..., 5:8].float() / 8], -1)
            h_c = self.lab(ctx[..., 0].clamp(min=0)) + self.tok(ctx[..., 1].clamp(min=0)) + self.geo(geo) + self.kind.weight[2]
            h_t = (self.lab.weight[len(LAB) + 1] + self.kind.weight[0]).expand(B, 1, -1)
            h_s = (self.shape(win) + self.kind.weight[1]).unsqueeze(1)
            pad = torch.cat([torch.zeros(B, 2, dtype=torch.bool, device=ctx.device), ctx[..., 0] < 0], 1)
            h = self.enc(torch.cat([h_t, h_s, h_c], 1), src_key_padding_mask=pad)
            return self.head(self.norm(h[:, 0]))
    return torch, Infill


def _hole(ctx, rng, p=0.8):
    """In most samples empty a random box around the target, keeping a random share of its parts (a part-way refill)."""
    ctx = ctx.copy()
    B = len(ctx)
    half = rng.integers(0, [4, 3, 3], size=(B, 3))
    shift = (rng.random((B, 3)) * 2 - 1) * half
    lo, hi = np.floor(shift - half), np.ceil(shift + half)
    inside = ((ctx[..., 2:5] >= lo[:, None]) & (ctx[..., 2:5] <= hi[:, None])).all(-1)
    keep = rng.random(ctx.shape[:2]) < rng.random((B, 1)) * 0.7
    ctx[inside & ~keep & (rng.random(B) < p)[:, None]] = -1
    return ctx


def _compact(ctx, n):
    """The nearest n present neighbours (rows are sorted by distance; holes leave gaps)."""
    order = np.argsort(ctx[..., 0] < 0, 1, kind="stable")[:, :n]
    return np.take_along_axis(ctx, order[..., None], 1)


def _fit(name: str, d: int, layers: int, n_ctx: int, epochs: int, teacher=None, lr=1e-3, batch=512) -> None:
    torch, Infill = _torch()
    torch.manual_seed(0)
    torch.set_num_threads(8)
    ds = np.load(OUT / "ds.npz")
    Y, W, CTX, val = ds["Y"].astype(np.int64), ds["W"], ds["CTX"], ds["val"].astype(bool)
    tr, va = np.where(~val)[0], np.where(val)[0]
    m = Infill(d, layers)
    opt = torch.optim.AdamW(m.parameters(), lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * (len(tr) // batch))
    rng = np.random.default_rng(0)
    print(f"{name}: {len(tr)} train, {len(va)} held out, {sum(p.numel() for p in m.parameters()) / 1e6:.2f}M parameters", flush=True)

    def inputs(ix, holed, r):
        c = _hole(CTX[ix], r) if holed else CTX[ix]
        return torch.from_numpy(W[ix].astype(np.float32) / 8), c

    def evaluate():
        m.eval(); res = {}
        for holed in (False, True):
            er = np.random.default_rng(1); ranks = []
            with torch.no_grad():
                for k in range(0, len(va), 2048):
                    ix = va[k:k + 2048]
                    w, c = inputs(ix, holed, er)
                    z = m(w, torch.from_numpy(_compact(c, n_ctx).astype(np.int64))).numpy()
                    ranks.append((z > z[np.arange(len(ix)), Y[ix]][:, None]).sum(1))
            r = np.concatenate(ranks)[Y[va] < len(LAB)]                        # parts the builder can place
            res[holed] = {k: float((r < k).mean()) for k in (1, 5, 20)}
        m.train()
        return res

    best, t0 = -1.0, time.time()
    for ep in range(epochs):
        perm = rng.permutation(tr)
        for s in range(len(tr) // batch):
            ix = perm[s * batch:(s + 1) * batch]
            w, c = inputs(ix, True, rng)
            z = m(w, torch.from_numpy(_compact(c, n_ctx).astype(np.int64)))
            loss = torch.nn.functional.cross_entropy(z, torch.from_numpy(Y[ix]))
            if teacher is not None:                                            # its answers on the same holed input
                with torch.no_grad():
                    zt = teacher(w, torch.from_numpy(c.astype(np.int64)))
                loss = 0.5 * loss + 0.5 * torch.nn.functional.kl_div(torch.log_softmax(z, -1), torch.log_softmax(zt, -1),
                                                                      log_target=True, reduction="batchmean")
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            if s % 200 == 0:
                print(f"  ep {ep} step {s} loss {loss.item():.3f} {time.time() - t0:.0f}s", flush=True)
        r = evaluate()
        print(f"ep {ep} (placeable parts): full top1 {r[False][1]:.3f} top5 {r[False][5]:.3f} top20 {r[False][20]:.3f}; "
              f"in a hole top1 {r[True][1]:.3f} top5 {r[True][5]:.3f} top20 {r[True][20]:.3f}  {time.time() - t0:.0f}s", flush=True)
        if r[True][5] > best:
            best = r[True][5]
            torch.save({"state_dict": m.state_dict(), "d": d, "layers": layers, "n_ctx": n_ctx, "labels": LAB, "eval": r},
                       OUT / f"{name}.pt")


def load(name: str):
    torch, Infill = _torch()
    ck = torch.load(OUT / f"{name}.pt", weights_only=False)
    m = Infill(ck["d"], ck["layers"]); m.load_state_dict(ck["state_dict"]); m.eval()
    return m, ck


def export() -> None:
    """Flat little-endian f32 weights in the order native/sampler's Infill::from reads them, plus the sizes."""
    m, ck = load("student")
    sd = {k: v.detach().numpy().astype(np.float32) for k, v in m.state_dict().items()}
    order = ["lab.weight", "tok.weight", "geo.weight", "geo.bias", "shape.weight", "shape.bias", "kind.weight"]
    for l in range(ck["layers"]):
        p = f"enc.layers.{l}."
        order += [p + k for k in ("norm1.weight", "norm1.bias", "self_attn.in_proj_weight", "self_attn.in_proj_bias",
                                  "self_attn.out_proj.weight", "self_attn.out_proj.bias", "norm2.weight", "norm2.bias",
                                  "linear1.weight", "linear1.bias", "linear2.weight", "linear2.bias")]
    order += ["norm.weight", "norm.bias", "head.weight", "head.bias"]
    assert set(order) == set(sd), set(sd) ^ set(order)
    np.savez(OUT / "student.npz", w=np.concatenate([sd[k].ravel() for k in order]),
             dims=np.array([ck["d"], ck["layers"], m.heads, ck["n_ctx"], len(LAB), len(TOK), WIN], np.int32),
             labels=np.array([f"{p}|{r}" for p, r in LAB]))
    print(f"exported {sum(sd[k].size for k in order)} weights -> {OUT / 'student.npz'}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "dataset":
        dataset()
    elif cmd == "teacher":
        _fit("teacher", 128, 4, N_CTX, 10)
    elif cmd == "student":
        _fit("student", 64, 2, N_STUDENT, 10, teacher=load("teacher")[0])
    elif cmd == "export":
        export()
