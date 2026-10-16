"""Pre-score every (option, voxel cell) pair with the trained CNN
decoder, aggregate per option, and save as a .npy bias that
sampler_demo can inject into its opts_logprior.

This chains the CNN's whole-target-conditioned predictions into the
existing Metropolis walk: Rust engine runs native-speed on the sampler
side; CNN scoring happens once at the Python side before sampling.

    python tools/cnn_prior.py mesh.glb --out /tmp/cnn_prior.npy
        [--kind building] [--model /path/to/cnn_decoder.pt]
        [--studs 32] [--n-special 160]
        [--agg mean|max] [--scale 1.0]

Then: sampler_demo mesh out --cnn-prior /tmp/cnn_prior.npy ...

The bias is added to each option's logprior before the Rust engine
sees it, so the CNN's whole-target knowledge biases the walk without
needing torch at native-engine speed.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools/learn"))

from sampler_demo import target_from_mesh  # noqa: E402
from everything2lego.sampler import select_vocabulary  # noqa: E402
from cnn_decoder import Model  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--studs", type=int, default=32)
    ap.add_argument("--shell", type=int, default=2)
    ap.add_argument("--kind", default=None)
    ap.add_argument("--model", type=Path,
                     default=Path("/Volumes/PRO-G40/lego_data/analysis/cnn_decoder.pt"))
    ap.add_argument("--n-special", type=int, default=160)
    ap.add_argument("--agg", choices=("mean", "max"), default="mean",
                    help="how to aggregate per-cell CNN scores into per-option bias")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiplier on the final bias (0 disables; 1 is already in log-prob units)")
    ap.add_argument("--cells", type=int, default=512,
                    help="sample this many target cells; full target can be huge, mean is stable from a sample")
    a = ap.parse_args()

    ckpt = torch.load(a.model, map_location="cpu", weights_only=False)
    tokens, kinds, grid = ckpt["tokens"], ckpt["kinds"], ckpt["grid"]
    n_tokens, n_kinds = len(tokens), len(kinds)
    kind_idx = {k: i for i, k in enumerate(kinds)}
    tok_idx = {t: i for i, t in enumerate(tokens)}

    target = target_from_mesh(a.mesh, a.studs, a.shell, a.kind)
    vocab = select_vocabulary(target.kind, a.n_special)
    opts = [(v, o) for v in vocab for o in v.orients]
    opt_tok = np.array([tok_idx.get(v.part.ldraw, n_tokens) for v, _ in opts])

    H, X, Z = target.want.shape
    device = "cpu"
    m = Model(n_tokens=n_tokens, n_kinds=n_kinds, grid=grid).to(device)
    m.load_state_dict(ckpt["state_dict"])
    m.eval()
    v = torch.from_numpy((target.want > 0).astype(np.float32))[None, None]
    v = F.interpolate(v, size=(grid, grid, grid), mode="trilinear", align_corners=False)
    v = (v.squeeze(0).squeeze(0) > 0.3).float().to(device)
    kind_t = torch.tensor([kind_idx.get(target.kind or "", n_kinds)], device=device)

    ys, xs, zs = np.nonzero(target.want)
    N = len(ys)
    rnd = np.random.default_rng(0)
    pick = rnd.choice(N, min(a.cells, N), replace=False) if N > 0 else np.array([], int)
    print(f"cnn_prior: scoring {len(pick)} sampled cells x {len(opts)} options...", flush=True)
    t0 = time.time()

    # score each sampled cell with the CNN
    agg = np.zeros((len(opts),), np.float64) if a.agg == "mean" else np.full((len(opts),), -np.inf)
    cnt = np.zeros((len(opts),), np.int64)
    for i, idx in enumerate(pick):
        cy, cx, cz = int(ys[idx]), int(xs[idx]), int(zs[idx])
        gy = min(int(cy * grid / max(1, H)), grid - 1)
        gx = min(int(cx * grid / max(1, X)), grid - 1)
        gz = min(int(cz * grid / max(1, Z)), grid - 1)
        pos_t = torch.tensor([[(gy, gx, gz)]], device=device)
        prev_t = torch.tensor([[n_tokens + 1]], device=device)
        chunk_t = pos_t[..., 0]
        with torch.no_grad():
            logits = m(v.unsqueeze(0), pos_t, prev_t, kind_t, chunk_id=chunk_t)[0, 0]
            lp = (logits - logits.max() - torch.logsumexp(logits - logits.max(), dim=-1)).cpu().numpy()
        scores = lp[opt_tok]                                            # (n_opts,)
        if a.agg == "mean":
            agg = agg + scores
            cnt = cnt + 1
        else:
            agg = np.maximum(agg, scores)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(pick)} ({time.time()-t0:.1f}s)", flush=True)
    bias = (agg / np.maximum(1, cnt)) if a.agg == "mean" else agg
    bias = bias * a.scale
    bias = bias - bias.max()                                            # anchor so max is 0; sampler already has logprior absolute scale
    np.save(a.out, bias.astype(np.float32))
    print(f"-> {a.out}  range {bias.min():.2f}..{bias.max():.2f}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
