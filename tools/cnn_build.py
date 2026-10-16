"""Greedy generator using the trained 3D CNN + transformer-decoder
(tools/learn/cnn_decoder.py).  Produces a build from a target mesh
without the Metropolis walk — the whole-target-conditioned learned
policy decides.

    python tools/cnn_build.py mesh.glb out/ --studs 32 --kind building
        [--grid 16]  [--model /path/to/cnn_decoder.pt]

Compared with tools/net_build.py (greedy over the per-cell MLP), this
one feeds the ENTIRE voxel target through a 3D CNN once at the start,
then at each cell concatenates (cell encoder feature, previous
placement embedding, kind embedding, positional embedding) and picks
the top-scored option by the transformer-decoder head.  Previous
token embedding is updated as placements are made (autoregressive
teacher-free decoding).

Honest expectation: at the model scale currently shipped (ch=128,
2-layer transformer, 5-epoch training) this will produce a buildable
model but won't beat net_build or the sampler yet; the architecture
needs more scale and training to compete.  Ship it so iteration can
happen without the Metropolis fallback blocking the comparison.
"""

from __future__ import annotations

import argparse
import subprocess
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
from everything2lego.pack import Placement  # noqa: E402
from everything2lego.io_ldraw import write_ldr  # noqa: E402
from everything2lego.sampler import select_vocabulary  # noqa: E402
from everything2lego.colors import palette  # noqa: E402
from cnn_decoder import Model, OMRDataset  # noqa: E402


def _best_colour(cell_cols: np.ndarray, allowed: set | None) -> int:
    from collections import Counter
    pal = palette()
    vals = cell_cols[cell_cols >= 0]
    c = int(Counter(vals.tolist()).most_common(1)[0][0]) if len(vals) else 0
    if allowed is None or pal[c].code in allowed:
        return c
    for i, p in enumerate(pal):
        if p.code in allowed:
            return i
    return c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh")
    ap.add_argument("out", type=Path)
    ap.add_argument("--studs", type=int, default=32)
    ap.add_argument("--shell", type=int, default=2)
    ap.add_argument("--kind", default=None)
    ap.add_argument("--model", type=Path,
                     default=Path("/Volumes/PRO-G40/lego_data/analysis/cnn_decoder.pt"))
    ap.add_argument("--n-special", type=int, default=160)
    ap.add_argument("--prefer-length", type=float, default=0.0,
                    help="bonus per extra cell per option (merges toward longer parts; 0.5-1.5 is reasonable)")
    a = ap.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(a.model, map_location="cpu", weights_only=False)
    tokens, kinds = ckpt["tokens"], ckpt["kinds"]
    grid = ckpt["grid"]
    n_tokens, n_kinds = len(tokens), len(kinds)
    tok_idx = {t: i for i, t in enumerate(tokens)}
    kind_idx = {k: i for i, k in enumerate(kinds)}

    target = target_from_mesh(a.mesh, a.studs, a.shell, a.kind)
    vocab = select_vocabulary(target.kind, a.n_special)
    opts = [(o.size[0], o.size[1], o.size[2], o.cells, v.part, v.colours, o.rot,
             tok_idx.get(v.part.ldraw, n_tokens))
            for v in vocab for o in v.orients]
    H, X, Z = target.want.shape

    device = "cpu"                                                    # Conv3D not supported on MPS in torch 2.2
    print(f"device: {device}, grid: {grid}", flush=True)

    m = Model(n_tokens=n_tokens, n_kinds=n_kinds, grid=grid).to(device)
    m.load_state_dict(ckpt["state_dict"])
    m.eval()

    # encode voxel target once
    v = torch.from_numpy((target.want > 0).astype(np.float32))[None, None]
    v = F.interpolate(v, size=(grid, grid, grid), mode="trilinear", align_corners=False)
    v = (v.squeeze(0).squeeze(0) > 0.3).float().to(device)
    with torch.no_grad():
        f = m.encoder(v.unsqueeze(0).unsqueeze(0))                     # (1, ch, G, G, G)
    f_flat = f.view(1, m.ch, -1).permute(0, 2, 1)                      # (1, G^3, ch)

    kind_i = kind_idx.get(target.kind or "", n_kinds)
    kind_t = torch.tensor([kind_i], device=device)

    # greedy cell order (bottom-up)
    ys, xs, zs = np.nonzero(target.want)
    order = np.lexsort((zs, xs, ys))
    cells = list(zip(ys[order].tolist(), xs[order].tolist(), zs[order].tolist()))
    print(f"cnn.generate @ {a.studs} studs, {len(cells)} cells, {len(opts)} options", flush=True)

    owner = np.full((H, X, Z), -1, dtype=np.int32)
    placements: list[Placement] = []
    prev_toks_seq: list[int] = []                                      # tokens placed so far
    pos_seq: list[tuple[int, int, int]] = []                           # (gy, gx, gz) in model grid
    t0 = time.time()
    for cy, cx, cz in cells:
        if owner[cy, cx, cz] >= 0:
            continue
        # map to model grid
        gy = int(cy * grid / max(1, H))
        gx = int(cx * grid / max(1, X))
        gz = int(cz * grid / max(1, Z))
        gy, gx, gz = min(gy, grid - 1), min(gx, grid - 1), min(gz, grid - 1)
        # build sequence including this new step with "empty" prior
        new_pos = pos_seq + [(gy, gx, gz)]
        new_prev = prev_toks_seq + [n_tokens + 1]                      # "empty" at the new cell
        pos_t = torch.tensor(new_pos, device=device)[None]              # (1, L, 3)
        prev_t = torch.tensor(new_prev, device=device)[None]            # (1, L)
        chunk_t = pos_t[..., 0]                                       # same-Y-layer chunking
        with torch.no_grad():
            logits = m(v.unsqueeze(0), pos_t, prev_t, kind_t, chunk_id=chunk_t)[0, -1]
            lp = logits - logits.max()
            lp = lp - torch.logsumexp(lp, dim=-1)
            lp = lp.cpu().numpy()
        # collect valid options with (score, index)
        valid = []
        for k, (h, W, D, cells_arr, part, colours, rot, tok) in enumerate(opts):
            if cy + h > H or cx + W > X or cz + D > Z:
                continue
            abs_cells = cells_arr + np.array([cy, cx, cz])
            if (owner[abs_cells[:, 0], abs_cells[:, 1], abs_cells[:, 2]] >= 0).any():
                continue
            if not target.want[abs_cells[:, 0], abs_cells[:, 1], abs_cells[:, 2]].all():
                continue
            score = float(lp[tok]) + a.prefer_length * (len(cells_arr) - 1)
            valid.append((score, k))
        if not valid:
            continue
        valid.sort(reverse=True)
        best_k = valid[0][1]
        h, W, D, cells_arr, part, colours, rot, tok = opts[best_k]
        abs_cells = cells_arr + np.array([cy, cx, cz])
        allowed_codes = set(colours) if colours else None
        colour = _best_colour(target.colour[abs_cells[:, 0], abs_cells[:, 1], abs_cells[:, 2]], allowed_codes)
        owner[abs_cells[:, 0], abs_cells[:, 1], abs_cells[:, 2]] = len(placements)
        placements.append(Placement(part, colour, cx, cz, cy, W, D, rot))
        prev_toks_seq.append(tok)
        pos_seq.append((gy, gx, gz))

    secs = time.time() - t0
    print(f"generated {len(placements)} placements, {int((owner >= 0).sum())}/{int(target.want.sum())} cells covered ({secs:.1f}s)")
    write_ldr(placements, a.out / "model.ldr")
    subprocess.run([sys.executable, str(ROOT / "tools/render_build.py"), str(a.out / "model.ldr"),
                    str(a.out / "render.png"), "--views", "30,120,210", "--size", "500"], check=False)


if __name__ == "__main__":
    main()
