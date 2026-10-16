"""Beam search inference with the trained CNN+chunked-attention
decoder (tools/learn/cnn_decoder.py).  Keeps K candidate placement
sequences; at each cell every beam expands to top-M options, we keep
the K lowest-cumulative-NLL extensions.

Compared with cnn_build (greedy), this surfaces fuller use of the
model's distribution — the chunked model learned to spread probability
mass sharply over the top choices, but greedy locks into the single
mode; beam explores a few neighbouring possibilities.

    python tools/cnn_beam.py mesh.glb out/ --studs 32 --kind building
        [--beams 4]  [--per-cell 4]  [--prefer-length W]

--prefer-length W adds a bonus of W * (part_cell_count - 1) to each
option's log-prob score, biasing beam toward merging into longer
parts (addressing the greedy-small-parts artefact noted in commit
21460a5).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
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
from cnn_decoder import Model  # noqa: E402


@dataclass
class Beam:
    owner: np.ndarray
    placements: list
    prev_toks: list
    pos: list
    nll: float


def _best_colour(cell_cols, allowed_codes):
    from collections import Counter
    pal = palette()
    vals = cell_cols[cell_cols >= 0]
    c = int(Counter(vals.tolist()).most_common(1)[0][0]) if len(vals) else 0
    if allowed_codes is None or pal[c].code in allowed_codes:
        return c
    for i, p in enumerate(pal):
        if p.code in allowed_codes:
            return i
    return c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh")
    ap.add_argument("out", type=Path)
    ap.add_argument("--studs", type=int, default=32)
    ap.add_argument("--shell", type=int, default=2)
    ap.add_argument("--kind", default=None)
    ap.add_argument("--beams", type=int, default=4)
    ap.add_argument("--per-cell", type=int, default=4)
    ap.add_argument("--prefer-length", type=float, default=0.0,
                    help="bonus per extra cell on each option (0 disables; 0.5-1.5 merges toward longer parts)")
    ap.add_argument("--model", type=Path,
                     default=Path("/Volumes/PRO-G40/lego_data/analysis/cnn_decoder.pt"))
    ap.add_argument("--n-special", type=int, default=160)
    a = ap.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(a.model, map_location="cpu", weights_only=False)
    tokens, kinds, grid = ckpt["tokens"], ckpt["kinds"], ckpt["grid"]
    n_tokens, n_kinds = len(tokens), len(kinds)
    tok_idx = {t: i for i, t in enumerate(tokens)}
    kind_idx = {k: i for i, k in enumerate(kinds)}

    target = target_from_mesh(a.mesh, a.studs, a.shell, a.kind)
    vocab = select_vocabulary(target.kind, a.n_special)
    opts = [(o.size[0], o.size[1], o.size[2], o.cells, v.part, v.colours, o.rot,
             tok_idx.get(v.part.ldraw, n_tokens))
            for v in vocab for o in v.orients]
    H, X, Z = target.want.shape

    device = "cpu"
    m = Model(n_tokens=n_tokens, n_kinds=n_kinds, grid=grid).to(device)
    m.load_state_dict(ckpt["state_dict"])
    m.eval()
    v = torch.from_numpy((target.want > 0).astype(np.float32))[None, None]
    v = F.interpolate(v, size=(grid, grid, grid), mode="trilinear", align_corners=False)
    v = (v.squeeze(0).squeeze(0) > 0.3).float().to(device)
    kind_i = kind_idx.get(target.kind or "", n_kinds)
    kind_t = torch.tensor([kind_i], device=device)

    ys, xs, zs = np.nonzero(target.want)
    order = np.lexsort((zs, xs, ys))
    cells = list(zip(ys[order].tolist(), xs[order].tolist(), zs[order].tolist()))
    print(f"cnn.beam @ {a.studs} studs, {len(cells)} cells, beams={a.beams} per-cell={a.per_cell} "
          f"prefer-length={a.prefer_length}", flush=True)

    beams = [Beam(np.full((H, X, Z), -1, np.int32), [], [], [], 0.0)]
    t0 = time.time()
    for ci, (cy, cx, cz) in enumerate(cells):
        if ci % 500 == 0 and ci:
            print(f"  cell {ci}/{len(cells)}  beams={len(beams)}  "
                  f"best_nll={min(b.nll for b in beams):.1f}", flush=True)
        expansions = []
        for bi, beam in enumerate(beams):
            if beam.owner[cy, cx, cz] >= 0:
                expansions.append((beam.nll, bi, None, None, None))
                continue
            gy = min(int(cy * grid / max(1, H)), grid - 1)
            gx = min(int(cx * grid / max(1, X)), grid - 1)
            gz = min(int(cz * grid / max(1, Z)), grid - 1)
            new_pos = beam.pos + [(gy, gx, gz)]
            new_prev = beam.prev_toks + [n_tokens + 1]
            pos_t = torch.tensor(new_pos, device=device)[None]
            prev_t = torch.tensor(new_prev, device=device)[None]
            chunk_t = pos_t[..., 0]
            with torch.no_grad():
                logits = m(v.unsqueeze(0), pos_t, prev_t, kind_t, chunk_id=chunk_t)[0, -1]
                lp = (logits - logits.max() - torch.logsumexp(logits - logits.max(), dim=-1)).cpu().numpy()
            # score each valid option
            cand = []
            for k, opt in enumerate(opts):
                h, W, D, cells_arr, part, colours, rot, tok = opt
                if cy + h > H or cx + W > X or cz + D > Z:
                    continue
                abs_cells = cells_arr + np.array([cy, cx, cz])
                if (beam.owner[abs_cells[:, 0], abs_cells[:, 1], abs_cells[:, 2]] >= 0).any():
                    continue
                if not target.want[abs_cells[:, 0], abs_cells[:, 1], abs_cells[:, 2]].all():
                    continue
                score = float(lp[tok]) + a.prefer_length * (len(cells_arr) - 1)
                cand.append((-score, k, abs_cells))
            cand.sort()
            for neg_score, k, abs_cells in cand[:a.per_cell]:
                expansions.append((beam.nll + neg_score, bi, k, abs_cells, (gy, gx, gz)))
        if not expansions:
            break
        expansions.sort(key=lambda t: t[0])
        new_beams = []
        used = set()
        for nll, bi, k, abs_cells, gpos in expansions:
            key = (bi, k, cy, cx, cz) if k is not None else (bi, None)
            if key in used:
                continue
            used.add(key)
            src = beams[bi]
            if k is None:
                new_beams.append(src)
            else:
                owner = src.owner.copy()
                owner[abs_cells[:, 0], abs_cells[:, 1], abs_cells[:, 2]] = len(src.placements)
                h, W, D, cells_arr, part, colours, rot, tok = opts[k]
                allowed = set(colours) if colours else None
                colour = _best_colour(target.colour[abs_cells[:, 0], abs_cells[:, 1], abs_cells[:, 2]], allowed)
                plc = Placement(part, colour, cx, cz, cy, W, D, rot)
                new_beams.append(Beam(owner, src.placements + [plc],
                                       src.prev_toks + [tok], src.pos + [gpos], nll))
            if len(new_beams) >= a.beams:
                break
        beams = new_beams

    secs = time.time() - t0
    winner = min(beams, key=lambda b: b.nll)
    print(f"done: {len(winner.placements)} placements, "
          f"{int((winner.owner >= 0).sum())}/{int(target.want.sum())} cells covered, "
          f"nll {winner.nll:.1f} ({secs:.1f}s, {len(beams)} beams left)")

    write_ldr(winner.placements, a.out / "model.ldr")
    subprocess.run([sys.executable, str(ROOT / "tools/render_build.py"), str(a.out / "model.ldr"),
                    str(a.out / "render.png"), "--views", "30,120,210", "--size", "500"], check=False)


if __name__ == "__main__":
    main()
