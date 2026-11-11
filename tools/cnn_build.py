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
from typing import Optional

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


def cnn_greedy(target, model_path, n_special: int = 160, prefer_length: float = 0.0,
                refine_rounds: int = 0, mask_pct: float = 0.1, seed: int = 0,
                support_penalty: float = 0.0,
                guidance_model: Optional[Path] = None, guidance: float = 1.0,
                verbose: bool = True) -> list[Placement]:
    """Run the CNN decoder greedily over the voxel target and return the list
    of Placements.  Reusable from sampler_demo (--init-from-cnn) or other
    composition pipelines.

    `refine_rounds > 0` enables iterative refinement: after the initial greedy
    pass, each round removes a random `mask_pct` fraction of placements and
    re-runs the greedy decoder on the freed cells only.  The model now has
    the full build as context (not just a growing prefix), so earlier choices
    get a second chance.  Changes are kept only if they don't increase total
    part count."""
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    tokens, kinds = ckpt["tokens"], ckpt["kinds"]
    grid = ckpt["grid"]
    n_tokens, n_kinds = len(tokens), len(kinds)
    tok_idx = {t: i for i, t in enumerate(tokens)}
    kind_idx = {k: i for i, k in enumerate(kinds)}
    factored = bool(ckpt.get("factored"))
    unet = bool(ckpt.get("unet"))
    ch = int(ckpt.get("ch", 32))

    vocab = select_vocabulary(target.kind, n_special)
    if factored:
        cats = ckpt["cats"]
        sizes = [tuple(s) for s in ckpt["sizes"]]
        cat_idx = {c: i for i, c in enumerate(cats)}
        size_idx = {tuple(s): i for i, s in enumerate(sizes)}
        n_cats, n_sizes = len(cats), len(sizes)
        opts = []
        for v in vocab:
            cid = cat_idx.get(v.category, n_cats)
            for o in v.orients:
                sid = size_idx.get(tuple(o.size), n_sizes)
                yid = o.rot & 3
                opts.append((o.size[0], o.size[1], o.size[2], o.cells, v.part, v.colours, o.rot,
                              cid, sid, yid))
    else:
        n_cats = n_sizes = 0
        opts = [(o.size[0], o.size[1], o.size[2], o.cells, v.part, v.colours, o.rot,
                 tok_idx.get(v.part.ldraw, n_tokens))
                for v in vocab for o in v.orients]
    H, X, Z = target.want.shape

    device = "cpu"                                                    # Conv3D not supported on MPS in torch 2.2
    if verbose:
        print(f"device: {device}, grid: {grid}, factored: {factored}, unet: {unet}", flush=True)

    mkw = {"n_cats": n_cats, "n_sizes": n_sizes} if factored else {}
    m = Model(n_tokens=n_tokens, n_kinds=n_kinds, grid=grid, ch=ch, factored=factored, unet=unet, **mkw).to(device)
    m.load_state_dict(ckpt["state_dict"])
    m.eval()

    # Classifier-free guidance (optional): load a second "unconditional" model.
    # Blend at inference: logits_cfg = logits_uncond + guidance * (logits_cond - logits_uncond)
    # where `m` is the conditional (polished/distilled) and `m_u` is the OMR-only unconditional.
    m_u: Optional[Model] = None
    if guidance_model is not None:
        if factored:
            raise ValueError("CFG not supported for factored checkpoints")
        ck_u = torch.load(guidance_model, map_location="cpu", weights_only=False)
        if ck_u["tokens"] != tokens or ck_u["kinds"] != kinds or ck_u["grid"] != grid:
            raise ValueError("CFG models must share tokens/kinds/grid")
        ch_u = int(ck_u.get("ch", 32)); unet_u = bool(ck_u.get("unet"))
        m_u = Model(n_tokens=n_tokens, n_kinds=n_kinds, grid=grid, ch=ch_u, unet=unet_u).to(device)
        m_u.load_state_dict(ck_u["state_dict"])
        m_u.eval()
        if verbose:
            print(f"cfg: guidance={guidance} unconditional={guidance_model.name}", flush=True)

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
    initial_cells = list(zip(ys[order].tolist(), xs[order].tolist(), zs[order].tolist()))
    if verbose:
        print(f"cnn.generate: {len(initial_cells)} cells, {len(opts)} options", flush=True)

    def run_greedy(cells, owner, placements, prev_toks_seq, pos_seq):
        """Fill `cells` using the model; mutates owner/placements/prev_toks_seq/pos_seq in place."""
        for cy, cx, cz in cells:
            if owner[cy, cx, cz] >= 0:
                continue
            gy = int(cy * grid / max(1, H))
            gx = int(cx * grid / max(1, X))
            gz = int(cz * grid / max(1, Z))
            gy, gx, gz = min(gy, grid - 1), min(gx, grid - 1), min(gz, grid - 1)
            new_pos = pos_seq + [(gy, gx, gz)]
            new_prev = prev_toks_seq + [n_tokens + 1]                   # "empty" at the new cell
            pos_t = torch.tensor(new_pos, device=device)[None]
            prev_t = torch.tensor(new_prev, device=device)[None]
            chunk_t = pos_t[..., 0]
            with torch.no_grad():
                out = m(v.unsqueeze(0), pos_t, prev_t, kind_t, chunk_id=chunk_t)
                if factored:
                    lc, ls, ly = (o[0, -1] for o in out)
                    lpc = (lc - lc.max() - torch.logsumexp(lc - lc.max(), dim=-1)).cpu().numpy()
                    lps = (ls - ls.max() - torch.logsumexp(ls - ls.max(), dim=-1)).cpu().numpy()
                    lpy = (ly - ly.max() - torch.logsumexp(ly - ly.max(), dim=-1)).cpu().numpy()
                else:
                    logits = out[0, -1]
                    if m_u is not None:
                        out_u = m_u(v.unsqueeze(0), pos_t, prev_t, kind_t, chunk_id=chunk_t)
                        logits_u = out_u[0, -1]
                        logits = logits_u + guidance * (logits - logits_u)      # CFG blend
                    lp = (logits - logits.max() - torch.logsumexp(logits - logits.max(), dim=-1)).cpu().numpy()
            valid = []
            for k, opt in enumerate(opts):
                if factored:
                    h, W, D, cells_arr, part, colours, rot, cid, sid, yid = opt
                else:
                    h, W, D, cells_arr, part, colours, rot, tok = opt
                if cy + h > H or cx + W > X or cz + D > Z:
                    continue
                abs_cells = cells_arr + np.array([cy, cx, cz])
                if (owner[abs_cells[:, 0], abs_cells[:, 1], abs_cells[:, 2]] >= 0).any():
                    continue
                if not target.want[abs_cells[:, 0], abs_cells[:, 1], abs_cells[:, 2]].all():
                    continue
                if factored:
                    score = float(lpc[cid]) + float(lps[sid]) + float(lpy[yid])
                else:
                    score = float(lp[tok])
                score += prefer_length * (len(cells_arr) - 1)
                if support_penalty > 0 and cy > 0:
                    bottom = abs_cells[abs_cells[:, 0] == cy]              # cells at this part's base (y = cy)
                    n_unsup = int((owner[cy - 1, bottom[:, 1], bottom[:, 2]] < 0).sum())
                    if n_unsup > 0:
                        score -= support_penalty * n_unsup
                valid.append((score, k))
            if not valid:
                continue
            valid.sort(reverse=True)
            best_k = valid[0][1]
            opt = opts[best_k]
            h, W, D, cells_arr, part, colours, rot = opt[:7]
            tok = tok_idx.get(part.ldraw, n_tokens) if factored else opt[7]
            abs_cells = cells_arr + np.array([cy, cx, cz])
            allowed_codes = set(colours) if colours else None
            colour = _best_colour(target.colour[abs_cells[:, 0], abs_cells[:, 1], abs_cells[:, 2]], allowed_codes)
            owner[abs_cells[:, 0], abs_cells[:, 1], abs_cells[:, 2]] = len(placements)
            placements.append(Placement(part, colour, cx, cz, cy, W, D, rot))
            prev_toks_seq.append(tok)
            pos_seq.append((gy, gx, gz))

    owner = np.full((H, X, Z), -1, dtype=np.int32)
    placements: list[Placement] = []
    prev_toks_seq: list[int] = []
    pos_seq: list[tuple[int, int, int]] = []
    t0 = time.time()
    run_greedy(initial_cells, owner, placements, prev_toks_seq, pos_seq)
    if verbose:
        print(f"initial: {len(placements)} placements, {int((owner >= 0).sum())}/{int(target.want.sum())} cells covered ({time.time()-t0:.1f}s)")

    # iterative refinement: pick a spatial region, remove every placement whose root is in it,
    # re-greedy that contiguous region as a block.  Random single-placement removal fragments
    # free cells so the greedy fills with 1x1s; a contiguous region gives longer parts a chance.
    rng = np.random.default_rng(seed)
    region_r = max(2, int(round((len(initial_cells) * mask_pct) ** (1/3))))     # cube half-size tuned to mask_pct of volume
    for r in range(refine_rounds):
        t_r = time.time()
        cy = int(rng.integers(0, H)); cx = int(rng.integers(0, X)); cz = int(rng.integers(0, Z))
        y0, y1 = max(0, cy - region_r), min(H, cy + region_r + 1)
        x0, x1 = max(0, cx - region_r), min(X, cx + region_r + 1)
        z0, z1 = max(0, cz - region_r), min(Z, cz + region_r + 1)
        remove_ids = set()
        for i, p in enumerate(placements):
            if y0 <= p.y_plate < y1 and x0 <= p.x_stud < x1 and z0 <= p.z_stud < z1:
                remove_ids.add(i)
        if not remove_ids:
            if verbose:
                print(f"  refine {r+1}: empty region, skipped ({time.time()-t_r:.1f}s)")
            continue
        kept_ids = [i for i in range(len(placements)) if i not in remove_ids]
        new_placements = [placements[i] for i in kept_ids]
        new_prev = [prev_toks_seq[i] for i in kept_ids]
        new_pos = [pos_seq[i] for i in kept_ids]
        new_owner = np.full_like(owner, -1)
        for new_idx, old_idx in enumerate(kept_ids):
            new_owner[owner == old_idx] = new_idx
        freed = [tuple(c) for c in np.argwhere((new_owner < 0) & (target.want > 0))]
        freed.sort()
        pre_kept = len(new_placements)
        run_greedy(freed, new_owner, new_placements, new_prev, new_pos)
        if len(new_placements) <= len(placements):
            if verbose:
                print(f"  refine {r+1}: region [{y0}:{y1},{x0}:{x1},{z0}:{z1}] -{len(remove_ids)} +{len(new_placements) - pre_kept} = {len(new_placements)} parts ({time.time()-t_r:.1f}s)")
            owner, placements, prev_toks_seq, pos_seq = new_owner, new_placements, new_prev, new_pos
        elif verbose:
            print(f"  refine {r+1}: no improvement ({len(new_placements)} > {len(placements)}, {len(remove_ids)} removed), kept original ({time.time()-t_r:.1f}s)")

    secs = time.time() - t0
    if verbose:
        print(f"generated {len(placements)} placements, {int((owner >= 0).sum())}/{int(target.want.sum())} cells covered ({secs:.1f}s)")
    return placements


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
    ap.add_argument("--refine", type=int, default=0,
                    help="iterative-refinement rounds: remove --mask-pct of placements and re-greedy on freed cells")
    ap.add_argument("--mask-pct", type=float, default=0.1,
                    help="fraction of placements to remove per refinement round")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--support-penalty", type=float, default=0.0,
                    help="penalty per unsupported bottom cell in each candidate option (0 disables; 1-5 soft, 1e6 effectively hard)")
    ap.add_argument("--guidance-model", type=Path, default=None,
                    help="path to an 'unconditional' (OMR-only) CNN checkpoint for classifier-free guidance; blends with --model (treated as 'conditional')")
    ap.add_argument("--guidance", type=float, default=1.0,
                    help="CFG scale: logits_cfg = logits_uncond + w * (logits_cond - logits_uncond); 1.0 = pure conditional, >1 amplifies the direction, <1 blends toward unconditional")
    ap.add_argument("--measure", action="store_true",
                    help="run sampler.sample(iters=0, warmup=0) on the CNN output to report full quality metrics (KL, outline, unsupported, ...)")
    a = ap.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    target = target_from_mesh(a.mesh, a.studs, a.shell, a.kind)
    placements = cnn_greedy(target, a.model, n_special=a.n_special, prefer_length=a.prefer_length,
                             refine_rounds=a.refine, mask_pct=a.mask_pct, seed=a.seed,
                             support_penalty=a.support_penalty,
                             guidance_model=a.guidance_model, guidance=a.guidance)
    write_ldr(placements, a.out / "model.ldr")
    if a.measure:
        from everything2lego.sampler import sample as _sample
        from sampler_demo import load_models
        import json
        models = load_models()
        print("measuring metrics via sampler (iters=0, warmup=0)...", flush=True)
        _, info = _sample(target, models, iters=0, warmup=0, init_placements=placements)
        keep = {k: info.get(k) for k in ("parts", "pairs", "kl", "kl_paths", "fidelity", "outline",
                                           "bad_cells", "unsupported", "bare_side_connectors",
                                           "content_loss", "style_loss", "dens", "cnt", "vocabulary")}
        print(json.dumps(keep, indent=2))
        (a.out / "metrics.json").write_text(json.dumps(keep, indent=2))
    subprocess.run([sys.executable, str(ROOT / "tools/render_build.py"), str(a.out / "model.ldr"),
                    str(a.out / "render.png"), "--views", "30,120,210", "--size", "500"], check=False)


if __name__ == "__main__":
    main()
