"""Parallel set-prediction generator using set_decoder (DETR-style).
Runs ONE forward pass to get all N query predictions in parallel, then
filters out "no placement" queries and resolves overlaps by descending
part logprob.

    python tools/set_build.py mesh.glb out/ --studs 32 --kind building
        [--model /path/to/set_decoder.pt]
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
from set_decoder import SetModel  # noqa: E402


def _best_colour(cell_cols, allowed):
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
                     default=Path("/Volumes/PRO-G40/lego_data/analysis/set_decoder.pt"))
    ap.add_argument("--n-special", type=int, default=160)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(a.model, map_location="cpu", weights_only=False)
    tokens, kinds, grid = ckpt["tokens"], ckpt["kinds"], ckpt["grid"]
    n_tokens, n_kinds = len(tokens), len(kinds)
    n_queries = ckpt.get("n_queries", 200)
    tok_idx = {t: i for i, t in enumerate(tokens)}
    kind_idx = {k: i for i, k in enumerate(kinds)}
    ep_saved = ckpt.get("epoch", "?")

    target = target_from_mesh(a.mesh, a.studs, a.shell, a.kind)
    vocab = select_vocabulary(target.kind, a.n_special)
    orients_by_ldraw: dict[int, list] = {}
    for v in vocab:
        tok = tok_idx.get(v.part.ldraw, -1)
        if tok < 0:
            continue
        orients_by_ldraw.setdefault(tok, []).extend(
            [(o.size[0], o.size[1], o.size[2], o.cells, v.part, v.colours, o.rot) for o in v.orients])
    H, X, Z = target.want.shape

    device = "cpu"
    print(f"device: {device}, grid: {grid}, n_queries: {n_queries}, trained epochs: {ep_saved}", flush=True)
    m = SetModel(n_tokens=n_tokens, n_kinds=n_kinds, grid=grid, n_queries=n_queries).to(device)
    m.load_state_dict(ckpt["state_dict"])
    m.eval()

    v = torch.from_numpy((target.want > 0).astype(np.float32))[None, None]
    v = F.interpolate(v, size=(grid, grid, grid), mode="trilinear", align_corners=False)
    v = (v.squeeze(0).squeeze(0) > 0.3).float().to(device)
    kind_i = kind_idx.get(target.kind or "", n_kinds)
    kind_t = torch.tensor([kind_i], device=device)

    t0 = time.time()
    with torch.no_grad():
        part_logits, py, px, pz, yaw_logits = m(v.unsqueeze(0), kind_t)
    part_pred = part_logits[0].argmax(-1).cpu().numpy()
    part_lp = F.log_softmax(part_logits[0], dim=-1).cpu().numpy()
    py_arg = py[0].argmax(-1).cpu().numpy()
    px_arg = px[0].argmax(-1).cpu().numpy()
    pz_arg = pz[0].argmax(-1).cpu().numpy()
    yaw_arg = yaw_logits[0].argmax(-1).cpu().numpy()
    print(f"set.forward: {time.time()-t0:.2f}s, n_queries={len(part_pred)}", flush=True)

    no_placement = m.no_placement_idx
    scores = []
    for q in range(len(part_pred)):
        p = part_pred[q]
        if p == no_placement:
            continue
        scores.append((float(part_lp[q, p]), q))
    scores.sort(reverse=True)
    print(f"non-nop queries: {len(scores)}/{len(part_pred)}", flush=True)

    sy, sx, sz = H / max(1, grid), X / max(1, grid), Z / max(1, grid)
    owner = np.full((H, X, Z), -1, dtype=np.int32)
    placements: list[Placement] = []
    yaw_to_rot = {0: 0, 1: 90, 2: 180, 3: 270}
    for _, q in scores:
        tok = int(part_pred[q])
        orients = orients_by_ldraw.get(tok)
        if not orients:
            continue
        cy = int(py_arg[q] * sy)
        cx = int(px_arg[q] * sx)
        cz = int(pz_arg[q] * sz)
        want_rot = yaw_to_rot.get(int(yaw_arg[q]), 0)
        orient = None
        for o in orients:
            if o[6] == want_rot:
                orient = o; break
        if orient is None:
            orient = orients[0]
        h, W, D, cells_arr, part, colours, rot = orient
        if cy + h > H or cx + W > X or cz + D > Z:
            continue
        abs_cells = cells_arr + np.array([cy, cx, cz])
        if (owner[abs_cells[:, 0], abs_cells[:, 1], abs_cells[:, 2]] >= 0).any():
            continue
        if not target.want[abs_cells[:, 0], abs_cells[:, 1], abs_cells[:, 2]].all():
            continue
        allowed_codes = set(colours) if colours else None
        colour = _best_colour(target.colour[abs_cells[:, 0], abs_cells[:, 1], abs_cells[:, 2]], allowed_codes)
        owner[abs_cells[:, 0], abs_cells[:, 1], abs_cells[:, 2]] = len(placements)
        placements.append(Placement(part, colour, cx, cz, cy, W, D, rot))

    secs = time.time() - t0
    covered = int((owner >= 0).sum())
    total = int(target.want.sum())
    print(f"generated {len(placements)} placements, {covered}/{total} cells covered ({secs:.1f}s)")
    write_ldr(placements, a.out / "model.ldr")
    subprocess.run([sys.executable, str(ROOT / "tools/render_build.py"), str(a.out / "model.ldr"),
                    str(a.out / "render.png"), "--views", "30,120,210", "--size", "500"], check=False)


if __name__ == "__main__":
    main()
