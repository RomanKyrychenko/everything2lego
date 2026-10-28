"""Mesh2Brick end to end on a mesh: voxelize at LEGO scale (the test path) -> (96,32,32) target grid in the training
frame -> free-run with constraints (m2b_gen.generate) -> LDR + render sheet.

    BRICKNET_DATA=/Volumes/PRO-G40/lego_data/bricknet .venv/bin/python tools/learn/m2b_watermill.py \
        --mesh examples/pixellabs-watermill-3425.glb --ckpt ~/Downloads/model.pt --out $LEGO_DATA/analysis/m2b/watermill
    ... --grid-only      # just report the voxel grid"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tools" / "set_study"))


def mesh_to_grid(mesh_path: Path, grid=(96, 32, 32), max_studs: int = 32, with_colours: bool = False):
    """voxelize() output (y,x,z, ground at y=0) placed in the training frame: x/z centred by whole studs, y cropped.
    with_colours: also return an (H,W,D) int grid of LDraw colour codes (-1 = empty) from the mesh."""
    from everything2lego.colors import palette
    from everything2lego.voxelize import load_mesh, voxelize
    vg = voxelize(load_mesh(mesh_path), max_studs=max_studs); occ = vg.occupied(); codes = np.array([c.code for c in palette()])
    H, W, D = grid; out = np.zeros(grid, bool); col = np.full(grid, -1, np.int64)
    ys, xs, zs = np.nonzero(occ)
    sx = int(xs.min()) + round((xs.max() + 1 - xs.min()) / 2) - W // 2; sz = int(zs.min()) + round((zs.max() + 1 - zs.min()) / 2) - D // 2
    for y, x, z in zip(ys, xs, zs):
        yy, xx, zz = y - int(ys.min()), x - sx, z - sz
        if 0 <= yy < H and 0 <= xx < W and 0 <= zz < D: out[yy, xx, zz] = True; col[yy, xx, zz] = codes[vg.palette_idx[y, x, z]]
    return (out, col) if with_colours else out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", type=Path, default=ROOT / "examples" / "pixellabs-watermill-3425.glb")
    ap.add_argument("--ckpt", type=Path); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-studs", type=int, default=32); ap.add_argument("--grid-only", action="store_true")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--max-steps", type=int, default=400); ap.add_argument("--retries", type=int, default=8); ap.add_argument("--restart", action="store_true"); ap.add_argument("--max-comps", type=int, default=64); ap.add_argument("--cover", action="store_true"); ap.add_argument("--max-bricks", type=int, default=4000); ap.add_argument("--max-roots", type=int, default=1)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    target, colours = mesh_to_grid(a.mesh, max_studs=a.max_studs, with_colours=True); ys, xs, zs = np.nonzero(target)
    print(f"target: {target.sum()} cells ({target.mean():.2%})  y[{ys.min()},{ys.max()}] x[{xs.min()},{xs.max()}] z[{zs.min()},{zs.max()}]", flush=True)
    np.save(a.out / "target.npy", target)
    if a.grid_only: return
    import torch
    from m2b_gen import Catalog, generate
    from m2b_model import Mesh2Brick
    from m2b_train import Vocab
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False); args = ck["args"]; grid = tuple(ck["grid"])
    vocab = Vocab(); model = Mesh2Brick(vocab.n_part, len(vocab.colors), grid, d=args["d"], layers=args["layers"], C=args["C"])
    model.load_state_dict(ck["model"]); model.eval(); cat = Catalog(vocab.n_part)
    print(f"checkpoint step {ck.get('step')}  model {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M", flush=True)
    if a.cover:
        from m2b_gen import generate_cover
        r = generate_cover(model, vocab, cat, target, grid, torch.device("cpu"), colours=colours, max_bricks=a.max_bricks, seed=a.seed, max_roots=a.max_roots)
    else: r = generate(model, vocab, cat, target, grid, torch.device("cpu"), max_steps=a.max_steps, seed=a.seed, retries=a.retries, restart=a.restart, max_comps=a.max_comps)
    ldr = a.out / "watermill.ldr"; ldr.write_text(r["ldr"])
    print(f"built {r['n']} bricks in {r.get('components', 1)} components  filled {r.get('filled', '?')}/{r.get('target', '?')}  IoU {r['iou']:.3f}  rejected {r['rejected']}", flush=True)
    json.dump({k: v for k, v in r.items() if k != "ldr"}, open(a.out / "summary.json", "w"), indent=1)
    if r["n"]:
        subprocess.run([sys.executable, str(ROOT / "tools" / "render_build.py"), str(ldr), str(a.out / "watermill.png"), "--views", "45,135,225,315", "--top"], check=False)
        print(f"render -> {a.out / 'watermill.png'}", flush=True)


if __name__ == "__main__":
    main()
