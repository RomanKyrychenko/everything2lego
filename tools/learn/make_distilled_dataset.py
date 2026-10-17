"""Generate a sampler-polished dataset from the OMR models.

For each OMR .npz we load the voxel target, reconstruct the original
placement sequence as Placement objects, and run the Metropolis sampler
with those placements as init.  The polished output is saved in the same
.npz schema (voxels, y/x/z/w/d/h, opt_yaw, opt_ldraw, opt_colour, kind)
so the existing OMRDataset in cnn_decoder.py can read it unchanged.

    python tools/learn/make_distilled_dataset.py \
        --out /Volumes/PRO-G40/lego_data/analysis/omr_distilled \
        --n 100 --iters 20000

This is the self-distillation step: we want the CNN to mimic placements
that already satisfy the sampler's energy function (support, outline,
real-set KL), not the raw OMR gold-set placements which are optimized for
display aesthetics but include SNOT connectors, hinges, etc. that the CNN
can't match without the sampler's global structure terms.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from everything2lego.sampler import Target, sample, vocabulary  # noqa: E402
from everything2lego.pack import Placement  # noqa: E402
from sampler_demo import load_models  # noqa: E402

OMR_DIR = Path("/Volumes/PRO-G40/lego_data/analysis/omr_dataset")


def _save(path: Path, voxels: np.ndarray, placements: list, kind: str) -> None:
    ys = np.array([p.y_plate for p in placements], np.int32)
    xs = np.array([p.x_stud for p in placements], np.int32)
    zs = np.array([p.z_stud for p in placements], np.int32)
    ws = np.array([p.w_world for p in placements], np.int32)
    ds = np.array([p.d_world for p in placements], np.int32)
    hs = np.array([p.part.height_units for p in placements], np.int32)
    yaws = np.array([p.rotation_deg for p in placements], np.int32)
    ldraws = np.array([p.part.ldraw for p in placements], dtype=object)
    colours = np.array([p.color_idx for p in placements], np.int32)
    np.savez_compressed(path, voxels=voxels, y=ys, x=xs, z=zs, w=ws, d=ds, h=hs,
                        opt_yaw=yaws, opt_ldraw=ldraws, opt_colour=colours,
                        kind=np.array(kind, dtype=object))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=100, help="number of OMR files to process")
    ap.add_argument("--iters", type=int, default=20000, help="sampler iterations per sample")
    ap.add_argument("--skip-existing", action="store_true")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    vocab = vocabulary()
    part_of = {x.part.ldraw: x.part for x in vocab}
    models = load_models()
    files = sorted(f for f in OMR_DIR.glob("*.npz") if not f.name.startswith("._"))[:a.n]
    print(f"distilling {len(files)} OMR models -> {a.out}  (iters={a.iters})", flush=True)

    for i, fn in enumerate(files):
        out_fn = a.out / fn.name
        if a.skip_existing and out_fn.exists():
            continue
        t0 = time.time()
        try:
            d = np.load(fn, allow_pickle=True)
            voxels = d["voxels"].astype(bool)
            kind = str(d["kind"]) if "kind" in d else None
            init_placements = []
            for j in range(len(d["opt_ldraw"])):
                ldraw = str(d["opt_ldraw"][j])
                part = part_of.get(ldraw)
                if part is None:
                    continue
                init_placements.append(Placement(
                    part=part, color_idx=int(d["opt_colour"][j]),
                    x_stud=int(d["x"][j]), z_stud=int(d["z"][j]), y_plate=int(d["y"][j]),
                    w_world=int(d["w"][j]), d_world=int(d["d"][j]),
                    rotation_deg=int(d["opt_yaw"][j]),
                ))
            colour = np.zeros_like(voxels, dtype=np.int32)
            target = Target(want=voxels, colour=colour, visible=voxels.copy(), kind=kind)
            placements, info = sample(target, models, iters=a.iters, warmup=200,
                                        init_placements=init_placements)
            _save(out_fn, d["voxels"], placements, kind or "")
            print(f"[{i+1}/{len(files)}] {fn.name}: {len(init_placements)} -> {len(placements)} parts, "
                  f"kl={info.get('kl', -1):.3f}, unsup={info.get('unsupported', -1)}, "
                  f"{time.time()-t0:.1f}s", flush=True)
        except Exception as e:
            print(f"[{i+1}/{len(files)}] {fn.name}: FAILED {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
