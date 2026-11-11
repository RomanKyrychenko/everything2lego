"""Per-pixel appearance of an object's picture, once per picture: DINOv2 patch features over the picture cropped to its
subject (as clip_bench.photo_of crops it, so cell_bench's uv index it), padded square and resized to --size (patches
of 14 px).  cell_scorer's `knn` feature reads a cell's patch and votes with the true parts of the most alike cells of
other sets -- appearance matched to appearance, not to a part's render or name.

Writes <out>/<stem>.patch.npz: feat (g, g, d) float16, unit length; off (2,) the crop's offset in the square, side
(its side, px): a pixel (u, v) of the crop is patch ((v, u) + off[::-1]) * g / side.

    tools/timed.sh patch-evidence .venv-arm/bin/python tools/patch_evidence.py /Volumes/PRO-G40/lego_data/analysis/clip_bench/gen --out /Volumes/PRO-G40/lego_data/analysis/cell_bench
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]


def patch_feats(img: np.ndarray, model, proc, size: int) -> tuple[np.ndarray, np.ndarray, int]:
    import torch
    from PIL import Image
    h, w = img.shape[:2]; side = max(h, w)
    off = np.array([(side - w) // 2, (side - h) // 2])
    sq = np.zeros((side, side, 3), np.uint8); sq[off[1]:off[1] + h, off[0]:off[0] + w] = img
    x = proc(images=Image.fromarray(sq).resize((size, size)), return_tensors="pt", do_resize=False, do_center_crop=False)
    with torch.no_grad():
        t = model(**x).last_hidden_state[0, 1:]                     # drop CLS (dinov2-base: no register tokens)
    g = size // 14
    f = t.reshape(g, g, -1).numpy()
    return (f / np.linalg.norm(f, axis=-1, keepdims=True)).astype(np.float16), off, side


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("gen", type=Path); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="facebook/dinov2-base"); ap.add_argument("--size", type=int, default=672)
    a = ap.parse_args()
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel
    from everything2lego.voxelize import _crop_to_subject
    proc = AutoImageProcessor.from_pretrained(a.model); model = AutoModel.from_pretrained(a.model).eval()
    for png in sorted(a.gen.glob("[!.]*.png")):
        f = a.out / f"{png.stem}.patch.npz"
        if f.exists() or not png.with_suffix(".glb").exists():
            continue
        t0 = time.time()
        feat, off, side = patch_feats(_crop_to_subject(np.asarray(Image.open(png).convert("RGB"))), model, proc, a.size)
        np.savez_compressed(f, feat=feat, off=off, side=side, secs=time.time() - t0)
        print(f"{png.stem}: {feat.shape} {time.time() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    main()
