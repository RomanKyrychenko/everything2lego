"""Perceptual match score between a photo and one or more LEGO builds,
in VGG16 feature space.  Lower is a build whose render looks more like
the photo.

    python tools/perceptual_match.py photo.png build1/model.ldr build2/model.ldr
        [--view LAT,LON]  [--size 256]  [--sam SAM_CHECKPOINT]

Renders each build at `--view` (default: silhouette-fit against the
photo's biggest SAM mask, same policy as segment_photo.py).  Computes
VGG16 perceptual loss (semantic_v2.perceptual.PerceptualLoss, uses
torchvision's ImageNet weights) between each render and the photo,
cropped and resampled to the same resolution.  Prints a ranked table.

Needs: torch, torchvision, segment-anything, LDView (set LDVIEW).  Runs
in the Intel .venv under Rosetta (the arm venv has no torch).

shortcut: no sampler-side integration; the metric is post-hoc so users
can run the sampler several times with different seeds and keep the
build whose render matches best.  A sampler loss term that evaluates
the metric periodically during annealing is the next upgrade.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from render_build import _first, LDRAW_CANDIDATES, LDVIEW_CANDIDATES, render_view  # noqa: E402

DEFAULT_SAM = os.environ.get("SAM_CHECKPOINT", "/Volumes/PRO-G40/models/sam/sam_vit_b_01ec64.pth")


def _fit_view(photo: Image.Image, mesh_path: Path | None, sam: str) -> tuple[float, float]:
    """Pick the view by silhouette-matching the photo's biggest SAM mask
    against the paired mesh (same policy as tools/segment_photo.py).  If
    no mesh is given, defaults to (0, 0)."""
    if mesh_path is None or not mesh_path.exists():
        return 0.0, 0.0
    import trimesh
    from segment_photo import _sam_masks, fit_view  # noqa: E402
    masks = _sam_masks(photo, sam)
    if not masks:
        return 0.0, 0.0
    biggest = np.asarray(max(masks, key=lambda m: m["area"])["segmentation"], bool)
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    view, _ = fit_view(mesh, biggest)
    return view


def _load_rgba(path: Path, size: int) -> np.ndarray:
    """Resize to size x size, drop alpha, return (H, W, 3) float in [0, 1]."""
    img = Image.open(path).convert("RGBA").resize((size, size), Image.LANCZOS)
    arr = np.asarray(img, np.float32) / 255.0
    rgb, a = arr[..., :3], arr[..., 3:4]
    return rgb * a + (1.0 - a)                                        # composite on white


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("photo", type=Path)
    ap.add_argument("builds", type=Path, nargs="+", help="model.ldr paths or directories containing one")
    ap.add_argument("--view", default="auto", help="LAT,LON; 'auto' fits by SAM silhouette if --mesh is given")
    ap.add_argument("--mesh", type=Path, default=None, help="paired mesh for 'auto' view fitting")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--sam", default=DEFAULT_SAM)
    a = ap.parse_args()

    photo = Image.open(a.photo).convert("RGB")
    if a.view == "auto":
        view = _fit_view(photo, a.mesh, a.sam)
        print(f"fit view lat={view[0]:+.0f} lon={view[1]:+.0f}", flush=True)
    else:
        view = tuple(float(v) for v in a.view.split(","))

    ldview = _first("LDVIEW", LDVIEW_CANDIDATES)
    ldraw = _first("LDRAWDIR", LDRAW_CANDIDATES)
    photo_arr = np.asarray(photo.resize((a.size, a.size), Image.LANCZOS), np.float32) / 255.0

    import torch
    from everything2lego.semantic_v2.perceptual import PerceptualLoss
    ploss = PerceptualLoss(device="cpu")
    tgt = torch.from_numpy(photo_arr)

    rows = []
    with tempfile.TemporaryDirectory() as td:
        for b in a.builds:
            ldr = b if b.suffix == ".ldr" else b / "model.ldr"
            if not ldr.exists():
                print(f"  {b}: no model.ldr", flush=True); continue
            png = Path(td) / f"{ldr.parent.name}.png"
            missing = render_view(ldr, png, lat=view[0], lon=view[1], size=a.size,
                                  ldview=ldview, ldraw=ldraw)
            if missing:
                print(f"  {ldr}: LDView missed {len(missing)} parts", flush=True)
            rendered = _load_rgba(png, a.size)
            with torch.no_grad():
                score = float(ploss.compute(torch.from_numpy(rendered), tgt))
            rows.append((score, str(ldr)))
    for score, ldr in sorted(rows):
        print(f"  perceptual {score:.4f}  {ldr}")


if __name__ == "__main__":
    main()
