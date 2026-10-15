"""Run the sampler N times with different seeds and keep the build whose
LDView render matches the photo best in VGG16 feature space.  A
practical loop-closer: the sampler is a stochastic walk, so different
seeds yield different-looking builds; pick the one that actually looks
like the photo.

    python tools/sample_best.py photo.png mesh.glb out/ --seeds 5
        [--studs 20]  [--kind building]  [--segmentation regions.npz]
        [--iters 20000]  [--view LAT,LON]

Runs under the Intel .venv (needs torch + torchvision + LDView), spawns
the sampler itself in the arm venv for speed.  View auto-fits to the
photo's biggest SAM mask when --segmentation is given (it already stores
`carve` which was computed at that view) or when --mesh is available for
fit_view; otherwise 0,0 or --view LAT,LON.

Output: writes the winning build's model.ldr/render.png/info.json to
`out/`, the per-seed scores and models under `out/_seeds/`.
"""
from __future__ import annotations

import argparse
import os
import shutil
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

ARM_PY = ROOT / ".venv-arm" / "bin" / "python"
DEFAULT_SAM = os.environ.get("SAM_CHECKPOINT", "/Volumes/PRO-G40/models/sam/sam_vit_b_01ec64.pth")


def _fit_view(photo: Image.Image, mesh: Path | None, sam: str) -> tuple[float, float]:
    if mesh is None or not mesh.exists():
        return 0.0, 0.0
    import trimesh
    from segment_photo import _sam_masks, fit_view
    masks = _sam_masks(photo, sam)
    if not masks:
        return 0.0, 0.0
    biggest = np.asarray(max(masks, key=lambda m: m["area"])["segmentation"], bool)
    m = trimesh.load(mesh, force="mesh", process=False)
    v, _ = fit_view(m, biggest)
    return v


def _photo_tensor(photo: Image.Image, size: int):
    import torch
    arr = np.asarray(photo.convert("RGB").resize((size, size), Image.LANCZOS), np.float32) / 255.0
    return torch.from_numpy(arr)


def _render_tensor(ldr: Path, view: tuple[float, float], size: int, ldview: str, ldraw: str):
    import torch
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        png = Path(f.name)
    try:
        render_view(ldr, png, lat=view[0], lon=view[1], size=size, ldview=ldview, ldraw=ldraw)
        img = Image.open(png).convert("RGBA").resize((size, size), Image.LANCZOS)
        arr = np.asarray(img, np.float32) / 255.0
        rgb, a = arr[..., :3], arr[..., 3:4]
        comp = rgb * a + (1.0 - a)                                    # white matte
        return torch.from_numpy(comp)
    finally:
        png.unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("photo", type=Path)
    ap.add_argument("mesh", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--studs", type=int, default=24)
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--shell", type=int, default=2)
    ap.add_argument("--kind", default=None)
    ap.add_argument("--style", type=float, default=1.0)
    ap.add_argument("--stand", type=int, default=0)
    ap.add_argument("--regions", default=None)
    ap.add_argument("--segmentation", default=None)
    ap.add_argument("--view", default="auto")
    ap.add_argument("--size", type=int, default=256, help="render resolution for scoring")
    ap.add_argument("--sam", default=DEFAULT_SAM)
    a = ap.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    seeds_dir = a.out / "_seeds"
    seeds_dir.mkdir(exist_ok=True)

    photo = Image.open(a.photo).convert("RGB")
    if a.view == "auto":
        view = _fit_view(photo, a.mesh, a.sam)
        print(f"fit view lat={view[0]:+.0f} lon={view[1]:+.0f}", flush=True)
    else:
        view = tuple(float(v) for v in a.view.split(","))

    ldview = _first("LDVIEW", LDVIEW_CANDIDATES)
    ldraw = _first("LDRAWDIR", LDRAW_CANDIDATES)

    import torch
    from everything2lego.semantic_v2.perceptual import PerceptualLoss
    ploss = PerceptualLoss(device="cpu")
    tgt = _photo_tensor(photo, a.size)

    rows: list[tuple[float, int, Path]] = []
    for s in range(a.seeds):
        seed_out = seeds_dir / f"seed{s}"
        cmd = [str(ARM_PY), str(ROOT / "tools/sampler_demo.py"),
               str(a.mesh), str(seed_out),
               "--studs", str(a.studs), "--iters", str(a.iters), "--shell", str(a.shell),
               "--style", str(a.style), "--stand", str(a.stand), "--seed", str(s)]
        if a.kind:
            cmd += ["--kind", a.kind]
        if a.regions:
            cmd += ["--regions", a.regions]
        if a.segmentation:
            cmd += ["--segmentation", a.segmentation]
        print(f"=== seed {s}/{a.seeds} ===", flush=True)
        subprocess.run(cmd, check=True)
        ldr = seed_out / "model.ldr"
        rendered = _render_tensor(ldr, view, a.size, ldview, ldraw)
        with torch.no_grad():
            score = float(ploss.compute(rendered, tgt))
        print(f"  perceptual {score:.4f}", flush=True)
        rows.append((score, s, seed_out))

    rows.sort()
    best_score, best_seed, best_dir = rows[0]
    print()
    print("ranked:")
    for score, s, _ in rows:
        print(f"  seed {s}  perceptual {score:.4f}" + ("  <= best" if s == best_seed else ""))

    for name in ("model.ldr", "render.png", "info.json"):
        src = best_dir / name
        if src.exists():
            shutil.copy(src, a.out / name)
    (a.out / "best.json").write_text(
        '{"seed": %d, "perceptual": %.4f, "view": [%g, %g]}\n' % (best_seed, best_score, view[0], view[1]))
    print(f"\nwinner: seed {best_seed} -> {a.out}/model.ldr (perceptual {best_score:.4f})")


if __name__ == "__main__":
    main()
