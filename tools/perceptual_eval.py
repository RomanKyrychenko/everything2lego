"""How much a build looks like the photo it was built from: render it from
a few angles, compare each view to the photo (same size), score by the
colour histogram, the outline and (if scikit-image is there) SSIM.

    python tools/perceptual_eval.py out_dir/model.ldr photo.png
        -> out_dir/perceptual.png (view-by-view strip), prints the scores
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]


def _hist_distance(a: np.ndarray, b: np.ndarray) -> float:
    """L1 distance of normalised 8-bin-per-channel RGB histograms (0..1)."""
    bins = np.linspace(0, 256, 9)
    da = np.stack([np.histogram(a[..., c], bins=bins)[0] for c in range(3)])
    db = np.stack([np.histogram(b[..., c], bins=bins)[0] for c in range(3)])
    return float(np.abs(da / max(1, da.sum()) - db / max(1, db.sum())).sum() / 2)


def _edge_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Share of pixels where edge presence (Sobel-ish) differs (0..1)."""
    def edges(x):
        g = Image.fromarray(x).convert("L").filter(ImageFilter.FIND_EDGES)
        return np.asarray(g) > 48
    return float((edges(a) != edges(b)).mean())


def _ssim(a: np.ndarray, b: np.ndarray) -> float | None:
    try:
        from skimage.metrics import structural_similarity
    except ImportError:
        return None
    return float(structural_similarity(a, b, channel_axis=-1))


def _crop_fit(img: Image.Image, size: int) -> Image.Image:
    """Centre-crop to square (preserves aspect), then resize."""
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    return img.resize((size, size), Image.LANCZOS).convert("RGB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ldr", type=Path); ap.add_argument("photo", type=Path)
    ap.add_argument("--views", default="30,90,150,210,330")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    out = a.out or a.ldr.parent / "perceptual.png"
    tmp = out.with_suffix(".render.png")
    subprocess.run([sys.executable, str(ROOT / "tools/render_build.py"), str(a.ldr), str(tmp),
                    "--views", a.views, "--size", str(a.size), "--cols", str(len(a.views.split(",")))],
                   check=True, capture_output=True)
    sheet = Image.open(tmp)
    n = len(a.views.split(","))
    photo = _crop_fit(Image.open(a.photo), a.size)
    p = np.asarray(photo)
    panel = Image.new("RGB", (sheet.width, a.size * 2 + 40), "white")
    panel.paste(sheet, (0, 0))
    panel.paste(photo.resize((a.size, a.size), Image.LANCZOS), (sheet.width // 2 - a.size // 2, a.size + 20))
    draw = ImageDraw.Draw(panel)
    scores = []
    for i, lon in enumerate(a.views.split(",")):
        view = _crop_fit(sheet.crop((i * sheet.width // n, 0, (i + 1) * sheet.width // n, a.size)), a.size)
        v = np.asarray(view)
        hist = _hist_distance(v, p)
        edge = _edge_distance(v, p)
        ssim = _ssim(v, p)
        scores.append({"lon": lon, "hist": hist, "edge": edge, "ssim": ssim})
        label = f"lon {lon}  hist {hist:.2f}  edge {edge:.2f}" + (f"  ssim {ssim:.2f}" if ssim is not None else "")
        draw.text((i * sheet.width // n + 8, a.size + 2), label, fill="black")
    draw.text((8, a.size * 2 + 24), f"photo (centre)", fill="black")
    panel.save(out)
    tmp.unlink(missing_ok=True)
    print(f"-> {out}")
    for s in scores:
        print(f"  lon {s['lon']:>3}: hist {s['hist']:.3f}  edge {s['edge']:.3f}"
              + (f"  ssim {s['ssim']:.3f}" if s['ssim'] is not None else ""))
    print(f"  mean hist {np.mean([s['hist'] for s in scores]):.3f}  "
          f"edge {np.mean([s['edge'] for s in scores]):.3f}"
          + (f"  ssim {np.mean([s['ssim'] for s in scores]):.3f}" if scores and scores[0]['ssim'] else ""))


if __name__ == "__main__":
    main()
