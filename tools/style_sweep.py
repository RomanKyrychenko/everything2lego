"""The content / style trade-off of the sampler, as style transfer shows
it: the same object built at several style weights, side by side, each
with its content loss (form) and style loss (real-set statistics).

    python tools/style_sweep.py mesh.glb out_dir --kind building [--styles 0.1,0.3,1,3,10] [--iters 100000]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from everything2lego.io_ldraw import write_ldr  # noqa: E402
from everything2lego.sampler import Weights, sample  # noqa: E402
from sampler_demo import load_models, target_from_mesh  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh"); ap.add_argument("out")
    ap.add_argument("--kind", default=None)
    ap.add_argument("--studs", type=int, default=24)
    ap.add_argument("--iters", type=int, default=100000)
    ap.add_argument("--styles", default="0.1,0.3,1,3,10")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    target = target_from_mesh(a.mesh, a.studs, 2, a.kind)
    models = load_models()
    tiles, rows = [], []
    for st in [float(x) for x in a.styles.split(",")]:
        ps, info = sample(target, models, iters=a.iters, seed=a.seed, weights=Weights(style=st))
        d = out / f"style_{st:g}"
        d.mkdir(exist_ok=True)
        write_ldr(ps, d / "model.ldr")
        subprocess.run([sys.executable, str(ROOT / "tools/render_build.py"), str(d / "model.ldr"),
                        str(d / "render.png"), "--views", "30", "--size", "420", "--cols", "1"], check=True,
                       capture_output=True)
        row = {"style": st, **{k: info[k] for k in ("parts", "content_loss", "style_loss", "fidelity", "outline",
                                                    "kl", "kl_paths", "local_per_pair", "unsupported")}}
        rows.append(row)
        print(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}), flush=True)
        tiles.append((Image.open(d / "render.png").convert("RGB"), row))
    w, h = tiles[0][0].size
    sheet = Image.new("RGB", (w * len(tiles), h + 70), "white")
    draw = ImageDraw.Draw(sheet)
    for k, (im, r) in enumerate(tiles):
        sheet.paste(im, (k * w, 0))
        draw.text((k * w + 8, h + 4), f"style {r['style']:g}   {r['parts']} parts", fill="black")
        draw.text((k * w + 8, h + 22), f"form: volume {r['fidelity']:.0f}, outline {r['outline']:.0f}", fill="black")
        draw.text((k * w + 8, h + 40), f"style: KL {r['kl']:.2f} / paths {r['kl_paths']:.2f}", fill="black")
    sheet.save(out / "sweep.png")
    (out / "sweep.json").write_text(json.dumps(rows, indent=1))
    print("wrote", out / "sweep.png")


if __name__ == "__main__":
    main()
