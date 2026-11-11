"""What the picture of an object says, gathered once per picture (no per-region work): image embeddings of tiles over
the object, and the object's parts named by a local VLM -- the appearance evidence the per-cell part scorer reads.

  - tiles: the picture cropped to its subject (as clip_bench.photo_of crops it, so cell_bench's uv index it), cut
    into an n x n grid for each --scales n, each tile 1.5x its cell (some context), embedded by SigLIP 2;
  - names: qwen3.5 (Ollama) lists the visible parts of the object the picture shows (one call), each embedded as
    "a photo of a <name>" in the same space.

Writes <out>/<stem>.evidence.npz: tile_box (n, 4) x0 y0 x1 y1 in the cropped picture, tile_scale (n,), tile_emb (n, d)
float16, names (m,), name_emb (m, d) float16, secs_tiles, secs_names.

    tools/timed.sh picture-evidence .venv-arm/bin/python tools/picture_evidence.py /Volumes/PRO-G40/lego_data/analysis/clip_bench/gen --out /Volumes/PRO-G40/lego_data/analysis/cell_bench
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]
PROMPT = ("This picture shows a model of an object or scene. List the distinct visible parts of what it depicts -- "
          "structural parts, features and details (for a house: roof, chimney, window, door, wall, steps...). "
          "Answer with JSON only: {\"parts\": [10 to 25 short noun phrases]}")


def name_parts(img: np.ndarray, model: str) -> list[str]:
    from PIL import Image
    b = io.BytesIO(); Image.fromarray(img).save(b, "PNG")
    req = {"model": model, "stream": False, "think": False, "format": "json", "options": {"temperature": 0},
           "messages": [{"role": "user", "content": PROMPT, "images": [base64.b64encode(b.getvalue()).decode()]}]}
    r = json.load(urllib.request.urlopen(urllib.request.Request("http://localhost:11434/api/chat", json.dumps(req).encode(),
                                                                {"Content-Type": "application/json"}), timeout=600))
    try:
        return [str(o).strip().lower() for o in json.loads(r["message"]["content"])["parts"]][:25]
    except Exception:                                    # noqa: BLE001
        return []


def tiles(img: np.ndarray, scales) -> tuple[np.ndarray, np.ndarray, list]:
    from PIL import Image
    h, w = img.shape[:2]
    boxes, sc, ims = [], [], []
    pic = Image.fromarray(img)
    for n in scales:
        cw, ch = w / n, h / n; side = 1.5 * max(cw, ch)
        for i in range(n):
            for j in range(n):
                cx, cy = (j + 0.5) * cw, (i + 0.5) * ch
                b = (int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2))
                boxes.append(b); sc.append(n)
                ims.append(pic.crop(b).resize((384, 384)))   # outside the picture: black; the subject crop's edge
    return np.array(boxes), np.array(sc), ims


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("gen", type=Path); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scales", default="4,8"); ap.add_argument("--vlm", default="qwen3.5:9b")
    ap.add_argument("--clip", default="google/siglip2-so400m-patch14-384")
    ap.add_argument("--no-names", action="store_true")
    a = ap.parse_args()
    import semantic_labels as S
    from PIL import Image
    from everything2lego.voxelize import _crop_to_subject
    S.MODEL = a.clip
    for png in sorted(a.gen.glob("[!.]*.png")):
        f = a.out / f"{png.stem}.evidence.npz"
        if f.exists() or not png.with_suffix(".glb").exists():
            continue
        img = _crop_to_subject(np.asarray(Image.open(png).convert("RGB")))
        t0 = time.time()
        box, sc, ims = tiles(img, [int(x) for x in a.scales.split(",")])
        te = S.embed_images(ims).astype(np.float16)
        t1 = time.time()
        names = [] if a.no_names else name_parts(img, a.vlm)
        ne = S.embed_texts([f"a photo of a {n}" for n in names]).astype(np.float16) if names else np.zeros((0, te.shape[1]), np.float16)
        t2 = time.time()
        np.savez_compressed(f, tile_box=box, tile_scale=sc, tile_emb=te, names=np.array(names), name_emb=ne,
                            secs_tiles=t1 - t0, secs_names=t2 - t1)
        print(f"{png.stem}: {len(ims)} tiles {t1 - t0:.1f} s, {len(names)} names {t2 - t1:.1f} s: {', '.join(names[:12])}", flush=True)


if __name__ == "__main__":
    main()
