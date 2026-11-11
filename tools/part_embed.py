"""Every vocabulary part's own embedding, by what it is called and what it looks like, in the labeller's image-text
space (semantic_labels.MODEL) -- so a mesh region is matched to parts one by one, not to a category.

  - text: its Rebrickable name ("Window 1 x 2 x 2 Castle"), embedded by the model's text side;
  - look: its real LDraw geometry, drawn by semantic_labels.render (the renderer the mesh regions go through) from
    four sides, slightly from above, studs-up, in its own colours (the 'main colour' faces mid grey), embedded by the
    image side, the views averaged.

Writes <out>/<model>.npz: ldraw (n,), name (n,), category (n,), text (n, d), look (n, d) float16, unit rows.

    .venv-arm/bin/python tools/part_embed.py [--out /Volumes/PRO-G40/lego_data/analysis/part_embed]
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools"), str(ROOT / "tools" / "set_study")]
OUT = Path("/Volumes/PRO-G40/lego_data/analysis/part_embed")
GREY = (160, 165, 169)                                   # light bluish grey, for the part's own colour (code 16)
VIEWS = [((0.0, 0.0, 1.0), (0.0, -20.0)), ((1.0, 0.0, 0.0), (0.0, -20.0)),
         ((0.0, 0.0, -1.0), (0.0, -20.0)), ((-1.0, 0.0, 0.0), (0.0, -20.0))]


def part_images(name: str, size: int = 224) -> list:
    """The part drawn from VIEWS (normal the side shown, view (yaw, pitch))."""
    import semantic_labels as S
    from ldr_to_mesh import _part, _rgb
    tri, code = _part(name)
    if not len(tri):
        return []
    rgb = _rgb()
    # LDU, y down -> studs, y up; LDraw winding is not reliable and render() drops faces turned away: both windings
    tri = np.concatenate([tri, tri[:, [0, 2, 1]]]); code = np.concatenate([code, code])
    P = (tri.reshape(-1, 3) * np.array([1.0, -1.0, 1.0])) / 20.0
    faces = np.arange(len(P)).reshape(-1, 3)
    t = P[faces]; fn = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]); fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
    fcol = np.array([GREY if c == 16 else rgb.get(int(c), GREY) for c in code], float)
    allf = np.arange(len(faces))
    return [S.render(P, faces, fcol, fn, allf, np.array(n), size=size, alone=True, view=v) for n, v in VIEWS]


def main() -> None:
    import semantic_labels as S
    from everything2lego.sampler import vocabulary
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT); ap.add_argument("--clip", default=S.DEFAULT_MODEL)
    ap.add_argument("--sheet", type=Path, default=None, help="also a contact sheet of the first 48 parts' first view")
    a = ap.parse_args()
    S.MODEL = a.clip
    seen, parts = set(), []
    for v in vocabulary():
        if v.part.ldraw not in seen:
            seen.add(v.part.ldraw); parts.append(v)
    db = sqlite3.connect("/Volumes/PRO-G40/lego_data/parts.sqlite")
    names = []
    for v in parts:
        r = db.execute("select p.name from ldraw_name l join part p on p.part_num = l.part_num where l.ldraw = ?",
                       (v.part.ldraw,)).fetchone()
        names.append((r[0] if r else "") or v.part.name)
    print(f"{len(parts)} parts", flush=True)
    text = S.embed_texts(names)
    look, ok, sheet = [], [], []
    for i, v in enumerate(parts):
        ims = part_images(v.part.ldraw)
        if not ims:
            look.append(np.zeros(text.shape[1])); ok.append(False); continue
        e = S.embed_images(ims).mean(0); look.append(e / np.linalg.norm(e)); ok.append(True)
        if a.sheet and len(sheet) < 48:
            sheet.append((ims[0], names[i]))
        if i % 100 == 0:
            print(f"  {i}/{len(parts)}", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    f = a.out / (a.clip.replace("/", "__") + ".npz")
    np.savez_compressed(f, ldraw=np.array([v.part.ldraw for v in parts]), name=np.array(names),
                        category=np.array([v.category for v in parts]), text=text.astype(np.float16),
                        look=np.array(look, np.float16), has_look=np.array(ok))
    print(f"-> {f}: {sum(ok)}/{len(parts)} with a look")
    if a.sheet:
        from PIL import Image, ImageDraw
        S_ = 160; im = Image.new("RGB", (8 * S_, 6 * (S_ + 14)), "white"); d = ImageDraw.Draw(im)
        for k, (x, n) in enumerate(sheet):
            im.paste(x.resize((S_, S_)), ((k % 8) * S_, (k // 8) * (S_ + 14))); d.text(((k % 8) * S_ + 2, (k // 8) * (S_ + 14) + S_), n[:26], fill="black")
        im.save(a.sheet); print(f"-> {a.sheet}")


if __name__ == "__main__":
    main()
