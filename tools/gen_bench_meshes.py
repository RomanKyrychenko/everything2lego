"""Generated meshes of clip_bench's real sets, so the labeller and part matching are measured on the kind of mesh
they meet (a photo reconstructed by an image-to-3D model), not on the LDraw geometry itself.

Per set: LDView draws it from the front, slightly above (white background); Hunyuan3D-2 makes a mesh of that
picture (the pipeline's backends.hunyuan3d), cleaned and coloured from the picture as the pipeline does.  The truth
stays the set's parts: clip_bench --gen maps them onto the generated mesh by nearest face (face_truth aligns it).

    .venv-arm/bin/python tools/gen_bench_meshes.py --out /Volumes/PRO-G40/lego_data/analysis/clip_bench/gen --n 40
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools"), str(ROOT / "tools" / "set_study")]


def main() -> None:
    from clip_bench import sets
    from render_build import LDRAW_CANDIDATES, LDVIEW_CANDIDATES, _first, render_view
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--min-parts", type=int, default=150); ap.add_argument("--max-parts", type=int, default=700)
    ap.add_argument("--lat", type=float, default=20.0); ap.add_argument("--lon", type=float, default=30.0)
    ap.add_argument("--size", type=int, default=1024)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    ldview, ldraw = _first("LDVIEW", LDVIEW_CANDIDATES), _first("LDRAWDIR", LDRAW_CANDIDATES)
    from everything2lego.backends.hunyuan3d import infer_mesh
    from everything2lego.voxelize import clean_mesh, project_photo_colors
    for k, f in enumerate(sets(a.min_parts, a.max_parts)):
        if k >= a.n:
            break
        glb, png = a.out / f"{f.stem}.glb", a.out / f"{f.stem}.png"
        if glb.exists():
            continue
        print(f"== {f.name}", flush=True)
        if not png.exists():
            render_view(f, png, lat=a.lat, lon=a.lon, size=a.size, ldview=ldview, ldraw=ldraw)
        mesh = clean_mesh(infer_mesh(png))
        mesh = project_photo_colors(mesh, png)
        tmp = glb.with_suffix(".tmp.glb"); mesh.export(tmp); tmp.rename(glb)
        print(f"   {len(mesh.faces)} faces -> {glb}", flush=True)


if __name__ == "__main__":
    main()
