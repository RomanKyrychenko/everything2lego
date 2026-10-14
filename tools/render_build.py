"""Render a built model with the real LDraw parts, from several sides, into
one PNG — to look at a build the way a builder would, not by its counts.

    python tools/render_build.py out/lego/model.ldr build.png
    python tools/render_build.py model.ldr build.png --views 45,135,225,315 --top

Uses LDView (https://github.com/tcobbs/ldview) and an LDraw parts
library.  Defaults: LDView from $LDVIEW or /Volumes/PRO-G40/ldview, the
library from $LDRAWDIR or the copy BrickLink Studio ships.  Each view is a
longitude around the model at --lat degrees above it; the sheet is a grid
of views, each labelled with its longitude.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

LDVIEW_CANDIDATES = [
    "/Volumes/PRO-G40/ldview/LDView.app/Contents/MacOS/LDView",
    "/Applications/LDView.app/Contents/MacOS/LDView",
]
LDRAW_CANDIDATES = [
    "/Applications/Studio 2.0/ldraw",
    os.path.expanduser("~/ldraw"),
]


def _first(env: str, candidates: list[str]) -> str:
    path = os.environ.get(env)
    if path:
        return path
    for c in candidates:
        if os.path.exists(c):
            return c
    sys.exit(f"{env} not set and none of {candidates} exists")


def render_view(ldr: Path, png: Path, *, lat: float, lon: float, size: int,
                ldview: str, ldraw: str) -> list[str]:
    """One view; returns the parts LDView could not find."""
    cmd = [ldview, f"-LDrawDir={ldraw}", str(ldr), f"-SaveSnapshot={png}",
           f"-SaveWidth={size}", f"-SaveHeight={size}", "-SaveActualSize=0",
           f"-WindowWidth={size}", f"-WindowHeight={size}",
           "-SaveZoomToFit=1", f"-DefaultLatLong={lat},{lon}",
           "-BackgroundColor3=0xFFFFFF"]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if not png.exists():
        sys.exit(f"LDView wrote nothing:\n{res.stdout}\n{res.stderr}")
    out = res.stdout + res.stderr
    return sorted({ln.split("submodel ")[1].rstrip(".")
                   for ln in out.splitlines() if "Error loading submodel" in ln})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ldr", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--views", default="0,45,90,180,270",
                    help="comma-separated longitudes in degrees")
    ap.add_argument("--lat", type=float, default=25.0)
    ap.add_argument("--top", action="store_true", help="add a view from straight above")
    ap.add_argument("--size", type=int, default=700)
    ap.add_argument("--cols", type=int, default=3)
    args = ap.parse_args()

    ldview = _first("LDVIEW", LDVIEW_CANDIDATES)
    ldraw = _first("LDRAWDIR", LDRAW_CANDIDATES)
    views = [(args.lat, float(v)) for v in args.views.split(",") if v.strip()]
    if args.top:
        views.append((89.0, 0.0))

    tiles, missing = [], set()
    with tempfile.TemporaryDirectory(prefix="render-build-") as tmp:
        for i, (lat, lon) in enumerate(views):
            png = Path(tmp) / f"v{i}.png"
            missing |= set(render_view(args.ldr, png, lat=lat, lon=lon, size=args.size,
                                       ldview=ldview, ldraw=ldraw))
            tile = Image.open(png).convert("RGB")
            ImageDraw.Draw(tile).text((8, 8), "top" if lat > 80 else f"lon {lon:g}",
                                      fill=(0, 0, 0))
            tiles.append(tile)

    cols = min(args.cols, len(tiles))
    rows = -(-len(tiles) // cols)
    sheet = Image.new("RGB", (cols * args.size, rows * args.size), "white")
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * args.size, (i // cols) * args.size))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out)
    print(f"wrote {args.out} ({len(tiles)} views)")
    if missing:
        print("parts missing from the library: " + ", ".join(sorted(missing)))


if __name__ == "__main__":
    main()
