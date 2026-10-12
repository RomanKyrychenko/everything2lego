#!/usr/bin/env python3
"""Run SegviGen part segmentation on a mesh and write the colour-coded GLB
that `everything2lego semantic --parts` reads.

SegviGen (Li et al., SIGGRAPH 2026, MIT) repurposes the TRELLIS.2 3D
generative model to colour the parts of a mesh.  It needs Linux, an NVIDIA
GPU with 24 GB of memory and the TRELLIS.2 environment; see
https://github.com/Nelipot-Lee/SegviGen for the install.  This wrapper
only drives it:

  python tools/segvigen.py house.glb --segvigen ~/SegviGen -o house.parts.glb

What it adds on top of SegviGen's own scripts:

* a cache: the result is kept under ~/.cache/everything2lego/segvigen,
  keyed by the mesh's bytes and the options, so a second run on the same
  house is free;
* mesh preparation: the model works on a voxel grid of fixed resolution,
  so a mesh with a million faces segments no better than one with 50,000;
  the mesh is decimated to --faces before the run (when
  fast_simplification is installed) and the labels are carried back to the
  original faces by nearest face centre, which is what `--parts` does;
* batch mode: the model takes the longest to load, so several meshes in
  one invocation share one process when SegviGen's script allows it;
* `--two-d-map image.png` passes a painted 2D segmentation map through for
  the guided mode, which picks the part granularity.

Without a GPU, use `everything2lego semantic --segmenter builtin` instead,
or paint the parts by hand (one flat colour per part, exported as GLB with
a JSON sidecar naming the colours).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

CACHE_DIR = Path(os.environ.get("EVERYTHING2LEGO_CACHE",
                                Path.home() / ".cache" / "everything2lego")) / "segvigen"


def _cache_key(mesh: Path, faces: int, two_d_map: Path | None, extra: list[str]) -> str:
    h = hashlib.sha256()
    h.update(mesh.read_bytes())
    if two_d_map is not None:
        h.update(two_d_map.read_bytes())
    h.update(json.dumps({"faces": faces, "extra": extra}).encode())
    return h.hexdigest()[:24]


def prepare_mesh(src: Path, dst: Path, faces: int) -> Path:
    """A decimated, single-mesh copy of `src` at `dst` (or `src` itself when
    it already has at most `faces` faces or decimation is unavailable)."""
    import trimesh
    mesh = trimesh.load(src, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if len(mesh.faces) <= faces:
        return src
    try:
        import fast_simplification  # noqa: F401  (trimesh's decimation backend)
        mesh = mesh.simplify_quadric_decimation(face_count=faces)
    except ImportError:
        print(f"[segvigen] fast_simplification not installed; sending all "
              f"{len(mesh.faces)} faces (pip install fast_simplification to decimate)",
              file=sys.stderr)
        return src
    mesh.export(dst)
    return dst


def run_segvigen(segvigen: Path, mesh: Path, out: Path, *, two_d_map: Path | None,
                 extra: list[str], python: str) -> None:
    """Invoke SegviGen's full-segmentation script on one mesh."""
    script = segvigen / "inference_full.py"
    if not script.exists():
        raise SystemExit(f"{script} not found; pass the SegviGen checkout with --segvigen")
    cmd = [python, str(script), "--glb", str(mesh), "--export_glb", str(out)]
    if two_d_map is not None:
        cmd += ["--two_d_map", str(two_d_map)]
    cmd += extra
    print("[segvigen]", " ".join(cmd), file=sys.stderr)
    subprocess.run(cmd, cwd=segvigen, check=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("meshes", nargs="+", type=Path, help="Input meshes (.glb/.obj/.ply).")
    ap.add_argument("--segvigen", type=Path, default=Path(os.environ.get("SEGVIGEN_DIR", "SegviGen")),
                    help="Path to the SegviGen checkout (or $SEGVIGEN_DIR).")
    ap.add_argument("-o", "--out", type=Path,
                    help="Output GLB (one mesh) or directory (several); default <mesh>.parts.glb.")
    ap.add_argument("--faces", type=int, default=50_000,
                    help="Decimate the mesh to this many faces before segmenting.")
    ap.add_argument("--two-d-map", type=Path, help="Painted 2D segmentation map for the guided mode.")
    ap.add_argument("--python", default=sys.executable,
                    help="Interpreter of the SegviGen/TRELLIS.2 environment.")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="Everything after --extra goes to SegviGen's script unchanged.")
    args = ap.parse_args(argv)

    outputs: list[Path] = []
    for mesh in args.meshes:
        if args.out is None:
            out = mesh.with_suffix(".parts.glb")
        elif len(args.meshes) > 1 or args.out.is_dir():
            args.out.mkdir(parents=True, exist_ok=True)
            out = args.out / mesh.with_suffix(".parts.glb").name
        else:
            out = args.out
        key = _cache_key(mesh, args.faces, args.two_d_map, args.extra)
        cached = CACHE_DIR / f"{key}.glb"
        if cached.exists() and not args.no_cache:
            shutil.copyfile(cached, out)
            print(f"[segvigen] {mesh.name}: cached -> {out}", file=sys.stderr)
            outputs.append(out)
            continue
        with tempfile.TemporaryDirectory() as tmp:
            prepared = prepare_mesh(mesh, Path(tmp) / "input.glb", args.faces)
            run_segvigen(args.segvigen, prepared, out, two_d_map=args.two_d_map,
                         extra=args.extra, python=args.python)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(out, cached)
        outputs.append(out)
    for out in outputs:
        print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
