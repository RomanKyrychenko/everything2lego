"""One step from a mesh to the sampler's --parts and --semantic: SAMesh parts, per-cell part labels, CLIP labels.

    python tools/prepare_parts.py mesh.glb --studs 32 [--snap-geometry [--snap-pitches roof]] [--cache DIR]
        [--samesh-from DIR]        # seed the cache with an existing SAMesh output (<stem>_face2label.json, _segmented.glb)

Cached under DIR (default /Volumes/PRO-G40/lego_data/analysis/parts_cache/<sha1 of the mesh's bytes>/):
    samesh/face2label.json, samesh/segmented.glb   SAMesh (tools/samesh_run.py in SAMesh's own venv, CPU, ~13 min on
                                                   the watermill), once per mesh, whatever --studs
    parts_<N>[_snap-<pitches>].npz                 tools/part_labels.py at --studs N (its target, snapped or not)
    labels_<N>[_snap-<pitches>]_<labeller>.npz     tools/semantic_labels.py --samesh, same target (labeller: model,
                                                   render setting and thresholds, hashed)
Each file is written to a temporary name and renamed, so a cached file is a finished one; a re-run is instant.
sampler_demo --prepare-parts calls prepare() for its mesh and --studs.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
DATA = Path("/Volumes/PRO-G40/lego_data")
CACHE = DATA / "analysis" / "parts_cache"
SAMESH_ENV = DATA / "samesh-env.sh"
INSTALL_HINT = ("SAMesh's venv is missing: expected {env} setting $PY and PYTHONPATH (SAMesh checkout at "
                f"{DATA / 'samesh'} with its segment-anything-2 and checkpoints/sam2_hiera_tiny.pt, venv at "
                f"{DATA / 'samesh-venv'}). Install SAMesh (github.com/gtangg12/samesh) into its own venv there and "
                "write samesh-env.sh (export PYTHONPATH=.../samesh/src:.../samesh/third_party/segment-anything-2; "
                "PY=.../samesh-venv/bin/python), or pass --samesh-from with an existing SAMesh output.")


def mesh_sha1(path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _run(cmd: list[str], what: str, **kw) -> None:
    t = time.time()
    print(f"prepare_parts: {what} ...", flush=True)
    r = subprocess.run(cmd, **kw)
    if r.returncode:
        raise SystemExit(f"prepare_parts: {what} failed (exit {r.returncode}): {' '.join(map(str, cmd))}")
    print(f"prepare_parts: {what} done in {time.time() - t:.0f} s", flush=True)


def samesh(mesh: Path, cache: Path, seed_from: Path | None = None) -> tuple[Path, Path]:
    """(face2label.json, segmented.glb) for the mesh, from the cache, `seed_from` or a SAMesh run."""
    d = cache / "samesh"
    f2l, seg = d / "face2label.json", d / "segmented.glb"
    if f2l.exists() and seg.exists():
        return f2l, seg
    tmp = cache / f"samesh.tmp{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    if seed_from:
        src = [next(Path(seed_from).glob(f"*{s}"), None) for s in ("_face2label.json", "_segmented.glb")]
        if None in src:
            raise SystemExit(f"--samesh-from {seed_from}: no *_face2label.json / *_segmented.glb there")
        for s, dst in zip(src, ("face2label.json", "segmented.glb")):
            shutil.copy(s, tmp / dst)
        print(f"prepare_parts: SAMesh output copied from {seed_from}")
    else:
        py = None
        if SAMESH_ENV.exists():
            py = subprocess.run(["bash", "-c", f'source "{SAMESH_ENV}" && echo "$PY"'], capture_output=True,
                                text=True).stdout.strip()
        if not py or not Path(py).exists():
            shutil.rmtree(tmp, ignore_errors=True)
            raise SystemExit("prepare_parts: " + INSTALL_HINT.format(env=SAMESH_ENV))
        _run(["bash", "-c", f'source "{SAMESH_ENV}" && exec "$PY" "$@"', "samesh",
              str(TOOLS / "samesh_run.py"), str(mesh), "--device", "cpu", "--out", str(tmp)], "SAMesh (CPU)")
        os.replace(tmp / f"{mesh.stem}_face2label.json", tmp / "face2label.json")
        os.replace(tmp / f"{mesh.stem}_segmented.glb", tmp / "segmented.glb")
        shutil.rmtree(tmp / "cache", ignore_errors=True)               # SAMesh's own render cache: big, not needed again
    if d.exists():
        shutil.rmtree(d)
    os.replace(tmp, d)
    return f2l, seg


def labeller() -> str:
    """'_<hash>' of semantic_labels' default model, render setting, fitted thresholds and part prototypes: labels made
    by another are not reused."""
    import hashlib
    from semantic_labels import THRESHOLDS, parser, protos_file, setting
    key = setting(parser().parse_args(["-", "--out", "-"])) + (THRESHOLDS.read_text() if THRESHOLDS.exists() else "")
    key += hashlib.sha1(protos_file().read_bytes()).hexdigest() if protos_file().exists() else ""
    return "_" + hashlib.sha1(key.encode()).hexdigest()[:8]


def prepare(mesh, studs: int, cache=None, snap_pitches: str | None = None, samesh_from=None) -> tuple[Path, Path]:
    """(parts.npz, labels.npz) for the sampler's target of `mesh` at `studs` (snapped with these pitches, or not)."""
    mesh = Path(mesh).resolve()
    cache = Path(cache) if cache else CACHE / mesh_sha1(mesh)
    cache.mkdir(parents=True, exist_ok=True)
    f2l, seg = samesh(mesh, cache, samesh_from)
    tag = f"{studs}" + (f"_snap-{snap_pitches}" if snap_pitches else "")
    snap = ["--snap-geometry", "--snap-pitches", snap_pitches] if snap_pitches else []
    out = []
    for name, script, args in (("parts", "part_labels.py", [str(f2l), str(seg)]),
                               ("labels", "semantic_labels.py", ["--samesh", str(f2l), str(seg)])):
        p = cache / f"{name}_{tag}{labeller() if name == 'labels' else ''}.npz"
        if not p.exists():
            tmp = cache / f"{name}_{tag}.tmp{os.getpid()}.npz"
            _run([sys.executable, str(TOOLS / script), str(mesh), *args, "--studs", str(studs), *snap, "--out", str(tmp)],
                 f"{script} --studs {studs}{' ' + ' '.join(snap) if snap else ''}")
            os.replace(tmp, p)
        out.append(p)
    return out[0], out[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mesh")
    ap.add_argument("--studs", type=int, required=True)
    ap.add_argument("--cache", default=None, help=f"cache directory (default {CACHE}/<sha1 of the mesh>)")
    ap.add_argument("--snap-geometry", action="store_true", help="labels for sampler_demo --snap-geometry's target")
    ap.add_argument("--snap-pitches", default="all", choices=["all", "roof"])
    ap.add_argument("--samesh-from", default=None, help="seed the cache with this existing SAMesh output directory")
    a = ap.parse_args()
    parts, labels = prepare(a.mesh, a.studs, a.cache, a.snap_pitches if a.snap_geometry else None, a.samesh_from)
    print(f"parts  {parts}\nlabels {labels}")


if __name__ == "__main__":
    main()
