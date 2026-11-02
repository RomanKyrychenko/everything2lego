"""SAMesh (zero-shot mesh part segmentation) on this Mac: tiny SAM2, default (CGL) OpenGL instead of EGL.

    source /Volumes/PRO-G40/lego_data/samesh-env.sh && $PY tools/samesh_run.py mesh.glb --device cpu [--out DIR]

Runs in SAMesh's own venv, not the repo's: samesh-env.sh sets $PY (/Volumes/PRO-G40/lego_data/samesh-venv) and the
PYTHONPATH to SAMesh's checkout (/Volumes/PRO-G40/lego_data/samesh/src and its segment-anything-2). SAMesh has no
licence file, so none of its code is in this repo; this runner only imports it. MPS crashes: use --device cpu
(~13 min on the watermill). tools/prepare_parts.py calls this and caches the result per mesh.

Writes <DIR>/<stem>_segmented.glb and <stem>_face2label.json (face index -> part label); DIR defaults to
/Volumes/PRO-G40/lego_data/analysis/samesh/<stem>.
"""
import argparse
import copy
import json
import os
import time
from pathlib import Path

os.environ.pop("PYOPENGL_PLATFORM", None)                      # macOS: pyrender's offscreen needs it absent (pyglet)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")      # ops MPS lacks run on the CPU
import OpenGL.GL                                                # noqa: E402,F401  platform picked now (darwin), before
import pyrender                                                 # noqa: E402,F401  SAMesh's renderer sets egl

import trimesh.visual                                            # noqa: E402
_interp = trimesh.visual.interpolate                            # trimesh 5 knows 4 colour maps by name; SAMesh asks for jet


def _interpolate(values, color_map=None, *a, **k):
    if isinstance(color_map, str) and color_map not in ("magma", "inferno", "plasma", "viridis"):
        import matplotlib
        color_map = matplotlib.colormaps[color_map]
    return _interp(values, color_map, *a, **k)


trimesh.visual.interpolate = _interpolate
trimesh.visual.color.interpolate = _interpolate
from omegaconf import OmegaConf                                 # noqa: E402
from samesh.data.loaders import read_mesh, remove_texture        # noqa: E402
from samesh.models.sam_mesh import SamModelMesh, colormap_faces_mesh  # noqa: E402

ROOT = Path("/Volumes/PRO-G40/lego_data/samesh")
OUT = Path("/Volumes/PRO-G40/lego_data/analysis/samesh")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh", type=Path)
    ap.add_argument("--device", default="cpu")                 # MPS crashes
    ap.add_argument("--points-per-side", type=int, default=32)
    ap.add_argument("--out", type=Path, default=None, help="output directory (default OUT/<stem>)")
    a = ap.parse_args()
    config = OmegaConf.load(ROOT / "configs" / "mesh_segmentation.yaml")
    config.sam.sam.checkpoint = str(ROOT / "checkpoints" / "sam2_hiera_tiny.pt")
    config.sam.sam.model_config = "sam2_hiera_t.yaml"
    config.sam.sam.engine_config.points_per_side = a.points_per_side
    config = copy.deepcopy(config)
    config.output = a.out or OUT / a.mesh.stem
    config.cache = config.output / "cache" if a.out else OUT / "cache" / a.mesh.stem
    t = time.time()
    os.environ.pop("PYOPENGL_PLATFORM", None)                   # SAMesh's import set egl; pyrender reads it per renderer
    model = SamModelMesh(config, device=a.device)
    tmesh = remove_texture(read_mesh(a.mesh, norm=True), visual_kind="vertex")
    face2label, _ = model(tmesh, visualize_path=str(config.output / "renders"))
    os.makedirs(config.output, exist_ok=True)
    colormap_faces_mesh(tmesh, face2label).export(str(config.output / f"{a.mesh.stem}_segmented.glb"))
    json.dump({int(k): int(v) for k, v in face2label.items()}, open(config.output / f"{a.mesh.stem}_face2label.json", "w"))
    print(f"{len(set(face2label.values()))} parts over {len(face2label)} faces in {time.time() - t:.0f} s -> {config.output}")


if __name__ == "__main__":
    main()
