"""Hunyuan3D-2 multi-view backend.

The `-mv` variant takes 4 canonical views (front / back / left / right) and
produces a mesh with substantially better back-side geometry than the single-
image DiT can guess. Falls back to the single-image path if fewer than 2 views
are supplied.

Model: `tencent/Hunyuan3D-2mv` / subfolder `hunyuan3d-dit-v2-mv`. Weights are
CC-BY-NC via the Tencent Hunyuan Community License.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image

from ..voxelize import VoxelGrid, clean_mesh, project_photo_colors, voxelize


HUNYUAN_ROOT = Path(
    os.environ.get(
        "HUNYUAN3D_ROOT",
        str(Path(__file__).resolve().parents[3] / "vendor" / "Hunyuan3D-2"),
    )
)

REPO = os.environ.get("HUNYUAN3D_MV_REPO", "tencent/Hunyuan3D-2mv")
SUBFOLDER = os.environ.get("HUNYUAN3D_MV_SUBFOLDER", "hunyuan3d-dit-v2-mv")


def _import_hunyuan3d():
    if not HUNYUAN_ROOT.exists():
        raise ImportError(
            f"Hunyuan3D-2 checkout not found at {HUNYUAN_ROOT}. "
            "Clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git into vendor/."
        )
    if str(HUNYUAN_ROOT) not in sys.path:
        sys.path.insert(0, str(HUNYUAN_ROOT))
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    return Hunyuan3DDiTFlowMatchingPipeline


_PIPE = None
_PIPE_DEVICE: str | None = None


def _get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_pipeline():
    global _PIPE, _PIPE_DEVICE
    if _PIPE is not None:
        return _PIPE
    Pipeline = _import_hunyuan3d()
    warnings.filterwarnings("ignore", category=FutureWarning)
    _PIPE_DEVICE = _get_device()
    _PIPE = Pipeline.from_pretrained(
        REPO,
        subfolder=SUBFOLDER,
        torch_dtype=torch.float32,
        device=_PIPE_DEVICE,
    )
    return _PIPE


def _load_view_images(view_paths: dict[str, Path]) -> dict[str, Image.Image]:
    """Load each canonical view as RGBA. Missing views are dropped; the mv
    pipeline handles a subset (it works best with all four)."""
    result: dict[str, Image.Image] = {}
    for name, path in view_paths.items():
        if path is not None:
            result[name] = Image.open(path).convert("RGBA")
    return result


def infer_mesh(view_paths: dict[str, Path], *, num_inference_steps: int = 50,
               octree_resolution: int = 384, guidance_scale: float = 5.0):
    """Run Hunyuan3D-2mv on a dict of canonical views. Keys are 'front',
    'back', 'left', 'right' (any subset accepted, but at least one)."""
    images = _load_view_images(view_paths)
    if not images:
        raise ValueError("Provide at least one canonical view.")
    pipe = _load_pipeline()
    with torch.no_grad():
        meshes = pipe(
            image=images,
            num_inference_steps=num_inference_steps,
            octree_resolution=octree_resolution,
            guidance_scale=guidance_scale,
        )
    return meshes[0]


def build(image_paths: Sequence[str | Path], *,
          max_studs: int = 40,
          num_inference_steps: int = 50,
          octree_resolution: int = 384,
          guidance_scale: float = 5.0,
          project_colors: bool = True,
          forward_axis: str = "-z",
          flip_v: bool = False,
          save_mesh_path: str | Path | None = None,
          save_colored_mesh_path: str | Path | None = None,
          cleanup: bool = True,
          cleanup_smooth_iterations: int = 5,
          **_kwargs) -> VoxelGrid:
    """Reconstruct from 1-4 canonical views. Maps by input order:
    [front, back, left, right] — pass the paths in that order."""
    if not image_paths:
        raise ValueError("No images provided")
    paths = [Path(p) for p in image_paths]
    canonical_names = ["front", "back", "left", "right"]
    view_paths = {name: path for name, path in zip(canonical_names, paths)}

    mesh = infer_mesh(view_paths,
                       num_inference_steps=num_inference_steps,
                       octree_resolution=octree_resolution,
                       guidance_scale=guidance_scale)
    if cleanup:
        mesh = clean_mesh(mesh, smooth_iterations=cleanup_smooth_iterations)
    if save_mesh_path:
        Path(save_mesh_path).parent.mkdir(parents=True, exist_ok=True)
        mesh.export(save_mesh_path)
    if project_colors:
        # Use the front view as the primary color source.
        mesh = project_photo_colors(mesh, paths[0],
                                     forward_axis=forward_axis, flip_v=flip_v)
    if save_colored_mesh_path:
        Path(save_colored_mesh_path).parent.mkdir(parents=True, exist_ok=True)
        mesh.export(save_colored_mesh_path)

    grid = voxelize(mesh, max_studs=max_studs)
    grid.metadata.setdefault("method", "hunyuan3d_mv")
    grid.metadata["num_input_views"] = len(view_paths)
    grid.metadata["view_order"] = list(view_paths.keys())
    return grid
