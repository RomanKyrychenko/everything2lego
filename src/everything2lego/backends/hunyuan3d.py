"""Hunyuan3D-2 image -> mesh backend.

Tencent Hunyuan3D-2 reconstructs a textured mesh from a single (or a small
set of) image(s). Runs on Apple Silicon via MPS in the arm64 venv.

We use the *mini* shape-generator variant to keep disk and memory bounded:
`hunyuan3d-dit-v2-mini` (~1.5 GB fp16 weights, 559 M params). Colours are
sampled from the input image by our own `voxelize` step; the heavier
texture-painter (`Hunyuan3DPaintPipeline`) is skipped.

Weights are CC-BY-NC via the Tencent Hunyuan Community License. Personal
builds are fine; commercial use needs a separate agreement.
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

DEFAULT_REPO = os.environ.get("HUNYUAN3D_REPO", "tencent/Hunyuan3D-2")
DEFAULT_SUBFOLDER = os.environ.get("HUNYUAN3D_SUBFOLDER", "hunyuan3d-dit-v2-0")


def _import_hunyuan3d():
    if not HUNYUAN_ROOT.exists():
        raise ImportError(
            f"Hunyuan3D-2 checkout not found at {HUNYUAN_ROOT}. "
            "Clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git into "
            "vendor/ or set HUNYUAN3D_ROOT."
        )
    if str(HUNYUAN_ROOT) not in sys.path:
        sys.path.insert(0, str(HUNYUAN_ROOT))
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline  # noqa
    try:
        from hy3dgen.rembg import BackgroundRemover  # noqa
    except Exception:
        BackgroundRemover = None
    return Hunyuan3DDiTFlowMatchingPipeline, BackgroundRemover


_PIPE = None


def _get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_pipeline():
    global _PIPE
    if _PIPE is not None:
        return _PIPE
    Hunyuan3DDiTFlowMatchingPipeline, _ = _import_hunyuan3d()
    warnings.filterwarnings("ignore", category=FutureWarning)
    _PIPE = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        DEFAULT_REPO,
        subfolder=DEFAULT_SUBFOLDER,
        torch_dtype=torch.float32,
        device=_get_device(),
    )
    return _PIPE


def _prepare_image(path: Path, *, use_rembg: bool = False) -> Image.Image:
    """Load an image as RGBA. Optionally run rembg (downloads a 176 MB ONNX
    model on first use) — skip it for catalog / studio shots on a plain
    background, since Hunyuan3D can handle a white-background input directly."""
    img = Image.open(path).convert("RGB")
    if not use_rembg:
        return img.convert("RGBA")
    _, BackgroundRemover = _import_hunyuan3d()
    if BackgroundRemover is None:
        return img.convert("RGBA")
    remover = BackgroundRemover()
    return remover(img)


def infer_mesh(image_path: Path, *, num_inference_steps: int = 30,
               octree_resolution: int = 256, mc_resolution: int | None = None,
               guidance_scale: float = 5.0, use_rembg: bool = False):
    """Run Hunyuan3D-2 mini on one image, return a trimesh.Trimesh."""
    if mc_resolution is not None:
        octree_resolution = mc_resolution
    pipe = _load_pipeline()
    img = _prepare_image(Path(image_path), use_rembg=use_rembg)
    with torch.no_grad():
        meshes = pipe(
            image=img,
            num_inference_steps=num_inference_steps,
            octree_resolution=octree_resolution,
            guidance_scale=guidance_scale,
        )
    return meshes[0]


def build(image_paths: Sequence[str | Path], *,
          max_studs: int = 40,
          num_inference_steps: int = 30,
          octree_resolution: int = 256,
          guidance_scale: float = 5.0,
          front_view: int = 0,
          use_rembg: bool = False,
          project_colors: bool = True,
          forward_axis: str = "-z",
          flip_v: bool = False,
          save_mesh_path: str | Path | None = None,
          save_colored_mesh_path: str | Path | None = None,
          cleanup: bool = True,
          cleanup_smooth_iterations: int = 5,
          **_kwargs) -> VoxelGrid:
    """Reconstruct a mesh from one image (`front_view`) and voxelize to LEGO scale.

    Multi-view Hunyuan3D-2mv exists but requires four canonical views (front,
    back, left, right); we default to the flexible single-image variant here.
    """
    if not image_paths:
        raise ValueError("No images provided")
    paths = [Path(p) for p in image_paths]
    mesh = infer_mesh(paths[front_view],
                       num_inference_steps=num_inference_steps,
                       octree_resolution=octree_resolution,
                       guidance_scale=guidance_scale,
                       use_rembg=use_rembg)
    if cleanup:
        mesh = clean_mesh(mesh, smooth_iterations=cleanup_smooth_iterations)
    saved_mesh: str | None = None
    if save_mesh_path:
        save_mesh_path = Path(save_mesh_path)
        save_mesh_path.parent.mkdir(parents=True, exist_ok=True)
        # STL discards color; save the raw geometry from Hunyuan directly.
        mesh.export(save_mesh_path)
        saved_mesh = str(save_mesh_path)

    if project_colors:
        mesh = project_photo_colors(mesh, paths[front_view],
                                     forward_axis=forward_axis, flip_v=flip_v)

    saved_colored: str | None = None
    if save_colored_mesh_path:
        save_colored_mesh_path = Path(save_colored_mesh_path)
        save_colored_mesh_path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(save_colored_mesh_path)
        saved_colored = str(save_colored_mesh_path)

    grid = voxelize(mesh, max_studs=max_studs)
    grid.metadata.setdefault("method", "hunyuan3d")
    grid.metadata["source_image"] = str(paths[front_view])
    grid.metadata["front_view"] = front_view
    grid.metadata["colors_projected"] = bool(project_colors)
    if saved_mesh:
        grid.metadata["saved_mesh_stl"] = saved_mesh
    if saved_colored:
        grid.metadata["saved_mesh_glb"] = saved_colored
    return grid
