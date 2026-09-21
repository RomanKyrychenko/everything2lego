"""Hunyuan3D-2.1 image → mesh backend.

Newer than `hunyuan3d.py`.  Uses `tencent/Hunyuan3D-2.1`, whose DiT is
`hunyuan3d-dit-v2-1` (fp16 .ckpt) and has an updated ShapeVAE
`hunyuan3d-vae-v2-1`.  Ships in the same MPS/Apple-Silicon-compatible way
as the v2-0 backend, with `PYTORCH_ENABLE_MPS_FALLBACK=1` for the
`_upsample_bilinear2d_aa` op.

Vendor code lives in `vendor/Hunyuan3D-2.1/hy3dshape/hy3dshape/` (the
`hy3dshape` package moved out of `hy3dgen.shapegen`).  We only wire the
shape-generation half; the texture-paint pipeline (hy3dpaint) is a
separate, heavier addition and is not required for producing a colored
mesh — the LEGO pipeline resamples colors from the mesh's vertex_colors
via projected photo colors in `voxelize.project_photo_colors`, same as
the v2-0 path.
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
        "HUNYUAN3D_V21_ROOT",
        str(Path(__file__).resolve().parents[3] / "vendor" / "Hunyuan3D-2.1"),
    )
)

DEFAULT_REPO = os.environ.get("HUNYUAN3D_V21_REPO", "tencent/Hunyuan3D-2.1")
DEFAULT_SUBFOLDER = os.environ.get("HUNYUAN3D_V21_SUBFOLDER",
                                      "hunyuan3d-dit-v2-1")


def _import_hy3dshape():
    if not HUNYUAN_ROOT.exists():
        raise ImportError(
            f"Hunyuan3D-2.1 checkout not found at {HUNYUAN_ROOT}. "
            "Clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1 into "
            "vendor/ or set HUNYUAN3D_V21_ROOT."
        )
    # torch<2.4 lacks torch.nn.RMSNorm — the 2.1 DiT references it at
    # module import.  Provide a small backport when needed so we don't
    # have to upgrade torch (which would cascade into diffusers/numpy).
    _patch_rmsnorm_shim()
    hy3dshape_pkg = HUNYUAN_ROOT / "hy3dshape" / "hy3dshape"
    hy3dshape_pkg_parent = str(hy3dshape_pkg.parent)
    if hy3dshape_pkg_parent not in sys.path:
        sys.path.insert(0, hy3dshape_pkg_parent)
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline  # noqa
    return Hunyuan3DDiTFlowMatchingPipeline


def _patch_rmsnorm_shim() -> None:
    import torch.nn as nn
    if hasattr(nn, "RMSNorm"):
        return
    import torch

    class _RMSNorm(nn.Module):
        """Minimal RMSNorm backport for torch < 2.4 (matches PyTorch's."""

        def __init__(self, normalized_shape, eps: float = 1e-6,
                      elementwise_affine: bool = True, device=None, dtype=None):
            super().__init__()
            if isinstance(normalized_shape, int):
                normalized_shape = (normalized_shape,)
            self.normalized_shape = tuple(normalized_shape)
            self.eps = eps
            self.elementwise_affine = elementwise_affine
            if elementwise_affine:
                self.weight = nn.Parameter(torch.ones(
                    *self.normalized_shape, device=device, dtype=dtype))
            else:
                self.register_parameter("weight", None)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            var = x.pow(2).mean(dim=-1, keepdim=True)
            x = x * torch.rsqrt(var + self.eps)
            if self.weight is not None:
                x = x * self.weight
            return x

    nn.RMSNorm = _RMSNorm  # type: ignore[attr-defined]


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
    Pipeline = _import_hy3dshape()
    warnings.filterwarnings("ignore", category=FutureWarning)
    device = _get_device()
    # fp16 to fit in RAM on Apple Silicon (the 2.1 DiT is bigger than
    # 2.0 and hits OOM at float32 on 16-24 GB machines).
    _PIPE = Pipeline.from_pretrained(
        DEFAULT_REPO,
        subfolder=DEFAULT_SUBFOLDER,
        use_safetensors=False,      # 2.1 ships .ckpt only
        dtype=torch.float16,
        device=device,
    )
    return _PIPE


def _prepare_image(path: Path) -> Image.Image:
    """Load as RGBA. 2.1 handles both white-background studio shots and
    photos with unremoved backgrounds; the shapegen's own image processor
    normalizes on the fly.  We do NOT wire rembg here (llvmlite build
    fails on macOS 27 arm64).  Feed a white-background photo where
    possible."""
    return Image.open(path).convert("RGBA")


def infer_mesh(image_path: Path, *, num_inference_steps: int = 50,
               octree_resolution: int = 384, guidance_scale: float = 5.0):
    """Run Hunyuan3D-2.1 on one image; return a trimesh.Trimesh."""
    pipe = _load_pipeline()
    img = _prepare_image(image_path)
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
          num_inference_steps: int = 50,
          octree_resolution: int = 384,
          guidance_scale: float = 5.0,
          front_view: int = 0,
          project_colors: bool = True,
          forward_axis: str = "-z",
          flip_v: bool = False,
          save_mesh_path: Path | None = None,
          save_colored_mesh_path: Path | None = None,
          ) -> VoxelGrid:
    """image → mesh → voxel grid, using Hunyuan3D-2.1."""
    images = [Path(p) for p in image_paths]
    if not images:
        raise ValueError("Provide at least one image.")
    front_image = images[front_view] if front_view < len(images) else images[0]
    mesh = infer_mesh(front_image,
                        num_inference_steps=num_inference_steps,
                        octree_resolution=octree_resolution,
                        guidance_scale=guidance_scale)
    mesh = clean_mesh(mesh)
    if save_mesh_path is not None:
        Path(save_mesh_path).parent.mkdir(parents=True, exist_ok=True)
        mesh.export(str(save_mesh_path))
    if project_colors:
        mesh = project_photo_colors(mesh, front_image,
                                       forward_axis=forward_axis,
                                       flip_v=flip_v)
    if save_colored_mesh_path is not None:
        Path(save_colored_mesh_path).parent.mkdir(parents=True, exist_ok=True)
        mesh.export(str(save_colored_mesh_path))
    grid = voxelize(mesh, max_studs=max_studs)
    grid.metadata.setdefault("method", "hunyuan3d_v21")
    grid.metadata["source_repo"] = DEFAULT_REPO
    grid.metadata["source_subfolder"] = DEFAULT_SUBFOLDER
    return grid
