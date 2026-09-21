"""TripoSR image -> mesh backend.

TripoSR (Stability AI + VAST) reconstructs a textured mesh from a single image.
It runs on Apple Silicon via MPS in a native arm64 Python venv.

The repo is cloned into `vendor/TripoSR` by convention; override with
`TRIPOSR_ROOT` if you keep it elsewhere.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image

from ..voxelize import VoxelGrid, voxelize


TRIPOSR_ROOT = Path(
    os.environ.get(
        "TRIPOSR_ROOT",
        str(Path(__file__).resolve().parents[3] / "vendor" / "TripoSR"),
    )
)


def _import_tsr():
    if not TRIPOSR_ROOT.exists():
        raise ImportError(
            f"TripoSR checkout not found at {TRIPOSR_ROOT}. "
            "Clone https://github.com/VAST-AI-Research/TripoSR.git into vendor/ "
            "or set TRIPOSR_ROOT."
        )
    if str(TRIPOSR_ROOT) not in sys.path:
        sys.path.insert(0, str(TRIPOSR_ROOT))
    from tsr.system import TSR
    from tsr.utils import resize_foreground

    return TSR, resize_foreground


_MODEL = None
_MODEL_DEVICE: str | None = None


def _get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def _patch_marching_cubes_for_mps() -> None:
    """torchmcubes only accepts CPU tensors, and TripoSR's built-in
    try/except catches AttributeError (CUDA-not-compiled) but not MPS's
    RuntimeError. Force the tensor to CPU before the call."""
    _import_tsr()
    from tsr.models.isosurface import MarchingCubeHelper
    from torchmcubes import marching_cubes

    def forward(self, level):
        level = -level.view(self.resolution, self.resolution, self.resolution)
        v_pos, t_pos_idx = marching_cubes(level.detach().cpu().contiguous(), 0.0)
        v_pos = v_pos[..., [2, 1, 0]]
        v_pos = v_pos / (self.resolution - 1.0)
        return v_pos.to(level.device), t_pos_idx.to(level.device)

    MarchingCubeHelper.forward = forward


def _load_model():
    global _MODEL, _MODEL_DEVICE
    if _MODEL is not None:
        return _MODEL
    TSR, _ = _import_tsr()
    _patch_marching_cubes_for_mps()
    warnings.filterwarnings("ignore", category=FutureWarning)
    model = TSR.from_pretrained(
        "stabilityai/TripoSR",
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    model.renderer.set_chunk_size(8192)
    _MODEL_DEVICE = _get_device()
    model.to(_MODEL_DEVICE)
    _MODEL = model
    return model


def _preprocess(image_path: Path, foreground_ratio: float = 0.85) -> Image.Image:
    """Prepare an image for TripoSR: alpha-mask foreground, resize, composite
    on the same 0.5-gray background TripoSR was trained on."""
    from ..carve import _foreground_mask

    _, resize_foreground = _import_tsr()

    pil = Image.open(image_path).convert("RGB")
    img = np.asarray(pil)
    mask = _foreground_mask(img, background=None, tolerance=28)
    if mask.sum() < 50:
        raise ValueError(f"{image_path}: could not detect foreground")
    rgba = np.dstack([img, (mask.astype(np.uint8) * 255)])
    resized = resize_foreground(Image.fromarray(rgba, mode="RGBA"), foreground_ratio)
    arr = np.array(resized).astype(np.float32) / 255.0
    composited = arr[..., :3] * arr[..., 3:4] + (1 - arr[..., 3:4]) * 0.5
    return Image.fromarray((composited * 255).astype(np.uint8))


def infer_mesh(image_path: Path, *, mc_resolution: int = 128,
               foreground_ratio: float = 0.85):
    """Run TripoSR on one image, return a trimesh.Trimesh with vertex colors."""
    pil = _preprocess(image_path, foreground_ratio=foreground_ratio)
    model = _load_model()
    with torch.no_grad():
        scene_codes = model([pil], device=_MODEL_DEVICE)
    meshes = model.extract_mesh(scene_codes, has_vertex_color=True,
                                 resolution=mc_resolution)
    return meshes[0]


def build(image_paths: Sequence[str | Path], *, max_studs: int = 32,
          mc_resolution: int = 128, front_view: int = 0,
          **_kwargs) -> VoxelGrid:
    """Reconstruct a mesh from one image (the `front_view` index) and voxelize.

    Additional images are ignored here; combine with silhouette carving via
    the `hybrid` backend for multi-view refinement.
    """
    if not image_paths:
        raise ValueError("No images provided")
    paths = [Path(p) for p in image_paths]
    mesh = infer_mesh(paths[front_view], mc_resolution=mc_resolution)
    return voxelize(mesh, max_studs=max_studs)
