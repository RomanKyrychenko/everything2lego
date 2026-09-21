"""DUSt3R backend: unstructured multi-view photos -> colored point cloud -> LEGO.

Suited to real-world subjects like houses that can't be captured on a
turntable. DUSt3R (Naver Labs) jointly estimates camera poses and dense
per-pixel 3D points from any set of overlapping images.

Model weights are CC-BY-NC (non-commercial). Fine for personal use.
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

from ..voxelize import VoxelGrid, voxelize_pointcloud


DUST3R_ROOT = Path(
    os.environ.get(
        "DUST3R_ROOT",
        str(Path(__file__).resolve().parents[3] / "vendor" / "dust3r"),
    )
)


def _import_dust3r():
    if not DUST3R_ROOT.exists():
        raise ImportError(
            f"DUSt3R checkout not found at {DUST3R_ROOT}. "
            "Clone https://github.com/naver/dust3r.git into vendor/ "
            "(with `git submodule update --init --recursive`) or set DUST3R_ROOT."
        )
    for p in (str(DUST3R_ROOT), str(DUST3R_ROOT / "croco")):
        if p not in sys.path:
            sys.path.insert(0, p)
    from dust3r.model import AsymmetricCroCo3DStereo
    from dust3r.inference import inference
    from dust3r.utils.image import load_images
    from dust3r.image_pairs import make_pairs
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
    return {
        "AsymmetricCroCo3DStereo": AsymmetricCroCo3DStereo,
        "inference": inference,
        "load_images": load_images,
        "make_pairs": make_pairs,
        "global_aligner": global_aligner,
        "GlobalAlignerMode": GlobalAlignerMode,
    }


_MODEL = None
_MODEL_DEVICE: str | None = None


def _get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_model():
    global _MODEL, _MODEL_DEVICE
    if _MODEL is not None:
        return _MODEL
    warnings.filterwarnings("ignore", category=FutureWarning)
    d = _import_dust3r()
    _MODEL_DEVICE = _get_device()
    model = d["AsymmetricCroCo3DStereo"].from_pretrained(
        "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
    )
    model = model.to(_MODEL_DEVICE)
    model.eval()
    _MODEL = model
    return model


def infer_pointcloud(image_paths: Sequence[str | Path], *,
                     image_size: int = 512, niter: int = 300,
                     min_conf: float = 3.0
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Run DUSt3R + global alignment on a set of unstructured photos.

    Returns (points, colors) where points is (N, 3) float32 in DUSt3R's world
    frame (metric-ambiguous) and colors is (N, 3) uint8. Only points whose
    per-pixel confidence exceeds `min_conf` are returned.
    """
    paths = [str(Path(p).resolve()) for p in image_paths]
    if len(paths) < 2:
        raise ValueError("DUSt3R needs at least two images")

    d = _import_dust3r()
    model = _load_model()
    device = _MODEL_DEVICE

    images = d["load_images"](paths, size=image_size)
    # All-pairs graph. For >4 images, "swin-2" is faster and often adequate.
    scene_graph = "complete" if len(paths) <= 4 else "swin-2"
    pairs = d["make_pairs"](images, scene_graph=scene_graph, prefilter=None, symmetrize=True)
    output = d["inference"](pairs, model, device, batch_size=1)

    mode = (d["GlobalAlignerMode"].PointCloudOptimizer if len(paths) > 2
            else d["GlobalAlignerMode"].PairViewer)
    scene = d["global_aligner"](output, device=device, mode=mode)
    if mode == d["GlobalAlignerMode"].PointCloudOptimizer:
        scene.compute_global_alignment(init="mst", niter=niter, schedule="cosine", lr=0.01)

    pts3d_list = scene.get_pts3d()          # list of (H, W, 3) tensors
    imgs_list = scene.imgs                   # list of (H, W, 3) 0..1 arrays
    conf_list = scene.get_conf()             # list of (H, W) confidences

    all_pts: list[np.ndarray] = []
    all_rgb: list[np.ndarray] = []
    for pts, img, conf in zip(pts3d_list, imgs_list, conf_list):
        pts_np = pts.detach().cpu().numpy().reshape(-1, 3)
        rgb_np = (np.asarray(img).reshape(-1, 3) * 255).clip(0, 255).astype(np.uint8)
        conf_np = conf.detach().cpu().numpy().reshape(-1) if hasattr(conf, "detach") \
                  else np.asarray(conf).reshape(-1)
        keep = conf_np > min_conf
        all_pts.append(pts_np[keep])
        all_rgb.append(rgb_np[keep])

    if not any(len(a) for a in all_pts):
        raise RuntimeError(
            "DUSt3R produced no points above the confidence threshold. "
            "Lower --min-conf or supply photos with more overlap."
        )
    return (np.concatenate(all_pts, axis=0).astype(np.float32),
            np.concatenate(all_rgb, axis=0))


def _orient_up(pts: np.ndarray) -> np.ndarray:
    """DUSt3R doesn't guarantee which axis is 'up'. Assume the axis with the
    smallest interquartile variance is the vertical axis (buildings and
    ground are tall on one axis and spread on the others). Rotate so up = +Y."""
    q75 = np.percentile(pts, 75, axis=0)
    q25 = np.percentile(pts, 25, axis=0)
    iqr = q75 - q25
    up_axis = int(np.argmin(iqr))
    if up_axis == 1:
        return pts
    axes = [0, 1, 2]
    axes[1], axes[up_axis] = axes[up_axis], axes[1]
    return pts[:, axes]


def _sample_background_colors(image_paths: Sequence[str | Path],
                               patch_size: int = 12) -> np.ndarray:
    """Median RGB of each corner of each input image. These are the sky, grass
    and neighbouring-building colors that DUSt3R will happily reconstruct as
    part of the subject unless we filter them out."""
    backgrounds: list[np.ndarray] = []
    for p in image_paths:
        img = np.asarray(Image.open(p).convert("RGB"))
        h, w = img.shape[:2]
        s = min(patch_size, h // 4, w // 4)
        for cy, cx in ((0, 0), (0, w - s), (h - s, 0), (h - s, w - s)):
            patch = img[cy:cy + s, cx:cx + s].reshape(-1, 3)
            backgrounds.append(np.median(patch, axis=0))
    return np.asarray(backgrounds, dtype=np.int16)


def _drop_background(pts: np.ndarray, rgb: np.ndarray,
                      backgrounds: np.ndarray, tolerance: int = 30
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Drop points whose color is within `tolerance` (Manhattan RGB distance)
    of ANY sampled background color."""
    if len(backgrounds) == 0:
        return pts, rgb
    # (N, 1, 3) - (1, M, 3) -> (N, M) Manhattan distance
    diff = np.abs(rgb.astype(np.int16)[:, None, :] - backgrounds[None, :, :]).sum(axis=-1)
    keep = diff.min(axis=1) > tolerance
    return pts[keep], rgb[keep]


def _drop_outliers(pts: np.ndarray, rgb: np.ndarray, *, k: float = 3.0
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Drop points further than k*MAD from the median along any axis. Using
    the median absolute deviation instead of std makes this robust to the
    handful of catastrophically-wrong predictions DUSt3R sometimes emits."""
    med = np.median(pts, axis=0)
    mad = np.median(np.abs(pts - med), axis=0)
    mad = np.where(mad > 0, mad, 1e-6)
    z = np.abs(pts - med) / mad
    keep = (z < k).all(axis=1)
    return pts[keep], rgb[keep]


def _snap_ground_plane(pts: np.ndarray, rgb: np.ndarray, *,
                        percentile: float = 2.0,
                        below_ground_tolerance: float = 0.0
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Assumes pts are already y-up. Shifts so the `percentile`-th y value
    sits at 0 and discards points below `-below_ground_tolerance`. Cheaper
    than a full RANSAC plane fit and adequate when the camera is roughly
    level. Assumes y is index 1."""
    ground_y = np.percentile(pts[:, 1], percentile)
    pts = pts.copy()
    pts[:, 1] -= ground_y
    keep = pts[:, 1] >= -below_ground_tolerance
    return pts[keep], rgb[keep]


def build(image_paths: Sequence[str | Path], *,
          max_studs: int = 40,
          min_conf: float = 3.0,
          image_size: int = 512,
          niter: int = 300,
          orient: bool = True,
          drop_background: bool = True,
          background_tolerance: int = 30,
          drop_outliers: bool = True,
          outlier_k: float = 3.0,
          snap_ground: bool = True,
          ground_percentile: float = 2.0,
          **_kwargs) -> VoxelGrid:
    """Reconstruct from unstructured photos into a LEGO-scale voxel grid.

    Filters (all on by default; tune or disable per the CLI flags):
      1. background rejection - drop points whose color matches any input
         image corner (sky, foliage, neighbouring buildings).
      2. MAD-outlier rejection - drop points too far from the median.
      3. ground snap - shift so the bottom of the cloud sits at y = 0 and
         drop below-ground artefacts.
    """
    pts, rgb = infer_pointcloud(image_paths, image_size=image_size,
                                 niter=niter, min_conf=min_conf)
    n_raw = len(pts)

    if drop_background:
        bgs = _sample_background_colors(image_paths)
        pts, rgb = _drop_background(pts, rgb, bgs,
                                     tolerance=background_tolerance)
    n_bg = len(pts)

    if drop_outliers and len(pts) > 100:
        pts, rgb = _drop_outliers(pts, rgb, k=outlier_k)
    n_out = len(pts)

    if orient:
        pts = _orient_up(pts)

    if snap_ground and len(pts) > 100:
        pts, rgb = _snap_ground_plane(pts, rgb, percentile=ground_percentile)

    if not len(pts):
        raise RuntimeError(
            "All points were filtered out. Loosen --background-tolerance, "
            "--outlier-k, or disable filters with --raw-points."
        )

    grid = voxelize_pointcloud(pts, rgb, max_studs=max_studs)
    grid.metadata.setdefault("method", "dust3r")
    grid.metadata["num_input_images"] = len(image_paths)
    grid.metadata["num_points_raw"] = n_raw
    grid.metadata["num_points_after_background"] = n_bg
    grid.metadata["num_points_after_outlier"] = n_out
    grid.metadata["num_points_final"] = int(len(pts))
    return grid
