"""Depth-informed voxel carve: Depth Anything V2 reads a per-pixel
depth from the photo; we compare to the mesh's own depth at the fitted
view and carve voxels whose predicted depth disagrees with the
photo's.  Catches Hunyuan3D hallucinations that silhouette carving
(tools/segment_photo.py's `carve`) can't — a wall pushed too far
back, a tail that doesn't exist.

    python tools/depth_carve.py photo.png mesh.glb --out depth_carve.npz
        [--studs 20]  [--view LAT,LON]  [--tol 0.08]  [--sam path]

Runs under the Intel .venv (needs torch + transformers).  First call
downloads depth-anything/Depth-Anything-V2-Small-hf (~100 MB) to the
HF cache; subsequent calls are offline-cached.

shortcut: depth is fitted linearly to the mesh's own depth distribution
over the mesh's projected pixels, then voxels whose residual exceeds
`--tol` are flagged.  Pixels outside the biggest SAM mask (background)
are not used for the fit and the voxels there are not carved by this
step (segment_photo.py's silhouette carve handles them).  Upgrades:
non-linear align, use the mesh pixel density to re-weight the fit,
or combine with MapAnything for metric depth.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from segment_photo import _camera, fit_view, _sam_masks  # noqa: E402

DEFAULT_SAM = os.environ.get("SAM_CHECKPOINT", "/Volumes/PRO-G40/models/sam/sam_vit_b_01ec64.pth")


def depth_anything(photo: Image.Image) -> np.ndarray:
    """(H, W) float in [0, 1], larger = nearer the camera.  Uses the
    HuggingFace Depth Anything V2 Small checkpoint; cached after first run."""
    from transformers import pipeline
    pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
    out = pipe(photo)
    d = np.asarray(out["depth"], np.float32) / 255.0
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("photo", type=Path)
    ap.add_argument("mesh", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--studs", type=int, default=24)
    ap.add_argument("--view", default="auto")
    ap.add_argument("--tol", type=float, default=0.08,
                    help="normalized depth disagreement above which the voxel is carved")
    ap.add_argument("--sam", default=DEFAULT_SAM)
    a = ap.parse_args()

    photo = Image.open(a.photo).convert("RGB")
    mesh = trimesh.load(a.mesh, force="mesh", process=False)

    print(f"SAM masks on photo ({photo.size}) ...", flush=True)
    masks = _sam_masks(photo, a.sam)
    biggest_mask = None
    if masks:
        biggest_mask = np.asarray(max(masks, key=lambda m: m["area"])["segmentation"], bool)
    if a.view == "auto":
        if biggest_mask is None:
            raise SystemExit("SAM found no masks; pass --view LAT,LON")
        view, iou = fit_view(mesh, biggest_mask)
        print(f"fit view lat={view[0]:+.0f} lon={view[1]:+.0f} (IoU {iou:.2f})", flush=True)
    else:
        view = tuple(float(v) for v in a.view.split(","))

    print(f"Depth Anything on photo ...", flush=True)
    depth = depth_anything(photo)                                 # (Hp, Wp) in [0,1], larger = nearer
    Hp, Wp = depth.shape
    if biggest_mask is not None and biggest_mask.shape != (Hp, Wp):
        biggest_mask = np.asarray(Image.fromarray(biggest_mask).resize((Wp, Hp))) > 0

    print(f"voxelising mesh at {a.studs} studs ...", flush=True)
    from everything2lego.semantic_v2 import MeshTarget
    mt = MeshTarget.from_path(a.mesh, max_studs=a.studs, n_samples=40000, min_island_cells=8)
    occ = mt.voxel_grid.palette_idx >= 0
    T = np.linalg.inv(mt.mesh_to_voxel)
    ys, xs, zs = np.nonzero(occ)
    cell_centres_mesh = (T @ np.c_[np.stack([xs + 0.5, ys + 0.5, zs + 0.5], 1), np.ones(len(xs))].T).T[:, :3]

    project = _camera(*view)
    verts = project(mesh.vertices)
    vox_cam = project(cell_centres_mesh)
    vlo, vhi = verts[:, :2].min(0), verts[:, :2].max(0)
    scale = min(Wp / (vhi[0] - vlo[0] + 1e-6), Hp / (vhi[1] - vlo[1] + 1e-6)) * 0.95
    cx = Wp / 2 - 0.5 * (vlo[0] + vhi[0]) * scale
    cy = Hp / 2 - 0.5 * (vlo[1] + vhi[1]) * scale
    px = (vox_cam[:, 0] * scale + cx).astype(int).clip(0, Wp - 1)
    py = (Hp - 1 - (vox_cam[:, 1] * scale + cy)).astype(int).clip(0, Hp - 1)

    # voxel's camera-axis depth (positive = away from the camera); invert so
    # larger = nearer, to match Depth Anything's convention
    vz = vox_cam[:, 2]
    vz_near = (vz.max() - vz) / (vz.max() - vz.min() + 1e-6)      # [0, 1], 1 = nearest
    dav = depth[py, px]

    # fit linearly on foreground pixels only (background has unrelated depth)
    if biggest_mask is not None:
        fg = biggest_mask[py, px]
        if fg.sum() < 20:
            print("warning: fewer than 20 foreground voxels; using all for fit", flush=True)
            fg = np.ones_like(fg, bool)
    else:
        fg = np.ones_like(dav, bool)
    slope, intercept = np.polyfit(vz_near[fg], dav[fg], 1)
    predicted = slope * vz_near + intercept
    residual = np.abs(dav - predicted)
    med = float(np.median(residual[fg]))
    mad = float(np.median(np.abs(residual[fg] - med)))
    print(f"linear fit over {int(fg.sum())} fg voxels: slope={slope:+.3f} intercept={intercept:+.3f} "
          f"median residual={med:.3f} mad={mad:.3f}", flush=True)

    # adaptive threshold: at least --tol, but also at least med + 3*MAD so we only
    # drop the voxels whose disagreement is a real outlier vs the per-image fit
    threshold = max(a.tol, med + 3.0 * mad)
    carve_flag = residual > threshold
    if biggest_mask is not None:
        carve_flag &= fg
    carve = np.zeros(occ.shape, bool)
    carve[ys, xs, zs] = carve_flag
    frac = float(carve.sum()) / max(1, int(occ.sum()))
    print(f"depth carve: {int(carve.sum())}/{int(occ.sum())} voxels drop ({frac:.0%}) at threshold {threshold:.3f}",
          flush=True)
    if frac > 0.5:
        print(f"skipping (>50% suggests fit failure); writing empty carve", flush=True)
        carve[:] = False
    np.savez(a.out, carve=carve, view=np.array(view, float), tol=np.array(a.tol, float))
    print(f"-> {a.out}", flush=True)


if __name__ == "__main__":
    main()
