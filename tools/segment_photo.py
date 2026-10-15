"""Segment a photo with SAM, lift the masks to voxel region kinds by
projecting the paired mesh.  The sampler then uses the right object kind
per region of the final build (wheels get road_vehicle weights, body gets
object weights, windows stay in walls, ...).

    python tools/segment_photo.py photo.png mesh.glb --out regions.npz
        [--studs 24] [--view LAT,LON] [--kinds "big=road_vehicle,0=object"]

SAM finds masks on the photo; each mesh face is projected to the photo
with an orthographic camera (default: frontal +Z, model Y up — the frame
Hunyuan3D leaves meshes in) and takes the mask covering its screen pixel.
Faces with no mask (background, mask gaps) go to the "other" region.
Then: voxelise, label each cell by the nearest face's mask id, map mask
id → kind as in segment_mesh.py.

Needs: segment-anything, torch (the Intel .venv), SAM_CHECKPOINT or the
default /Volumes/PRO-G40/models/sam/sam_vit_b_01ec64.pth.  The arm venv
has no torch; this tool runs under .venv (Intel + Rosetta).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from everything2lego.sampler import known_kinds  # noqa: E402
from segment_mesh import _components  # noqa: E402

DEFAULT_SAM = os.environ.get("SAM_CHECKPOINT", "/Volumes/PRO-G40/models/sam/sam_vit_b_01ec64.pth")


def _sam_masks(photo: Image.Image, checkpoint: str, model_type: str = "vit_b") -> list[dict]:
    """[{"segmentation": HxW bool, "area": int, ...}] of SAM's automatic masks."""
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    gen = SamAutomaticMaskGenerator(sam, points_per_side=16, min_mask_region_area=200)
    arr = np.array(photo.convert("RGB"))
    return gen.generate(arr)


def _camera(lat_deg: float, lon_deg: float):
    """Orthographic look-at matrix (world → camera) for lat/lon in degrees,
    model Y up (as Hunyuan3D meshes come out).  Returns a function
    world (N,3) → camera (x, y, depth)."""
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    f = np.array([np.cos(lat) * np.sin(lon), np.sin(lat), np.cos(lat) * np.cos(lon)])  # forward (world → camera z)
    up = np.array([0.0, 1.0, 0.0])
    right = np.cross(up, f); right /= np.linalg.norm(right) + 1e-9
    up2 = np.cross(f, right)
    R = np.stack([right, up2, f])                                                      # 3 x 3

    def project(pts: np.ndarray):
        return pts @ R.T                                                               # (N, 3) in camera frame
    return project


def _face_to_mask(mesh: trimesh.Trimesh, masks: list[dict], view: tuple[float, float], W: int, H: int) -> np.ndarray:
    """mask id per face (-1 none).  Project triangle centres with an
    orthographic camera onto a W x H image, read the mask layer at that
    pixel; the mask with the largest overlap on that pixel wins."""
    project = _camera(*view)
    centres = project(mesh.triangles_center)
    # fit the mesh into the image
    lo, hi = centres[:, :2].min(0), centres[:, :2].max(0)
    scale = min(W / (hi[0] - lo[0] + 1e-6), H / (hi[1] - lo[1] + 1e-6)) * 0.95
    cx, cy = W / 2 - 0.5 * (lo[0] + hi[0]) * scale, H / 2 - 0.5 * (lo[1] + hi[1]) * scale
    px = (centres[:, 0] * scale + cx).astype(int).clip(0, W - 1)
    py = (H - 1 - (centres[:, 1] * scale + cy)).astype(int).clip(0, H - 1)             # image y goes down
    # mask stack: for each (y, x), the mask with highest area wins
    stack = np.full((H, W), -1, np.int32)
    for i, m in enumerate(sorted(masks, key=lambda m: -m["area"])):
        seg = np.asarray(m["segmentation"], bool)
        if seg.shape != (H, W):
            seg = np.asarray(Image.fromarray(seg).resize((W, H))) > 0
        stack = np.where((stack < 0) & seg, i, stack)
    return stack[py, px]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("photo", type=Path)
    ap.add_argument("mesh", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--studs", type=int, default=24, help="voxel resolution (match sampler_demo --studs)")
    ap.add_argument("--view", default="0,0", help="LAT,LON (deg) of the camera; 0,0 is frontal +Z, model Y up")
    ap.add_argument("--kind", default=None, help="main object kind (biggest mask gets this)")
    ap.add_argument("--kinds", default=None, help='comma-separated mask_id=kind pairs, e.g. "0=road_vehicle,1=object"')
    ap.add_argument("--sam", default=DEFAULT_SAM, help="SAM checkpoint (env SAM_CHECKPOINT, default ViT-B)")
    ap.add_argument("--sam-type", default="vit_b")
    a = ap.parse_args()
    photo = Image.open(a.photo).convert("RGB")
    print(f"segmenting {a.photo.name} ({photo.size}) with SAM {a.sam_type} ...", flush=True)
    masks = _sam_masks(photo, a.sam, a.sam_type)
    print(f"  {len(masks)} masks", flush=True)
    mesh = trimesh.load(a.mesh, force="mesh", process=False)
    view = tuple(float(v) for v in a.view.split(","))
    W, H = photo.size
    face_mask = _face_to_mask(mesh, masks, view, W, H)
    # mask id per cluster: split faces with same mask id into connected components,
    # then each component is a region (same pipeline as segment_mesh)
    comp = _components(mesh, face_mask)
    face_region = comp
    # kind per mask id: biggest mask → main kind, rest → object; overrides win
    kinds_valid = known_kinds()
    overrides: dict[str, str] = {}
    if a.kinds:
        for pair in a.kinds.split(","):
            name, k = pair.split("=")
            k = k.strip()
            if k not in kinds_valid:
                raise SystemExit(f"unknown kind {k!r}; valid: {sorted(kinds_valid)}")
            overrides[name.strip()] = k
    main_kind = a.kind or "object"
    counts = np.bincount(face_mask[face_mask >= 0], minlength=len(masks))
    order = np.argsort(-counts)
    mask_kind: dict[int, str] = {}
    for rank, mid in enumerate(order):
        mask_kind[int(mid)] = overrides.get(str(int(mid))) or (main_kind if rank == 0 else "object")
    face_kind = np.array([mask_kind.get(int(m), "object") if m >= 0 else "object"
                          for m in face_mask], dtype=object)
    # map regions to kind by majority face-kind
    region_kind: dict[int, str] = {}
    for rid in np.unique(face_region):
        faces = np.flatnonzero(face_region == rid)
        if len(faces):
            from collections import Counter
            region_kind[int(rid)] = Counter(face_kind[faces]).most_common(1)[0][0]
    # voxelise and label
    from everything2lego.semantic_v2 import MeshTarget
    mt = MeshTarget.from_path(a.mesh, max_studs=a.studs, n_samples=40000, min_island_cells=8)
    pal = mt.voxel_grid.palette_idx
    occ = pal >= 0
    from scipy.spatial import cKDTree
    tree = cKDTree(mesh.triangles_center)
    T = np.linalg.inv(mt.mesh_to_voxel)
    ys, xs, zs = np.nonzero(occ)
    cell_centres_mesh = (T @ np.c_[np.stack([xs + 0.5, ys + 0.5, zs + 0.5], 1), np.ones(len(xs))].T).T[:, :3]
    _, nearest_face = tree.query(cell_centres_mesh)
    label = np.full(occ.shape, -1, np.int64)
    label[ys, xs, zs] = face_region[nearest_face]
    np.savez(a.out, label=label,
             kind_of=np.array([[str(r), region_kind[r]] for r in region_kind], dtype=object))
    seen = {int(r): int((label == r).sum()) for r in region_kind}
    print(f"-> {a.out}: {len(region_kind)} regions by mask-majority; cells {dict(sorted(seen.items(), key=lambda x: -x[1])[:10])}")


if __name__ == "__main__":
    main()
