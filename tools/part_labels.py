"""Label the sampler target's cells with the mesh part they belong to, from a zero-shot mesh part segmentation
(SAMesh: SAM2 on multi-view renders), for the sampler's --parts (per-part mirror symmetry and echo).

    python tools/part_labels.py mesh.glb face2label.json segmented.glb --studs 32 --out parts.npz [--sheet parts.png]

  - SAMesh read the mesh its own way (read_mesh(norm=True): bounding-box centre to 0, max |coordinate| to
    1 - 1e-3; process=True may merge / reorder), so its faces are matched by geometry: both meshes normalised so, each
    face of ours takes the label of the segmented face whose centre is nearest (the distance is reported);
  - our mesh is the one MeshTarget reads (voxelize.load_mesh, processed; its axis alignment keeps the faces), its
    surface sampled (~16 per stud^2), each sample to the nearest wanted cell of the sampler's own target
    (sampler_demo.target_from_mesh, same --studs) within a cell, as semantic_labels does; a cell takes its samples'
    majority part, every other wanted cell the part of the nearest labelled cell -- a part is a solid region;
  - `cell_part` (H, X, Z) int16, parts numbered by size (0 largest), -1 outside the target; `samesh_label` the
    SAMesh label of each part.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]

PLATE = 0.4                                      # a plate's height in studs


def samesh_norm(v: np.ndarray) -> np.ndarray:
    """SAMesh's norm_mesh: bounding-box centre to the origin, the largest |coordinate| to 1 - 1e-3."""
    v = v - (v.min(0) + v.max(0)) / 2
    return v / np.abs(v).max() * (1 - 1e-3)


def face_labels(mesh, face2label: dict, seg) -> tuple[np.ndarray, np.ndarray]:
    """Per face of `mesh`: the SAMesh label of the nearest segmented face (normalised face centres), and the distance."""
    from scipy.spatial import cKDTree
    lab = np.array([face2label[str(i)] for i in range(len(seg.faces))])
    d, j = cKDTree(samesh_norm(seg.vertices)[seg.faces].mean(1)).query(samesh_norm(mesh.vertices)[mesh.faces].mean(1))
    return lab[j], d


def cell_parts(t, mt, flab: np.ndarray, per_area: float = 16.0, seed: int = 0) -> tuple[np.ndarray, np.ndarray, dict]:
    """(cell_part (H, X, Z) int16 by size, SAMesh label per part, stats): samples' majority per cell, the rest nearest."""
    import trimesh
    from scipy.ndimage import distance_transform_edt
    from scipy.spatial import cKDTree
    T = mt.mesh_to_voxel.astype(np.float64)
    tri = mt.mesh.triangles @ T[:3, :3].T + T[:3, 3]
    tri = tri * np.array([1.0, PLATE, 1.0])                       # grid units -> studs, for the area
    area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum() / 2
    pts, fid = trimesh.sample.sample_surface(mt.mesh, int(per_area * area), seed=seed)
    wc = np.argwhere(t.want)                                      # (y, x, z)
    d, j = cKDTree(wc[:, [1, 0, 2]] + 0.5).query(pts @ T[:3, :3].T + T[:3, 3], distance_upper_bound=1.0)
    ok = np.isfinite(d)
    ids, lab = np.unique(flab, return_inverse=True)
    votes = np.zeros((len(wc), len(ids)), np.int32)
    np.add.at(votes, (j[ok], lab[fid[ok]]), 1)
    hit = votes.sum(1) > 0
    known = np.full(t.want.shape, -1, np.int32)
    known[tuple(wc[hit].T)] = votes[hit].argmax(1)
    _, (iy, ix, iz) = distance_transform_edt(known < 0, sampling=(PLATE, 1.0, 1.0), return_indices=True)
    full = np.where(t.want, known[iy, ix, iz], -1)
    n = np.bincount(full[full >= 0], minlength=len(ids))
    order = np.argsort(-n, kind="stable")
    order = order[n[order] > 0]
    rank = np.full(len(ids), -1); rank[order] = np.arange(len(order))
    cell_part = np.where(full >= 0, rank[np.maximum(full, 0)], -1).astype(np.int16)
    stats = dict(samples=len(pts), samples_in_cells=float(ok.mean()), cells_hit=float(hit.mean()),
                 purity=float((votes[hit].max(1) / votes[hit].sum(1)).mean()))
    return cell_part, ids[order], stats


def sheet(cell_part: np.ndarray, path: str, scale: int = 8) -> None:
    """Front (-z), side (+x) and top views: each pixel the part of the first labelled cell along the view."""
    from PIL import Image
    rng = np.random.default_rng(1)
    pal = np.vstack([rng.integers(40, 255, (cell_part.max() + 1, 3)), [[255, 255, 255]]]).astype(np.uint8)
    views = []
    for ax, flip_y in ((2, True), (1, True), (0, False)):           # along z, along x, from above
        m = np.moveaxis(cell_part, ax, -1)
        if ax == 0:
            m = m[..., ::-1]                                       # from the top down
        has = m >= 0
        first = np.where(has.any(-1), np.take_along_axis(m, has.argmax(-1)[..., None], -1)[..., 0], -1)
        img = pal[first]
        views.append(img[::-1] if flip_y else img)
    h = max(v.shape[0] for v in views)
    row = np.hstack([np.pad(v, ((0, h - v.shape[0]), (0, 2), (0, 0)), constant_values=255) for v in views])
    Image.fromarray(row).resize((row.shape[1] * scale, row.shape[0] * scale), Image.NEAREST).save(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh"); ap.add_argument("face2label"); ap.add_argument("segmented")
    ap.add_argument("--studs", type=int, default=32)
    ap.add_argument("--out", required=True); ap.add_argument("--sheet", default=None)
    a = ap.parse_args()
    import trimesh
    from everything2lego.semantic_v2.mesh_target import MeshTarget
    from everything2lego.voxelize import load_mesh
    from sampler_demo import target_from_mesh
    t = target_from_mesh(a.mesh, a.studs, shell=2)
    mt = MeshTarget.from_path(a.mesh, max_studs=a.studs, n_samples=40000, min_island_cells=8)
    mesh = load_mesh(a.mesh)
    mesh.process(validate=True)                                   # as MeshTarget.from_path, before it aligns the axes
    assert len(mesh.faces) == len(mt.mesh.faces), "MeshTarget's mesh has other faces than load_mesh's"
    seg = trimesh.load(a.segmented, force="mesh", process=False)
    flab, dist = face_labels(mesh, json.loads(Path(a.face2label).read_text()), seg)
    print(f"faces: ours {len(mesh.faces)}, segmented {len(seg.faces)}; nearest segmented face centre (normalised units,"
          f" box half-size 1): median {np.median(dist):.2g}, p99 {np.quantile(dist, 0.99):.2g}, max {dist.max():.2g}")
    cell_part, ids, st = cell_parts(t, mt, flab)
    print(f"{st['samples']} surface samples, {st['samples_in_cells']:.1%} within a cell of a wanted cell; "
          f"{st['cells_hit']:.1%} of wanted cells hit, majority purity {st['purity']:.2f}")
    np.savez_compressed(a.out, cell_part=cell_part, samesh_label=ids)
    print(f"-> {a.out}: {len(ids)} parts over {int((cell_part >= 0).sum())} wanted cells {cell_part.shape}")
    print("  part  samesh  cells     y (plates)   x (studs)   z (studs)")
    for p in range(len(ids)):
        c = np.argwhere(cell_part == p)
        lo, hi = c.min(0), c.max(0)
        print(f"  {p:>4}  {ids[p]:>6}  {len(c):>5}   {lo[0]:>3}-{hi[0]:<3}      {lo[1]:>3}-{hi[1]:<3}     {lo[2]:>3}-{hi[2]:<3}")
    if a.sheet:
        sheet(cell_part, a.sheet)
        print(f"-> {a.sheet}")


if __name__ == "__main__":
    main()
