"""Segment a coloured mesh into regions, each a connected cluster of faces
whose colours are similar — a wheel on a car, a roof on a building, a bust
on a sculpture.  Each region can be assigned a sampler kind (building,
road_vehicle, object, ...), so the sampler uses the right weights per
voxel.  Lighter than SAM and works offline on the mesh Hunyuan3D or any
coloured-mesh backend produces.

    python tools/segment_mesh.py mesh.glb --out regions.npz [--studs 24] [--k 4]

The segmenter:
  - samples face colours in CIE Lab and clusters them with k-means
    (k = `--k`, default 4);
  - splits each colour cluster by connected components over face
    adjacency, so a car's black wheels are one region, its black cabin
    is another;
  - voxelises the mesh at `--studs` resolution, labels each occupied cell
    by the nearest region's assignment and writes {palette_code: kind}
    for the sampler's --regions path (every cell of colour C goes to the
    region's kind — a shortcut that works when colours partition cells).

Pass --kinds "car=road_vehicle,body=object" (region name = kind) to label
regions by hand; otherwise the biggest region keeps the main --kind, the
other regions become 'object'.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from everything2lego.colors import _palette_lab, _rgb_to_lab, palette  # noqa: E402
from everything2lego.sampler import known_kinds  # noqa: E402


def _face_colours(mesh: trimesh.Trimesh) -> np.ndarray | None:
    """(F, 3) uint8 face colours — from face colours, vertex colours, or
    the texture via visual.to_color() for textured (PBR) meshes."""
    v = mesh.visual
    if hasattr(v, "face_colors") and v.face_colors is not None and len(v.face_colors) == len(mesh.faces):
        return np.asarray(v.face_colors[:, :3], np.uint8)
    if hasattr(v, "vertex_colors") and v.vertex_colors is not None and len(v.vertex_colors) == len(mesh.vertices):
        vc = np.asarray(v.vertex_colors[:, :3], np.uint8)
        return vc[mesh.faces].mean(axis=1).astype(np.uint8)
    if hasattr(v, "to_color"):
        try:
            vc = np.asarray(v.to_color().vertex_colors[:, :3], np.uint8)
            if len(vc) == len(mesh.vertices):
                return vc[mesh.faces].mean(axis=1).astype(np.uint8)
        except Exception:
            pass
    return None


def _cluster_colours(rgb: np.ndarray, k: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """(cluster id per face, cluster centres as RGB uint8)."""
    from sklearn.cluster import KMeans
    lab = _rgb_to_lab((rgb / 255.0).astype(np.float32))
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(lab)
    centres = ((km.cluster_centers_ * np.array([1.0, 1.0, 1.0]))).clip(0, 100)
    # back to approximate RGB via nearest palette for display
    pal = _palette_lab()
    reps = np.argmin(((pal[None, :, :] - km.cluster_centers_[:, None, :]) ** 2).sum(-1), axis=1)
    pal_rgb = np.array([c.rgb for c in palette()], np.uint8)
    return km.labels_, pal_rgb[reps]


def _components(mesh: trimesh.Trimesh, face_label: np.ndarray) -> np.ndarray:
    """Connected components on faces, respecting label boundaries: two
    adjacent faces are in the same component iff they share a label."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    e = mesh.face_adjacency                              # (M, 2) pairs of face ids
    same = face_label[e[:, 0]] == face_label[e[:, 1]]
    e = e[same]
    n = len(mesh.faces)
    g = csr_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    _, comp = connected_components(g, directed=False)
    return comp


def _merge_small(mesh: trimesh.Trimesh, face_region: np.ndarray, min_faces: int = 20) -> np.ndarray:
    """Absorb regions with fewer than `min_faces` into the largest adjacent
    region (and from there into theirs' if still small).  Keeps the big
    regions' ids; small regions' faces take their new region's id.  Removes
    projection noise and SAM mask edge artefacts that split one surface
    into many tiny components."""
    from collections import Counter, defaultdict
    region_faces = defaultdict(list)
    for i, r in enumerate(face_region):
        region_faces[int(r)].append(i)
    small = {r for r, f in region_faces.items() if len(f) < min_faces}
    if not small:
        return face_region
    # neighbour sizes per small region
    e = mesh.face_adjacency
    adj = defaultdict(lambda: Counter())
    for a, b in e:
        ra, rb = int(face_region[a]), int(face_region[b])
        if ra != rb:
            adj[ra][rb] += 1
            adj[rb][ra] += 1
    out = face_region.copy()
    changed = True
    while changed and small:
        changed = False
        for r in list(small):
            # pick the LARGEST neighbour that isn't also small
            big_neighbours = [(len(region_faces[nb]), nb) for nb in adj[r] if nb not in small]
            if not big_neighbours:
                continue
            _, target = max(big_neighbours)
            for f in region_faces[r]:
                out[f] = target
            region_faces[target].extend(region_faces[r])
            # retarget adj
            for nb, w in adj[r].items():
                if nb != target:
                    adj[target][nb] += w
                    adj[nb][target] += w
                    adj[nb].pop(r, None)
            del adj[r]
            del region_faces[r]
            small.remove(r)
            changed = True
    return out


def segment(mesh: trimesh.Trimesh, k: int = 4) -> list[dict]:
    """[{id, size (faces), rgb, bbox}] of each region (connected faces of
    the same colour cluster), largest first."""
    rgb = _face_colours(mesh)
    if rgb is None:
        raise ValueError("mesh has no face or vertex colours")
    cluster, cluster_rgb = _cluster_colours(rgb, k)
    comp = _components(mesh, cluster)
    regions = []
    for c in np.unique(comp):
        faces = np.flatnonzero(comp == c)
        if len(faces) < 4:
            continue
        cl = int(cluster[faces[0]])
        v = mesh.vertices[mesh.faces[faces].ravel()]
        regions.append({"id": int(c), "size": int(len(faces)), "cluster": cl,
                        "rgb": [int(x) for x in cluster_rgb[cl]],
                        "bbox": [v.min(axis=0).tolist(), v.max(axis=0).tolist()]})
    regions.sort(key=lambda r: -r["size"])
    return regions


def _nearest_palette(rgb) -> int:
    """Palette code for an RGB triple."""
    pal_lab = _palette_lab()
    lab = _rgb_to_lab(np.asarray(rgb, np.float32)[None] / 255.0)[0]
    return int(palette()[int(np.argmin(((pal_lab - lab) ** 2).sum(1)))].code)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="where to write regions.json for sampler_demo --regions")
    ap.add_argument("--k", type=int, default=8, help="k-means clusters over face colours (default 8; bump for highly textured meshes)")
    ap.add_argument("--studs", type=int, default=24, help="voxelise at this grid resolution (same as sampler_demo)")
    ap.add_argument("--kind", default=None, help="the object's main kind (used to label the biggest region)")
    ap.add_argument("--kinds", default=None, help="comma-separated region=kind, e.g. 'wheel=road_vehicle,body=object'")
    a = ap.parse_args()
    mesh = trimesh.load(a.mesh, force="mesh", process=False)
    regions = segment(mesh, k=a.k)
    print(f"{len(regions)} regions:")
    for r in regions[:10]:
        print(f"  id {r['id']:>3}  cluster {r['cluster']}  faces {r['size']:>6}  rgb {tuple(r['rgb'])}")
    if a.out is None:
        return
    kinds = known_kinds()
    overrides: dict[str, str] = {}
    if a.kinds:
        for pair in a.kinds.split(","):
            name, k = pair.split("=")
            if k.strip() not in kinds:
                raise SystemExit(f"unknown kind {k!r}; valid: {sorted(kinds)}")
            overrides[name.strip()] = k.strip()
    # Voxelise and label each occupied cell by the region of the nearest face.
    # Then write a .npz the sampler loads directly: `region_label` (per cell, int)
    # and `kind_of` (region id -> kind string).
    from everything2lego.semantic_v2 import MeshTarget
    mt = MeshTarget.from_path(a.mesh, max_studs=a.studs, n_samples=40000, min_island_cells=8)
    pal = mt.voxel_grid.palette_idx
    occ = pal >= 0
    main_kind = a.kind or "object"
    # kind per CLUSTER (all wheels share one cluster): biggest cluster -> main_kind,
    # rest -> object.  Override by cluster index: 'cluster0=road_vehicle,cluster1=object'.
    cluster_size: dict[int, int] = {}
    for r in regions:
        cluster_size[r["cluster"]] = cluster_size.get(r["cluster"], 0) + r["size"]
    cluster_order = sorted(cluster_size, key=lambda c: -cluster_size[c])
    cluster_kind: dict[int, str] = {}
    for i, c in enumerate(cluster_order):
        cluster_kind[c] = overrides.get(f"cluster{i}") or (main_kind if i == 0 else "object")
    region_kind = {r["id"]: cluster_kind[r["cluster"]] for r in regions}
    # for each occupied cell, find the nearest face centre of the mesh and
    # take its region id (connected-components label).  Fast query via trimesh.
    face_region = np.zeros(len(mesh.faces), np.int64)
    comp = _components(mesh, _cluster_colours(_face_colours(mesh), a.k)[0])
    face_region[:] = _merge_small(mesh, comp)
    centres = mesh.triangles_center
    from scipy.spatial import cKDTree
    tree = cKDTree(centres)
    # cell centres in mesh coordinates
    T = np.linalg.inv(mt.mesh_to_voxel)
    ys, xs, zs = np.nonzero(occ)
    cell_centres_vox = np.stack([xs + 0.5, ys + 0.5, zs + 0.5], 1)       # (N, 3) in voxel space (x, y, z)
    cell_centres_mesh = (T @ np.c_[cell_centres_vox, np.ones(len(cell_centres_vox))].T).T[:, :3]
    _, nearest_face = tree.query(cell_centres_mesh)
    label = np.full(occ.shape, -1, np.int64)
    label[ys, xs, zs] = face_region[nearest_face]
    np.savez(a.out, label=label, kind_of=np.array([[str(rid), region_kind.get(rid, main_kind)]
                                                    for rid in region_kind], dtype=object))
    seen = {k: int((label == k).sum()) for k in region_kind}
    print(f"-> {a.out}: {len(region_kind)} regions, cells per region {seen}")


if __name__ == "__main__":
    main()
