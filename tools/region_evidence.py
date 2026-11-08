"""Spots at every scale: each benchmark set's generated mesh cut into the labeller's candidate regions
(semantic_labels.regions: selective search over mesh faces, a pane, the pane with its frame, a whole wall band), each
region the picture shows (facing the camera, clip_bench.photo_of) cropped from it (semantic_labels.photo_crop) and
embedded by SigLIP 2 -- no rendering.  A surface cell belongs to every region holding at least half of its surface
samples; cell_scorer.py scores each part by the cell's best region.

Writes <bench>/<stem>.regions.npz: cells (n, 3) wanted cells (y, x, z), member (m, 2) pairs (cell row, region),
emb (k, d) float16 (zero where unseen), seen (k,), secs.

    tools/timed.sh region-evidence .venv-arm/bin/python tools/region_evidence.py /Volumes/PRO-G40/lego_data/analysis/cell_bench
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]
GEN = Path("/Volumes/PRO-G40/lego_data/analysis/clip_bench/gen")


def membership(mesh, T: np.ndarray, want: np.ndarray, regs: list, half: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """(cells (n, 3), member (m, 2)): each wanted cell with the regions holding >= `half` of its surface samples
    (cell_bench's sampling: 16 per stud^2, a sample's cell the nearest wanted cell centre within 1)."""
    import trimesh
    from scipy.sparse import csr_matrix
    from scipy.spatial import cKDTree
    from cell_bench import SAMPLES_PER_STUD2
    g = lambda p: p @ T[:3, :3].T + T[:3, 3]
    area = np.linalg.norm(np.cross(*(g(mesh.triangles[:, k]) - g(mesh.triangles[:, 0]) for k in (1, 2))), axis=1).sum() / 2
    pts, fid = trimesh.sample.sample_surface(mesh, int(area * SAMPLES_PER_STUD2), seed=0)
    cells = np.argwhere(want)
    d, j = cKDTree(cells[:, [1, 0, 2]] + 0.5).query(g(pts), distance_upper_bound=1.0)
    ok = np.isfinite(d)
    nf, nc = len(mesh.faces), len(cells)
    S = csr_matrix((np.ones(ok.sum()), (j[ok], fid[ok])), shape=(nc, nf))           # cell x face samples
    fr = np.concatenate([r["faces"] for r in regs]); rr = np.repeat(np.arange(len(regs)), [len(r["faces"]) for r in regs])
    R = csr_matrix((np.ones(len(fr)), (fr, rr)), shape=(nf, len(regs)))           # face x region
    C = (S @ R).tocoo()
    tot = np.asarray(S.sum(1)).ravel()
    keep = C.data >= half * tot[C.row]
    return cells, np.stack([C.row[keep], C.col[keep]], 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bench", type=Path); ap.add_argument("--gen", type=Path, default=GEN)
    ap.add_argument("--clip", default="google/siglip2-so400m-patch14-384"); ap.add_argument("--only", default="")
    a = ap.parse_args()
    import semantic_labels as S
    from clip_bench import photo_of
    from sampler_demo import target_from_mesh
    S.MODEL = a.clip
    L = S.parser().parse_args(["-", "--out", "-"])           # the labeller's region settings
    for f in sorted(a.bench.glob("[!.]*.gen.npz")):
        stem = f.name.split(".")[0]; out = a.bench / f"{stem}.regions.npz"
        if out.exists() or (a.only and stem not in a.only.split(",")):
            continue
        t0 = time.time()
        gen = a.gen / f"{stem}.glb"
        r = np.load(f, allow_pickle=True)
        t = target_from_mesh(str(gen), 32, shell=3)
        assert (t.want == r["want"]).all(), "target differs from cell_bench's"
        mesh, T = t.mesh_target.mesh, t.mesh_target.mesh_to_voxel.astype(np.float64)
        regs = S.regions(mesh, T, L.k, L.min_area, L.max_area)
        img, uv, fc = photo_of(gen)
        assert len(uv) == len(mesh.vertices) and len(fc) == len(mesh.faces), "photo is not of this mesh"
        fa = mesh.area_faces
        seen = np.array([(fc[r_["faces"]] * fa[r_["faces"]]).sum() >= S.PHOTO_FACING * fa[r_["faces"]].sum() for r_ in regs])
        cells, member = membership(mesh, T, t.want, regs)
        used = np.zeros(len(regs), bool); used[member[:, 1]] = True
        shown = np.flatnonzero(seen & used)
        t1 = time.time()
        e = S.embed_images([S.photo_crop(img, uv, mesh.faces, regs[k]["faces"]) for k in shown]) if len(shown) else np.zeros((0, 1152))
        emb = np.zeros((len(regs), e.shape[1]), np.float16); emb[shown] = e
        sw = np.zeros(len(regs), bool); sw[shown] = True
        np.savez_compressed(out, cells=cells, member=member, emb=emb, seen=sw, secs=time.time() - t0)
        print(f"{stem}: {len(regs)} regions, {len(shown)} shown and on cells, {len(member)} cell-region pairs; "
              f"regions {t1 - t0:.1f} s, crops+SigLIP {time.time() - t1:.1f} s", flush=True)


if __name__ == "__main__":
    main()
