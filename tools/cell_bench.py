"""The part-finding benchmark on the sampler's own grid: each real building set's target cells (as the builder sees
them, --studs) with the part that truly fills each one -- no rendering, no embedding, seconds per set.

Per set (clip_bench.sets, those with a generated mesh in --gen) and per source ("gen": its Hunyuan3D-2 mesh, gen_bench_meshes.py;
"ldraw": its own geometry): sampler_demo.target_from_mesh builds the target; its mesh's faces take the set's parts
(clip_bench.face_truth: nearest face of the set's mesh, aligned); surface samples go to the nearest wanted cell (as
semantic_labels hands regions to cells), and each cell takes the part instance holding most of its samples.

Writes <out>/<set>.<source>.npz: want, visible, colour, bits, normal (H, X, Z, 3; surface_dir.target_normals),
inst (H, X, Z) int32 (-1 no sample), share (its share of the cell's samples), inst_part (instance -> LDraw name), secs.

    tools/timed.sh cell-bench .venv-arm/bin/python tools/cell_bench.py --out /Volumes/PRO-G40/lego_data/analysis/cell_bench
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools"), str(ROOT / "tools" / "set_study")]
GEN = Path("/Volumes/PRO-G40/lego_data/analysis/clip_bench/gen")
SAMPLES_PER_STUD2 = 16


def cell_truth(src, path: str, studs: int, shell: int, photo: bool = False) -> dict:
    """The target of the mesh at `path` and the true part instance of each of its wanted cells.  `photo` (a generated
    mesh, its picture beside it): also each cell's pixel in the picture (uv, (H, X, Z, 2), -1 none) and how squarely
    the picture sees it (seen, the nearest vertex's faces' camera cosine, 0 hidden: clip_bench.photo_of)."""
    import trimesh
    from scipy.spatial import cKDTree
    from clip_bench import face_truth
    from sampler_demo import target_from_mesh
    from everything2lego.surface_dir import target_normals
    t = target_from_mesh(path, studs, shell=shell)
    mt = t.mesh_target
    mesh, T = mt.mesh, mt.mesh_to_voxel.astype(np.float64)
    _, finst = face_truth(src, mesh)
    g = lambda p: p @ T[:3, :3].T + T[:3, 3]                 # mesh -> grid units (x stud, y plate, z stud)
    area = np.linalg.norm(np.cross(*(g(mesh.triangles[:, k]) - g(mesh.triangles[:, 0]) for k in (1, 2))), axis=1).sum() / 2
    pts, fid = trimesh.sample.sample_surface(mesh, int(area * SAMPLES_PER_STUD2), seed=0)
    wc = np.argwhere(t.want)                                 # (y, x, z)
    d, j = cKDTree(wc[:, [1, 0, 2]] + 0.5).query(g(pts), distance_upper_bound=1.0)
    ok = np.isfinite(d)
    cell, inst = j[ok], finst[fid[ok]]
    u, ui = np.unique(np.stack([cell, inst], 1), axis=0, return_inverse=True)
    cnt = np.bincount(ui.ravel())
    tot = np.bincount(cell, minlength=len(wc))
    best = np.full(len(wc), -1); bc = np.zeros(len(wc))
    for (c, q), n in zip(u, cnt):                            # the instance holding most of each cell's samples
        if n > bc[c]:
            best[c], bc[c] = q, n
    H, X, Z = t.want.shape
    out_inst = np.full((H, X, Z), -1, np.int32); share = np.zeros((H, X, Z), np.float32)
    out_inst[tuple(wc.T)] = best; share[tuple(wc.T)] = bc / np.maximum(tot, 1)
    extra = {}
    if photo:                                                # each cell -> its nearest vertex -> where the picture shows it
        from clip_bench import photo_of
        _, uv, fc = photo_of(Path(path))
        vf = np.zeros(len(mesh.vertices)); np.maximum.at(vf, mesh.faces.ravel(), np.repeat(fc, 3))
        _, v = cKDTree(g(mesh.vertices)).query(wc[:, [1, 0, 2]] + 0.5)
        cu = np.full((H, X, Z, 2), -1, np.int16); cs = np.zeros((H, X, Z), np.float16)
        cu[tuple(wc.T)] = uv[v]; cs[tuple(wc.T)] = vf[v]
        extra = dict(uv=cu, seen=cs)
    return dict(want=t.want, visible=t.visible, colour=t.colour, bits=t.bits, **extra,
                normal=target_normals(t.bits, t.want, sigma=1.5).reshape(H, X, Z, 3).astype(np.float16), inst=out_inst, share=share)


def main() -> None:
    from clip_bench import sets
    from ldr_to_mesh import build_mesh
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--gen", type=Path, default=GEN)
    ap.add_argument("--n", type=int, default=40); ap.add_argument("--studs", type=int, default=32)
    ap.add_argument("--shell", type=int, default=3)
    ap.add_argument("--sources", default="gen,ldraw")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    done = 0
    for f in sets(150, 700):
        if done >= a.n:
            break
        if not (a.gen / f"{f.stem}.glb").exists():
            continue
        done += 1
        todo = [s for s in a.sources.split(",") if not (a.out / f"{f.stem}.{s}.npz").exists()]
        if not todo:
            continue
        src = build_mesh(f)
        names = np.asarray(src.metadata["face_part"]); sinst = np.asarray(src.metadata["face_inst"])
        inst_part = {int(q): str(names[np.flatnonzero(sinst == q)[0]]) for q in np.unique(sinst)}
        part_arr = np.array([inst_part.get(q, "") for q in range(max(inst_part) + 1)])
        for s in todo:
            t0 = time.time()
            if s == "gen":
                r = cell_truth(src, str(a.gen / f"{f.stem}.glb"), a.studs, a.shell, photo=True)
            else:
                with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
                    src.export(tmp.name)
                try:
                    r = cell_truth(src, tmp.name, a.studs, a.shell)
                finally:
                    Path(tmp.name).unlink()
            secs = time.time() - t0
            np.savez_compressed(a.out / f"{f.stem}.{s}.npz", inst_part=part_arr, secs=secs, **r)
            print(f"{f.stem} {s}: {int(r['want'].sum())} cells, {int((r['inst'] >= 0).sum())} with a part, {secs:.1f} s", flush=True)


if __name__ == "__main__":
    main()
