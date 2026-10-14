"""Sample a build of any mesh with the statistics sampler and render it.

    python tools/sampler_demo.py mesh.glb out_dir [--studs 24] [--iters 20000] [--kind building]

The target is the mesh's voxel shell (real sets are hollow): the solid's
cells within `--shell` studs of its surface, coloured by the mesh, each
with the mesh's shape in it (4x4x4 samples) — so a sloped or rounded
surface asks for slopes and rounds, where it is in the cell.  `--kind` is what
the object is (vocabulary.json's kinds: building, road_vehicle, creature,
...); the reference models are official models of that kind.
"""
from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from everything2lego.io_ldraw import write_ldr  # noqa: E402
from everything2lego.sampler import FULL, SUB, Target, popcount, sample  # noqa: E402
from everything2lego.semantic_v2.massing import solid_volume  # noqa: E402
from everything2lego.semantic_v2.mesh_target import MeshTarget  # noqa: E402

FEATURES = Path("/Volumes/PRO-G40/lego_data/analysis/features.pkl")


KINDS = Path("/Volumes/PRO-G40/lego_data/analysis/set_kinds.csv")
def mesh_bits(t: MeshTarget, shape) -> np.ndarray:
    """Each grid cell's shape: the mesh voxelised SUB times finer in grid
    units (studs, plates, studs), closed and filled solid, each cell's
    SUB^3 samples as a 64-bit mask (sampler.pack's bit order)."""
    from scipy.ndimage import binary_closing, binary_fill_holes
    m = t.mesh.copy()
    m.apply_transform(t.mesh_to_voxel)                   # mesh -> grid units (x stud, y plate, z stud)
    half = 0.5 / SUB                                     # trimesh centres voxels on multiples of the
    m.apply_translation([-half] * 3)                     # pitch; samples sit mid sub-cell
    v = m.voxelized(pitch=1.0 / SUB)
    surf = v.matrix
    solid = binary_fill_holes(binary_closing(surf, iterations=1) | surf)
    idx = np.argwhere(solid)
    c = (v.transform @ np.c_[idx, np.ones(len(idx))].T).T[:, :3] + half   # sample centres, grid units
    cell = np.floor(c).astype(int)
    sub = np.clip(np.floor((c - cell) * SUB).astype(int), 0, SUB - 1)
    H, X, Z = shape
    ok = ((cell >= 0) & (cell < np.array([X, H, Z]))).all(axis=1)
    cell, sub = cell[ok], sub[ok]
    bit = (sub[:, 1] * SUB * SUB + sub[:, 0] * SUB + sub[:, 2]).astype(np.uint64)
    bits = np.zeros(shape, dtype=np.uint64)
    np.bitwise_or.at(bits, (cell[:, 1], cell[:, 0], cell[:, 2]), np.left_shift(np.uint64(1), bit))
    return bits


def target_from_mesh(path: str, studs: int, shell: int, kind: str | None = None) -> Target:
    t = MeshTarget.from_path(path, max_studs=studs, n_samples=40000, min_island_cells=8)
    occ = t.voxel_grid.occupied()
    solid = solid_volume(occ)
    bits = mesh_bits(t, occ.shape)
    frac = popcount(bits) / SUB ** 3
    inner = solid.copy()
    for _ in range(shell):
        inner = binary_erosion(inner, structure=np.ones((3, 3, 3), bool) & np.array(
            [[[0, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, 1, 0], [1, 1, 1], [0, 1, 0]], [[0, 0, 0], [0, 1, 0], [0, 0, 0]]], bool))
    want = (solid & ~inner) | ((frac > 0.1) & ~inner)
    want[0] |= solid[0]                                   # the bottom layer is the base
    bits[0] = np.where(solid[0], FULL, bits[0])
    visible = (solid & ~binary_erosion(solid)) | (want & ~solid)
    pal = t.voxel_grid.palette_idx
    from scipy.ndimage import distance_transform_edt
    known = pal >= 0
    _, (iy, ix, iz) = distance_transform_edt(~known, return_indices=True)
    colour = pal[iy, ix, iz]
    return Target(want=want, colour=colour, visible=visible, bits=bits, kind=kind)


def load_models() -> list[dict]:
    """The official models' features, each with what it builds (kinds.py)."""
    import pandas as pd
    rows = pickle.load(open(FEATURES, "rb"))
    if KINDS.exists():
        k = pd.read_csv(KINDS).drop_duplicates("id", keep="last")
        kind = dict(zip(k.id, k.kind))
        for r in rows:
            r["object_kind"] = kind.get(r["model"]) or kind.get(r["model"].split("_")[0])
    return [r for r in rows if not r["held_out"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh"); ap.add_argument("out")
    ap.add_argument("--studs", type=int, default=24)
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--shell", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kind", default=None)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    target = target_from_mesh(a.mesh, a.studs, a.shell, a.kind)
    models = load_models()
    t0 = time.time()
    ps, info = sample(target, models, iters=a.iters, seed=a.seed)
    info["seconds"] = round(time.time() - t0, 1)
    write_ldr(ps, out / "model.ldr")
    (out / "info.json").write_text(json.dumps({k: v for k, v in info.items()}, indent=1, default=float))
    print(json.dumps({k: v for k, v in info.items() if k != "history"}, default=float))
    subprocess.run([sys.executable, str(ROOT / "tools/render_build.py"), str(out / "model.ldr"),
                    str(out / "render.png"), "--views", "30,120,210", "--size", "500"], check=False)


if __name__ == "__main__":
    main()
