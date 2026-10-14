"""Sample a build of any mesh with the statistics sampler and render it.

    python tools/sampler_demo.py mesh.glb out_dir [--studs 24] [--iters 20000]

The target is the mesh's voxel shell (real sets are hollow): the solid's
cells within `--shell` studs of its surface, coloured by the mesh.
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
from everything2lego.sampler import Target, sample  # noqa: E402
from everything2lego.semantic_v2.massing import solid_volume  # noqa: E402
from everything2lego.semantic_v2.mesh_target import MeshTarget  # noqa: E402

FEATURES = Path("/Volumes/PRO-G40/lego_data/analysis/features.pkl")


def target_from_mesh(path: str, studs: int, shell: int) -> Target:
    t = MeshTarget.from_path(path, max_studs=studs, n_samples=40000, min_island_cells=8)
    occ = t.voxel_grid.occupied()
    solid = solid_volume(occ)
    inner = solid.copy()
    for _ in range(shell):
        inner = binary_erosion(inner, structure=np.ones((3, 3, 3), bool) & np.array(
            [[[0, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, 1, 0], [1, 1, 1], [0, 1, 0]], [[0, 0, 0], [0, 1, 0], [0, 0, 0]]], bool))
    want = solid & ~inner
    want[0] |= solid[0]                                   # the bottom layer is the base
    visible = solid & ~binary_erosion(solid)
    pal = t.voxel_grid.palette_idx
    from scipy.ndimage import distance_transform_edt
    known = pal >= 0
    _, (iy, ix, iz) = distance_transform_edt(~known, return_indices=True)
    colour = pal[iy, ix, iz]
    return Target(want=want, colour=colour, visible=visible)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh"); ap.add_argument("out")
    ap.add_argument("--studs", type=int, default=24)
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--shell", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    target = target_from_mesh(a.mesh, a.studs, a.shell)
    models = pickle.load(open(FEATURES, "rb"))
    t0 = time.time()
    ps, info = sample(target, [m for m in models if not m["held_out"]], iters=a.iters, seed=a.seed)
    info["seconds"] = round(time.time() - t0, 1)
    write_ldr(ps, out / "model.ldr")
    (out / "info.json").write_text(json.dumps({k: v for k, v in info.items()}, indent=1, default=float))
    print(json.dumps({k: v for k, v in info.items() if k != "history"}, default=float))
    subprocess.run([sys.executable, str(ROOT / "tools/render_build.py"), str(out / "model.ldr"),
                    str(out / "render.png"), "--views", "30,120,210", "--size", "500"], check=False)


if __name__ == "__main__":
    main()
