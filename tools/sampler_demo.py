"""Sample a build of any mesh with the statistics sampler and render it.

    python tools/sampler_demo.py mesh.glb out_dir [--studs 24] [--iters 20000] [--kind building] [--style 1]

The target is the mesh's voxel shell (real sets are hollow): the solid's
cells within `--shell` studs of its surface, coloured by the mesh, each
with the mesh's shape in it (4x4x4 samples) — so a sloped or rounded
surface asks for slopes and rounds, where it is in the cell.  `--kind` is what
the object is (vocabulary.json's kinds: building, road_vehicle, creature,
...); the reference models are official models of that kind.
"""
from __future__ import annotations

import os
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

FEATURES = Path(os.environ.get("LEGO_DATA", "/Volumes/PRO-G40/lego_data")) / "analysis" / "features.pkl"


KINDS = Path(os.environ.get("LEGO_DATA", "/Volumes/PRO-G40/lego_data")) / "analysis" / "set_kinds.csv"
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


def target_from_mesh(path: str, studs: int, shell: int, kind: str | None = None,
                     stand: int = 0, regions: dict[int, str] | None = None,
                     seg_label: np.ndarray | None = None,
                     seg_kind_of: dict[int, str] | None = None,
                     carve: np.ndarray | None = None) -> Target:
    t = MeshTarget.from_path(path, max_studs=studs, n_samples=40000, min_island_cells=8)
    occ = t.voxel_grid.occupied()
    if carve is not None:
        if carve.shape != occ.shape:
            raise ValueError(f"carve shape {carve.shape} != voxel shape {occ.shape}; "
                             f"use the same --studs for segment_photo and sampler_demo")
        occ = occ & ~carve
    solid = solid_volume(occ)
    bits = mesh_bits(t, occ.shape)
    if carve is not None:
        bits[carve] = 0
    frac = popcount(bits) / SUB ** 3
    inner = solid.copy()
    for _ in range(shell):
        inner = binary_erosion(inner, structure=np.ones((3, 3, 3), bool) & np.array(
            [[[0, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, 1, 0], [1, 1, 1], [0, 1, 0]], [[0, 0, 0], [0, 1, 0], [0, 0, 0]]], bool))
    want = (solid & ~inner) | ((frac > 0.1) & ~inner)
    # the base: layer 0 is the model's full x-z shadow, one stud wider than
    # the model, so the build sits on a plate that reaches past its walls.
    # Vehicles don't want a solid chassis plate — they sit on wheels at the
    # corners; let the mesh decide what layer 0 contains.
    shadow = want.any(0)
    from scipy.ndimage import binary_dilation as _dil
    if kind != "road_vehicle":
        base = _dil(shadow, iterations=1)
        want[0] |= base
        bits[0] = np.where(base, FULL, bits[0])
    if stand > 0:
        # a plate `stand` studs wider on each side than the model's shadow,
        # for a figure or small object that would otherwise sit on no plate
        pad = _dil(shadow, iterations=stand)
        want[0] |= pad
        bits[0] = np.where(pad, FULL, bits[0])
    visible = (solid & ~binary_erosion(solid)) | (want & ~solid)
    pal = t.voxel_grid.palette_idx
    from scipy.ndimage import distance_transform_edt
    known = pal >= 0
    _, (iy, ix, iz) = distance_transform_edt(~known, return_indices=True)
    colour = pal[iy, ix, iz]
    region_kind = None
    if regions:
        from everything2lego.sampler import known_kinds
        valid = known_kinds()
        bad = set(regions.values()) - valid
        if bad:
            raise ValueError(f"regions: unknown kinds {bad}; valid: {sorted(valid)}")
        region_kind = np.full(want.shape, "", dtype=object)
        for code, k in regions.items():
            region_kind[colour == int(code)] = k
    if seg_label is not None and seg_kind_of:
        from everything2lego.sampler import known_kinds
        valid = known_kinds()
        bad = set(seg_kind_of.values()) - valid
        if bad:
            raise ValueError(f"segmentation kinds not known: {bad}; valid: {sorted(valid)}")
        if seg_label.shape != want.shape:
            raise ValueError(f"segmentation shape {seg_label.shape} != target {want.shape}; "
                             f"use the same --studs for segment_mesh and sampler_demo")
        region_kind = np.full(want.shape, "", dtype=object)
        for rid, k in seg_kind_of.items():
            region_kind[seg_label == int(rid)] = k
    return Target(want=want, colour=colour, visible=visible, bits=bits, kind=kind, region_kind=region_kind)


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
    ap.add_argument("--iters", type=int, default=0, help="sampler iterations; 0 (default) scales with target volume (~40 per occupied cell, min 20k)")
    ap.add_argument("--shell", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kind", default=None)
    ap.add_argument("--style", type=float, default=1.0, help="style weight: real-set statistics against form")
    ap.add_argument("--stand", type=int, default=0, help="pad base plate by N studs (for figures or small objects)")
    ap.add_argument("--regions", default=None, help='JSON file mapping palette codes to kinds, e.g. {"0":"road_vehicle","4":"object"} — gives different sampler weights per colour region')
    ap.add_argument("--segmentation", default=None, help="npz from tools/segment_mesh.py: per-voxel region_kind (overrides --regions)")
    ap.add_argument("--prune-unsupported", action="store_true", help="drop parts with no vertical supporter below from the final build (post-process)")
    ap.add_argument("--depth-carve", default=None, help="npz from tools/depth_carve.py: carve mask OR-ed into --segmentation's carve")
    ap.add_argument("--tile-boost", type=float, default=0.0, help="log-bonus on every tile option at proposal time; real sets use 15-23%% tiles vs our 11%%, +1 to +2 is a reasonable range")
    ap.add_argument("--snot-boost", type=float, default=0.0, help="log-bonus on every SNOT option (brickS/plateS/tileS) at proposal time; real sets use 17-32%% SNOT vs our ~0%%")
    ap.add_argument("--subassemblies", type=int, default=0, help="group placements into N-part submodels in an MPD .ldr (real sets have 11-26 sub-assemblies; 25 is a reasonable target)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    regions = None
    if a.regions:
        regions = {int(k): v for k, v in json.loads(Path(a.regions).read_text()).items()}
    seg_label, seg_kind_of, carve = None, None, None
    if a.segmentation:
        d = np.load(a.segmentation, allow_pickle=True)
        seg_label = d["label"]
        seg_kind_of = {int(k): v for k, v in d["kind_of"]}
        if "carve" in d.files:
            carve = d["carve"]
    if a.depth_carve:
        d = np.load(a.depth_carve, allow_pickle=True)
        extra = d["carve"]
        carve = extra if carve is None else (carve | extra)
    target = target_from_mesh(a.mesh, a.studs, a.shell, a.kind, stand=a.stand, regions=regions,
                              seg_label=seg_label, seg_kind_of=seg_kind_of, carve=carve)
    models = load_models()
    t0 = time.time()
    from everything2lego.sampler import Weights
    iters = a.iters if a.iters > 0 else max(20000, 40 * int(target.want.sum()))
    ps, info = sample(target, models, iters=iters, seed=a.seed, weights=Weights(style=a.style),
                      tile_boost=a.tile_boost, snot_boost=a.snot_boost)
    info["iters"] = iters
    info["seconds"] = round(time.time() - t0, 1)
    if a.prune_unsupported:
        from everything2lego.supports import prune_unsupported
        ps, pruned = prune_unsupported(list(ps))
        info["pruned_unsupported"] = pruned
        info["parts"] = len(ps)
    subs = None
    if a.subassemblies > 0:
        from everything2lego.subassemblies import split_by_y_bands
        subs = split_by_y_bands(list(ps), target_size=a.subassemblies)
        info["subassemblies"] = [len(g) for g in subs]
    write_ldr(ps, out / "model.ldr", subassemblies=subs)
    (out / "info.json").write_text(json.dumps({k: v for k, v in info.items()}, indent=1, default=float))
    print(json.dumps({k: v for k, v in info.items() if k != "history"}, default=float))
    subprocess.run([sys.executable, str(ROOT / "tools/render_build.py"), str(out / "model.ldr"),
                    str(out / "render.png"), "--views", "30,120,210", "--size", "500"], check=False)


if __name__ == "__main__":
    main()
