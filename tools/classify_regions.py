"""Classify each occupied voxel of a mesh target as an architectural
region: base / wall / roof / column / interior / other.  First step
toward item 4 (pattern templates) — the classifier gives the templates
a per-cell anchor to key off of.  By itself it also makes a useful
input to tools/sampler_demo.py --segmentation (region_kind drives the
sampler's per-cell object kind, if the labels map to known kinds).

    python tools/classify_regions.py mesh.glb --out regions.npz
        [--studs 32]  [--kind building]

Classification rules (simple, deterministic):

- base       : y_plate <= 1 (bottom one or two plates)
- interior   : all 6 axis-aligned neighbours are also occupied
- roof       : the +y neighbour is empty AND at least one of the
               four diagonal (+y, ±x/±z) neighbours is empty too
               (an unshaded cell on a slanted/flat top surface)
- wall       : the +y neighbour is occupied AND the cell sits on an
               outer face (either +x, -x, +z or -z neighbour is empty)
- column     : narrow (max 2 studs) connected vertical cluster of
               wall cells with the same (x, z) footprint
- other      : everything the above missed (overhangs, soffits, etc.)

The output has the same shape as segment_photo/segment_mesh's output:
`label` is a (H, X, Z) int array of region ids and `kind_of` is a
(region_id, region_kind) object array.  Region ids start at 0 per
label; the mapping is {0: base, 1: wall, 2: roof, 3: column, 4:
interior, 5: other}.  By default region_kind for each region is
`"building"`; override via --kinds "wall=object,roof=object,..." if the
sampler should pick different reference statistics per region.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

LABEL = {"base": 0, "wall": 1, "roof": 2, "column": 3, "interior": 4, "other": 5}


def classify(occ: np.ndarray) -> np.ndarray:
    """(H, X, Z) int array of per-voxel region labels."""
    H, X, Z = occ.shape
    out = np.full(occ.shape, -1, np.int64)
    # 6-neighbour pads
    nx_pos = np.zeros_like(occ); nx_pos[:, :-1, :] = occ[:, 1:, :]
    nx_neg = np.zeros_like(occ); nx_neg[:, 1:, :] = occ[:, :-1, :]
    nz_pos = np.zeros_like(occ); nz_pos[:, :, :-1] = occ[:, :, 1:]
    nz_neg = np.zeros_like(occ); nz_neg[:, :, 1:] = occ[:, :, :-1]
    ny_pos = np.zeros_like(occ); ny_pos[:-1, :, :] = occ[1:, :, :]
    ny_neg = np.zeros_like(occ); ny_neg[1:, :, :] = occ[:-1, :, :]
    # diagonal (+y, ±x/±z) for roof test
    diag = np.zeros_like(occ)
    for dy in (1,):
        for dx in (-1, 1):
            s = np.zeros_like(occ)
            yr, xr = slice(max(0, -dy), min(H, H - dy)), slice(max(0, -dx), min(X, X - dx))
            s[yr, xr, :] = occ[max(0, dy):min(H, H + dy), max(0, dx):min(X, X + dx), :]
            diag |= ~s & occ
        for dz in (-1, 1):
            s = np.zeros_like(occ)
            yr, zr = slice(max(0, -dy), min(H, H - dy)), slice(max(0, -dz), min(Z, Z - dz))
            s[yr, :, zr] = occ[max(0, dy):min(H, H + dy), :, max(0, dz):min(Z, Z + dz)]
            diag |= ~s & occ

    interior = occ & nx_pos & nx_neg & ny_pos & ny_neg & nz_pos & nz_neg
    out[interior] = LABEL["interior"]

    outer_side = occ & (~nx_pos | ~nx_neg | ~nz_pos | ~nz_neg)
    # roof: +y empty AND at least one diagonal above-neighbour is empty too
    roof = occ & ~ny_pos & diag
    out[roof & (out < 0)] = LABEL["roof"]
    # wall: outer side with a part above (not roof)
    wall = outer_side & ny_pos & (out < 0)
    out[wall] = LABEL["wall"]

    # base: bottom 1-2 plates get base label unless already interior/roof
    base_mask = np.zeros_like(occ, bool)
    base_mask[:min(2, H), :, :] = occ[:min(2, H), :, :]
    out[base_mask & (out < 0)] = LABEL["base"]
    # also reclassify very-low walls/roofs as base
    out[base_mask & ((out == LABEL["wall"]) | (out == LABEL["roof"]))] = LABEL["base"]

    out[occ & (out < 0)] = LABEL["other"]

    # column: among WALL cells, find (x, z) footprints that are narrow and
    # extend vertically through many layers with few neighbours
    wall_mask = out == LABEL["wall"]
    col_xz_mask = np.zeros((X, Z), bool)
    xz_count = wall_mask.sum(axis=0)
    # a (x, z) footprint is column-like if it has >=3 wall cells vertically AND
    # the 3x3 neighbourhood around it has few other wall cells
    for x in range(X):
        for z in range(Z):
            if xz_count[x, z] < 3:
                continue
            x0, x1 = max(0, x - 1), min(X, x + 2)
            z0, z1 = max(0, z - 1), min(Z, z + 2)
            around = (xz_count[x0:x1, z0:z1] > 0).sum() - 1  # exclude self
            if around <= 2:
                col_xz_mask[x, z] = True
    for x, z in zip(*np.nonzero(col_xz_mask)):
        out[wall_mask[:, x, z], x, z] = LABEL["column"]

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--studs", type=int, default=32)
    ap.add_argument("--kinds", default=None,
                    help='override per-region kind, e.g. "roof=object,wall=building"')
    a = ap.parse_args()

    from everything2lego.semantic_v2 import MeshTarget
    from everything2lego.sampler import known_kinds
    mt = MeshTarget.from_path(a.mesh, max_studs=a.studs, n_samples=40000, min_island_cells=8)
    occ = mt.voxel_grid.palette_idx >= 0
    label = classify(occ)
    counts = {name: int((label == idx).sum()) for name, idx in LABEL.items()}
    print(f"classified {int(occ.sum())} voxels: " +
          ", ".join(f"{name}={counts[name]}" for name in LABEL))

    region_kinds = {idx: "building" for idx in LABEL.values()}  # default
    valid = known_kinds()
    if a.kinds:
        for pair in a.kinds.split(","):
            name, k = pair.split("=")
            name, k = name.strip(), k.strip()
            if name not in LABEL:
                raise SystemExit(f"unknown region name {name!r}; valid: {sorted(LABEL)}")
            if k not in valid:
                raise SystemExit(f"unknown kind {k!r}; valid: {sorted(valid)}")
            region_kinds[LABEL[name]] = k

    kind_of = np.array([[str(idx), region_kinds[idx]] for idx in LABEL.values()
                         if counts[next(n for n, i in LABEL.items() if i == idx)] > 0],
                        dtype=object)
    np.savez(a.out, label=label, kind_of=kind_of)
    print(f"-> {a.out} ({len([_ for _ in kind_of])} non-empty regions)")


if __name__ == "__main__":
    main()
