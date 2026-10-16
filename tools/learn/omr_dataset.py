"""Extract OMR models into a (voxel_target, placement_sequence, kind)
dataset — the common foundation for both the supervised placement
model (option 2) and the pattern-mining pass (option 3).

For each OMR .mpd/.ldr:
  1. flatten() -> list of (ldraw_name, colour, 4x4 M)
  2. map each to our Placement (same inverse as retrieve_adapt.py:
     drops non-vocab parts and tilted rotations)
  3. voxelise the placement list into a (H, X, Z) occupancy grid
  4. kind from the KINDS CSV ($LEGO_DATA/analysis/set_kinds.csv) when present

Output: one .npz per model under $LEGO_DATA/analysis/omr_dataset/:
    voxels     (H, X, Z) bool
    y, x, z    (N,) int32 — placement positions
    opt_ldraw  (N,) object  — part ldraw name
    opt_yaw    (N,) int32   — placement rotation_deg
    opt_colour (N,) int32   — palette index
    opt_w, opt_d, opt_h  (N,) int32 — rotated footprint + height
    kind       str          — object kind

    python tools/learn/omr_dataset.py [--n 50] [--only PATTERN]
"""

from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools/set_study"))
sys.path.insert(0, str(ROOT / "tools/learn"))

OMR = Path("/Volumes/PRO-G40/lego_data/omr/files")
OUT_DIR = Path("/Volumes/PRO-G40/lego_data/analysis/omr_dataset")


def _extract(ldr_path: Path, by_ldraw, _ROT, LDU_STUD=20, LDU_PLATE=8):
    """-> (y, x, z, w, d, h, yaw, ldraw, colour) arrays, or None on parse error."""
    from ldraw_read import flatten
    from retrieve_adapt import _best_yaw, _ldr_colour_to_palette
    try:
        parts = flatten(ldr_path)
    except Exception:
        return None
    ys, xs, zs, ws, ds, hs, yaws, ldraws, colours = [], [], [], [], [], [], [], [], []
    for p in parts:
        v = by_ldraw.get(p.name)
        if v is None:
            continue
        R = p.M[:3, :3]
        matched = _best_yaw(R)
        if matched is None:
            continue
        rot = (matched - v.part.yaw) % 360
        if rot % 90 != 0:
            continue
        fw, fd = v.part.footprint
        w_world, d_world = (fd, fw) if rot in (90, 270) else (fw, fd)
        m = _ROT[matched]
        ox, oy, oz = v.part.origin
        x_prime = p.M[0, 3] + m[0] * ox + m[2] * oz
        z_prime = p.M[2, 3] + m[6] * ox + m[8] * oz
        y_prime = p.M[1, 3] + oy
        x_stud = int(round((x_prime / 10 - w_world + 1) / 2))
        z_stud = int(round((z_prime / 10 - d_world + 1) / 2))
        y_plate = int(round(-y_prime / 8 - v.part.height_units))
        colour_idx = _ldr_colour_to_palette(p.colour)
        if colour_idx < 0:
            colour_idx = 0
        ys.append(y_plate); xs.append(x_stud); zs.append(z_stud)
        ws.append(w_world); ds.append(d_world); hs.append(v.part.height_units)
        yaws.append(rot); ldraws.append(p.name); colours.append(colour_idx)
    return (np.array(ys, np.int32), np.array(xs, np.int32), np.array(zs, np.int32),
            np.array(ws, np.int32), np.array(ds, np.int32), np.array(hs, np.int32),
            np.array(yaws, np.int32), np.array(ldraws, object), np.array(colours, np.int32))


def _voxelise(ys, xs, zs, ws, ds, hs) -> np.ndarray:
    """(H, X, Z) bool occupancy from placement arrays (model's own bbox)."""
    if not len(ys):
        return np.zeros((1, 1, 1), bool)
    y0 = int(ys.min()); y1 = int((ys + hs).max())
    x0 = int(xs.min()); x1 = int((xs + ws).max())
    z0 = int(zs.min()); z1 = int((zs + ds).max())
    H, X, Z = y1 - y0, x1 - x0, z1 - z0
    occ = np.zeros((H, X, Z), bool)
    for i in range(len(ys)):
        for cy in range(ys[i] - y0, ys[i] - y0 + hs[i]):
            for cx in range(xs[i] - x0, xs[i] - x0 + ws[i]):
                for cz in range(zs[i] - z0, zs[i] - z0 + ds[i]):
                    if 0 <= cy < H and 0 <= cx < X and 0 <= cz < Z:
                        occ[cy, cx, cz] = True
    return occ


_G: dict = {}


def _init():
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "src"))
    _sys.path.insert(0, str(ROOT / "tools"))
    _sys.path.insert(0, str(ROOT / "tools/set_study"))
    from everything2lego.sampler import vocabulary
    from everything2lego.io_ldraw import _ROT
    _G["by_ldraw"] = {v.part.ldraw: v for v in vocabulary()}
    _G["ROT"] = _ROT


def _one(args):
    name, ldr_path, kind, out_dir = args
    r = _extract(ldr_path, _G["by_ldraw"], _G["ROT"])
    if r is None or len(r[0]) == 0:
        return (name, 0)
    ys, xs, zs, ws, ds, hs, yaws, ldraws, colours = r
    # normalise origin to (0, 0, 0)
    y0, x0, z0 = int(ys.min()), int(xs.min()), int(zs.min())
    ys, xs, zs = ys - y0, xs - x0, zs - z0
    occ = _voxelise(ys, xs, zs, ws, ds, hs)
    np.savez(out_dir / f"{name}.npz", voxels=occ, y=ys, x=xs, z=zs, w=ws, d=ds, h=hs,
             opt_yaw=yaws, opt_ldraw=ldraws, opt_colour=colours, kind=np.array(kind, object))
    return (name, int(len(ys)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="limit to first N models (0 = all)")
    ap.add_argument("--only", default=None, help="process only files whose name contains this")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    a = ap.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    kinds_csv = Path("/Volumes/PRO-G40/lego_data/analysis/set_kinds.csv")
    kind_of: dict[str, str] = {}
    if kinds_csv.exists():
        df = pd.read_csv(kinds_csv).drop_duplicates("id", keep="last")
        kind_of = dict(zip(df.id, df.kind))

    files = sorted(OMR.glob("*.mpd")) + sorted(OMR.glob("*.ldr"))
    if a.only:
        files = [f for f in files if a.only in f.name]
    if a.n:
        files = files[:a.n]
    jobs = [(f.stem, f, kind_of.get(f.stem) or kind_of.get(f.stem.split("_")[0]) or "object", a.out)
             for f in files]
    print(f"extracting {len(jobs)} OMR models -> {a.out}", flush=True)
    extracted = 0
    with Pool(max(1, (os.cpu_count() or 2) - 1), initializer=_init) as pool:
        for i, (name, n_parts) in enumerate(pool.imap_unordered(_one, jobs, chunksize=4)):
            extracted += (n_parts > 0)
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(jobs)} ({extracted} with parts)", flush=True)
    print(f"done: {extracted}/{len(jobs)} models extracted")


if __name__ == "__main__":
    main()
