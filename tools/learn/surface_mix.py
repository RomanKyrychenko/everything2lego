"""Count real building sets' part classes by surface type -> src/everything2lego/data/surface_mix.json.

Each set's parts are put on the sampler's grid (motifs.grid_items); the set's own occupancy gives its surface
directions the way the target's are found (surface_dir.target_normals), so a real roof is typed "roof" whatever it
is built from.  See everything2lego.surface_mix.

    .venv-arm/bin/python tools/learn/surface_mix.py [--n 150]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "src")]
from motifs import DATA, grid_items, option_poses  # noqa: E402

from everything2lego.sampler import FULL, select_vocabulary  # noqa: E402
from everything2lego.surface_dir import target_normals  # noqa: E402
from everything2lego.surface_mix import CLASSES, DATA as OUT, TYPES, cell_types, item_type, part_class  # noqa: E402


def set_counts(items, cls: dict) -> np.ndarray:
    lo = np.min([it[2] for it in items], 0) - 3
    hi = np.max([np.add(it[2], it[3]) for it in items], 0) + 3
    occ = np.zeros(tuple(hi - lo), bool)
    for _, _, p, s in items:
        y, x, z = np.subtract(p, lo)
        occ[y:y + s[0], x:x + s[1], z:z + s[2]] = True
    bits = np.where(occ, FULL, np.uint64(0))
    types = cell_types(target_normals(bits, occ), occ.shape)
    n = np.zeros((len(TYPES), len(CLASSES)))
    for ld, _, p, s in items:
        y, x, z = np.subtract(p, lo)
        cells = [(yy, xx, zz) for yy in range(y, y + s[0]) for xx in range(x, x + s[1]) for zz in range(z, z + s[2])]
        n[item_type(types, cells), cls[ld]] += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="building"); ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--min-parts", type=int, default=200)
    a = ap.parse_args()
    poses = option_poses(a.kind)
    cls = {v.part.ldraw.lower(): part_class(v.category) for v in select_vocabulary(a.kind, 100000)}
    kinds = {r["id"]: r["kind"] for r in csv.DictReader(open(DATA / "analysis" / "set_kinds.csv"))}
    files = [f for f in sorted((DATA / "omr" / "files").glob("*.mpd")) if kinds.get(f.stem.split("_")[0]) == a.kind]
    total = np.zeros((len(TYPES), len(CLASSES))); used = 0
    for f in files:
        try:
            items = grid_items(f, poses)
        except Exception:                                                     # noqa: BLE001
            continue
        if len(items) < a.min_parts:
            continue
        total += set_counts(items, cls); used += 1
        if used >= a.n:
            break
    OUT.write_text(json.dumps({"source": "tools/learn/surface_mix.py", "kind": a.kind, "sets": used,
                               "types": TYPES, "classes": CLASSES, "counts": total.astype(int).tolist()}))
    print(f"{used} sets -> {OUT}")
    print(f"{'':10s}" + "".join(f"{c:>8s}" for c in CLASSES) + "   parts")
    for t, row in zip(TYPES, total):
        print(f"{t:10s}" + "".join(f"{v / max(1, row.sum()):8.2f}" for v in row) + f"   {int(row.sum())}")


if __name__ == "__main__":
    main()
