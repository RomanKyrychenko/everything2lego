"""Pitched roofs of real building sets: their part mix and their patterns -> data/roof_mix.json, data/motifs.json.

Most building sets have flat roofs, so statistics over whole sets say little about a pitched one.  Here only the
pitched-roof areas count (surface_mix.pitched_columns: a large connected area where the model's top rises 20-60
deg; the top plates there, surface_mix.roof_cells): the class mix of the parts in them, and the 2-3-part patterns
(motifs.py) all of whose parts are in them.  Roof patterns are added to motifs.json with type 1, so the pattern move
draws them on the target's own pitched roof.

    .venv-arm/bin/python tools/learn/roof_mix.py [--min-sets 5]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "src")]
from motifs import DATA, OUT as MOTIFS, grid_items, key, option_poses, overlaps, touching  # noqa: E402

from everything2lego.sampler import select_vocabulary  # noqa: E402
from everything2lego.surface_mix import CLASSES, part_class, roof_cells  # noqa: E402

OUT = HERE.parents[1] / "src/everything2lego/data/roof_mix.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="building"); ap.add_argument("--min-parts", type=int, default=150)
    ap.add_argument("--min-sets", type=int, default=5); ap.add_argument("--top", type=int, default=1500)
    a = ap.parse_args()
    poses = option_poses(a.kind)
    cls = {v.part.ldraw.lower(): part_class(v.category) for v in select_vocabulary(a.kind, 100000)}
    kinds = {r["id"]: r["kind"] for r in csv.DictReader(open(DATA / "analysis" / "set_kinds.csv"))}
    files = [f for f in sorted((DATA / "omr" / "files").glob("*.mpd")) if kinds.get(f.stem.split("_")[0]) == a.kind]
    counts = np.zeros(len(CLASSES)); sets = Counter(); used = 0
    for f in files:
        try:
            items = grid_items(f, poses)
        except Exception:                                                     # noqa: BLE001
            continue
        if len(items) < a.min_parts:
            continue
        lo = np.min([it[2] for it in items], 0)
        hi = np.max([np.add(it[2], it[3]) for it in items], 0)
        occ = np.zeros(tuple(hi - lo), bool)
        for _, _, p, s in items:
            y, x, z = np.subtract(p, lo)
            occ[y:y + s[0], x:x + s[1], z:z + s[2]] = True
        rc = roof_cells(occ)
        if not rc.any():
            continue
        used += 1
        on = []
        for ld, _, p, s in items:
            y, x, z = np.subtract(p, lo)
            on.append(bool(rc[y:y + s[0], x:x + s[1], z:z + s[2]].any()))
        for (ld, *_), r in zip(items, on):
            if r:
                counts[cls[ld]] += 1
        nb = touching(items)
        seen = set()
        for i in (i for i in range(len(items)) if on[i]):
            ns = sorted(j for j in nb[i] if on[j])
            for j in ns:
                if j > i:
                    seen.add(key(items, (i, j)))
            for p in range(len(ns)):
                for q in range(p + 1, len(ns)):
                    m = (i, ns[p], ns[q])
                    if not overlaps(items, m):
                        seen.add(key(items, m))
        for k in seen:
            sets[k] += 1
    OUT.write_text(json.dumps({"source": "tools/learn/roof_mix.py", "kind": a.kind, "sets": used,
                               "classes": CLASSES, "counts": counts.astype(int).tolist()}))
    keep = [(k, n) for k, n in sets.most_common() if n >= a.min_sets][:a.top]
    m = json.loads(MOTIFS.read_text())
    m["motifs"] = [x for x in m["motifs"] if x.get("type", 0) == 0] + \
                  [{"parts": [list(p) for p in k], "sets": n, "type": 1} for k, n in keep]
    m["roof_sets"] = used
    MOTIFS.write_text(json.dumps(m))
    print(f"{used} sets with a pitched roof; roof parts {int(counts.sum())}: " +
          "  ".join(f"{c} {v / counts.sum():.2f}" for c, v in zip(CLASSES, counts)))
    print(f"roof patterns in >= {a.min_sets} sets: {len(keep)} -> {MOTIFS}")
    for k, n in keep[:12]:
        print(f"  {n:3d} sets  " + "  ".join(f"{p[0]}@{p[1]}+{tuple(p[2:])}" for p in k))


if __name__ == "__main__":
    main()
