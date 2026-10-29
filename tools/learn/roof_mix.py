"""Pitched roofs and flat tops of real building sets: their part mix and their patterns -> data/roof_mix.json,
data/motifs.json.

Most building sets have flat roofs, so statistics over whole sets say little about a pitched one.  Here only the
pitched-roof areas count (surface_mix.pitched_columns: a large connected area where the model's top rises 20-60
deg; the top plates there, surface_mix.roof_cells), and apart from them the flat tops (surface_mix.flat_cells: level
column tops -- flat roofs, ledges, the base): the class mix of the parts in each, and the 2-6-part patterns
(motifs.mine) all of whose parts are in it.  Patterns go into motifs.json with type 1 (roof) or 2 (flat top), so the
pattern move draws them on the target's own roof or flat tops.

    .venv-arm/bin/python tools/learn/roof_mix.py [--min-sets 3 --max-size 6]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "src")]
from motifs import OUT as MOTIFS, load_sets, mine, option_poses  # noqa: E402

from everything2lego.sampler import select_vocabulary  # noqa: E402
from everything2lego.surface_mix import CLASSES, flat_cells, part_class, roof_cells  # noqa: E402

OUT = HERE.parents[1] / "src/everything2lego/data/roof_mix.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="building"); ap.add_argument("--min-parts", type=int, default=150)
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--min-sets", type=int, default=3); ap.add_argument("--max-size", type=int, default=6)
    ap.add_argument("--top", type=int, default=800, help="per size and type")
    a = ap.parse_args()
    poses = option_poses(a.kind)
    cls = {v.part.ldraw.lower(): part_class(v.category) for v in select_vocabulary(a.kind, 100000)}
    counts = {1: np.zeros(len(CLASSES)), 2: np.zeros(len(CLASSES))}
    on = {1: [], 2: []}
    for _, items, nb in load_sets(a.kind, a.n, a.min_parts, poses):
        lo = np.min([it[2] for it in items], 0)
        hi = np.max([np.add(it[2], it[3]) for it in items], 0)
        occ = np.zeros(tuple(hi - lo), bool)
        for _, _, p, s in items:
            y, x, z = np.subtract(p, lo)
            occ[y:y + s[0], x:x + s[1], z:z + s[2]] = True
        for t, cells in ((1, roof_cells(occ)), (2, flat_cells(occ))):
            if not cells.any():
                continue
            m = []
            for ld, _, p, s in items:
                y, x, z = np.subtract(p, lo)
                m.append(bool(cells[y:y + s[0], x:x + s[1], z:z + s[2]].any()))
            for (ld, *_), r in zip(items, m):
                if r:
                    counts[t][cls[ld]] += 1
            on[t].append((items, nb, m))
    OUT.write_text(json.dumps({"source": "tools/learn/roof_mix.py", "kind": a.kind, "sets": len(on[1]),
                               "classes": CLASSES, "counts": counts[1].astype(int).tolist(),
                               "flat_sets": len(on[2]), "flat_counts": counts[2].astype(int).tolist()}))
    m = json.loads(MOTIFS.read_text())
    m["motifs"] = [x for x in m["motifs"] if x.get("type", 0) == 0]
    for t, name in ((1, "pitched roof"), (2, "flat top")):
        c = counts[t]
        print(f"{len(on[t])} sets with a {name}; parts {int(c.sum())}: " +
              "  ".join(f"{k} {v / c.sum():.2f}" for k, v in zip(CLASSES, c)))
        found = mine(on[t], a.min_sets, a.max_size)
        keep = [(kk, n) for k in sorted(found) for kk, n in found[k].most_common(a.top)]
        m["motifs"] += [{"parts": [list(p) for p in kk], "sets": n, "type": t} for kk, n in keep]
        print(f"  {name} patterns in >= {a.min_sets} sets: {len(keep)}")
    m["roof_sets"] = len(on[1]); m["flat_sets"] = len(on[2])
    MOTIFS.write_text(json.dumps(m))


if __name__ == "__main__":
    main()
