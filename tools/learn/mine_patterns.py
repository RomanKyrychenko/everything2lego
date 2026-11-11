"""Pattern mining on the OMR dataset: find small groups of placements
that appear with the same relative pose across many real sets.  A
"pattern" is a connected cluster of 2-6 placements, keyed by
(part_ldraw, yaw, dy, dx, dz) tuples normalised to the first part.
Patterns that appear in MIN_SETS+ distinct sets become the catalogue
option 3's grammar / template-match energy will build on.

Why: a sampler that doesn't know "a wall with evenly spaced studded
windows" as a *pattern* will never produce one by chance.  The
catalogue extracts those patterns from real sets directly.

    python tools/learn/mine_patterns.py [--size 2] [--k 3] [--min-sets 5]

Writes $LEGO_DATA/analysis/patterns_sizeN.json: each entry is
{a, b, ...: (ldraw, yaw)}, {rel: [(dy, dx, dz), ...]}, models, count.

Honest limits:
- Clusters bigger than --size 3 blow up combinatorially; the default
  mines pairs.  Call out --size 3 explicitly if you have the time.
- Patterns are POSE-exact.  Masonry-offset bricks that happen one way
  in Castle A and the mirror way in Castle B count as two patterns.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATASET = Path("/Volumes/PRO-G40/lego_data/analysis/omr_dataset")
OUT = Path("/Volumes/PRO-G40/lego_data/analysis/patterns_size{sz}.json")


def _load(f: Path):
    d = np.load(f, allow_pickle=True)
    return (d["y"], d["x"], d["z"], d["w"], d["d"], d["h"],
            d["opt_yaw"], d["opt_ldraw"])


def _pairs_in(ys, xs, zs, ws, ds, hs, yaws, ldraws, max_dist: int) -> list[tuple]:
    """All (i, j) pairs where j's bbox is within max_dist cells of i's."""
    n = len(ys)
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            dy, dx, dz = int(ys[j] - ys[i]), int(xs[j] - xs[i]), int(zs[j] - zs[i])
            if abs(dy) > max_dist or abs(dx) > max_dist or abs(dz) > max_dist:
                continue
            key = ((str(ldraws[i]), int(yaws[i])), (str(ldraws[j]), int(yaws[j])), dy, dx, dz)
            out.append(key)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=2, choices=(2,),
                     help="cluster size; only 2 supported in this first pass")
    ap.add_argument("--k", type=int, default=3, help="neighbour-pair distance cap (cells)")
    ap.add_argument("--min-sets", type=int, default=5, help="require pattern in >= this many distinct sets")
    a = ap.parse_args()

    files = sorted(f for f in DATASET.glob("*.npz") if not f.name.startswith("._"))
    if not files:
        raise SystemExit(f"no dataset .npz files under {DATASET}; run omr_dataset.py first")
    print(f"mining size-{a.size} patterns across {len(files)} sets ...", flush=True)

    seen: Counter = Counter()
    pair_sets: dict[tuple, set] = {}
    for i, f in enumerate(files):
        try:
            data = _load(f)
        except Exception:
            continue
        pairs = _pairs_in(*data, max_dist=a.k)
        unique_in_set = set(pairs)
        for p in unique_in_set:
            seen[p] = seen.get(p, 0) + 1
        for p in unique_in_set:
            pair_sets.setdefault(p, set()).add(f.stem)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(files)}", flush=True)

    kept = [(k, n) for k, n in seen.most_common() if len(pair_sets[k]) >= a.min_sets]
    print(f"patterns: {len(kept)} seen in {a.min_sets}+ sets (of {len(seen)} total)", flush=True)
    out_rows = []
    for key, count in kept:
        (ldraw_a, yaw_a), (ldraw_b, yaw_b), dy, dx, dz = key
        out_rows.append({
            "a": {"ldraw": ldraw_a, "yaw": yaw_a},
            "b": {"ldraw": ldraw_b, "yaw": yaw_b},
            "offset": {"dy": dy, "dx": dx, "dz": dz},
            "models": len(pair_sets[key]),
            "count": count,
        })
    out_path = Path(str(OUT).format(sz=a.size))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_rows, indent=1))
    print(f"-> {out_path}")
    for r in out_rows[:10]:
        print(f"  {r['a']['ldraw']:>10} y{r['a']['yaw']:<3} + {r['b']['ldraw']:>10} y{r['b']['yaw']:<3}  "
              f"(+{r['offset']['dy']:+d}p {r['offset']['dx']:+d}s {r['offset']['dz']:+d}s)  "
              f"{r['models']} models")


if __name__ == "__main__":
    main()
