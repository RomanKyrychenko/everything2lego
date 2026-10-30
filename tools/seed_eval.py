"""Run one sampler recipe over several seeds and report each measure as mean +- spread, so a change between two
recipes can be told from seed noise.

    .venv-arm/bin/python tools/seed_eval.py NAME --seeds 4 [--jobs 4] -- <sampler_demo args after the output dir>
    .venv-arm/bin/python tools/seed_eval.py NAME --report            # the table again, from the runs on disk
    .venv-arm/bin/python tools/seed_eval.py --compare NAME1 NAME2    # side by side

Each seed's build goes to sampler_stat/NAME/s<seed>/; per seed it measures the shape error (ldr_to_mesh.compare,
% of model size), studded parts with an exposed top (all, and those whose top is in the lowest quarter: the base),
unsupported parts, part count and class shares, and parts in a colour the target does not have.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools/learn"), str(ROOT / "tools"), str(ROOT / "src")]
BASE = Path("/Volumes/PRO-G40/lego_data/analysis/sampler_stat")
MESH = ROOT / "examples/pixellabs-watermill-3425.glb"


def exposed(ldr: Path, kind: str = "building") -> dict:
    """Studded parts (plates, bricks) with a top cell nothing sits on, and part class shares, from the build's LDR."""
    from motifs import grid_items, option_poses
    from everything2lego.sampler import select_vocabulary
    from everything2lego.surface_mix import CLASSES, part_class
    from ldraw_read import flatten
    poses = option_poses(kind)
    cls = {v.part.ldraw.lower(): part_class(v.category) for v in select_vocabulary(kind, 100000)}
    items = grid_items(ldr, poses)
    occ = set()
    for _, _, (y, x, z), (h, w, d) in items:
        occ.update((yy, xx, zz) for yy in range(y, y + h) for xx in range(x, x + w) for zz in range(z, z + d))
    tops = [it[2][0] + it[3][0] for it in items]
    low = np.percentile(tops, 25) if tops else 0
    n = e = nl = el = 0
    counts = np.zeros(len(CLASSES))
    for ld, _, (y, x, z), (h, w, d) in items:
        c = cls.get(ld, 5)
        counts[c] += 1
        if c not in (2, 3):
            continue
        bare = any((y + h, xx, zz) not in occ for xx in range(x, x + w) for zz in range(z, z + d))
        n += 1; e += bare
        if y + h <= low:
            nl += 1; el += bare
    all_parts = len(flatten(ldr))
    return {"exposed_pct": 100 * e / max(1, n), "exposed_base_pct": 100 * el / max(1, nl),
            "on_grid": len(items) / max(1, all_parts),
            **{f"{c}_pct": 100 * v / max(1, counts.sum()) for c, v in zip(CLASSES, counts)}}


def off_colour(ldr: Path, info: dict) -> int:
    """Parts whose colour is not among the target's colours (info["target_colours"], LDraw codes)."""
    want = set(info.get("target_colours", []))
    if not want:
        return -1
    n = 0
    for line in ldr.read_text().splitlines():
        f = line.split()
        if len(f) >= 15 and f[0] == "1" and f[-1].lower().endswith(".dat"):
            n += int(f[1]) not in want
    return n


def measure(d: Path, studs: int) -> dict:
    from ldr_to_mesh import build_mesh, compare, input_mesh
    info = json.loads((d / "info.json").read_text())
    m = build_mesh(d / "model.ldr")
    shape = compare(m, input_mesh(MESH, studs), d / "model.glb")
    return {"energy": info["energy"], "parts": info["parts"], "unsupported": info["unsupported"],
            "shape_pct": shape["chamfer_pct"], **exposed(d / "model.ldr"), "off_colour": off_colour(d / "model.ldr", info)}


def run_seed(name: str, seed: int, args: list[str]) -> Path:
    d = BASE / name / f"s{seed}"
    if (d / "info.json").exists():
        return d
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "log.txt", "w") as log:
        subprocess.run([sys.executable, str(ROOT / "tools/sampler_demo.py"), str(MESH), str(d), "--seed", str(seed), *args],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    return d


def table(name: str) -> dict:
    rows = {}
    for d in sorted((BASE / name).glob("s*")):
        f = d / "measures.json"
        if f.exists():
            rows[d.name] = json.loads(f.read_text())
    return rows


def summary(rows: dict) -> dict:
    keys = next(iter(rows.values())).keys()
    return {k: (float(np.mean([r[k] for r in rows.values()])), float(np.std([r[k] for r in rows.values()])))
            for k in keys}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?"); ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--jobs", type=int, default=2, help="parallel runs; a 32-stud run peaks at ~5 GB"); ap.add_argument("--report", action="store_true")
    ap.add_argument("--compare", nargs="+")
    a, rest = ap.parse_known_args()
    rest = [r for r in rest if r != "--"]
    if a.compare:
        sums = {n: summary(table(n)) for n in a.compare}
        keys = next(iter(sums.values())).keys()
        print(f"{'':18s}" + "".join(f"{n:>22s}" for n in a.compare))
        for k in keys:
            print(f"{k:18s}" + "".join(f"{s[k][0]:14.3f} ± {s[k][1]:<5.3f}" for s in sums.values()))
        return
    if not a.report:
        studs = int(rest[rest.index("--studs") + 1]) if "--studs" in rest else 24
        (BASE / a.name).mkdir(parents=True, exist_ok=True)
        (BASE / a.name / "recipe.json").write_text(json.dumps(rest))
        with ThreadPoolExecutor(a.jobs) as ex:
            dirs = list(ex.map(lambda s: run_seed(a.name, s, rest), range(a.seeds)))
        for d in dirs:
            (d / "measures.json").write_text(json.dumps(measure(d, studs)))
    rows = table(a.name)
    for s, r in rows.items():
        print(s, "  ".join(f"{k} {v:.3f}" for k, v in r.items()))
    print("mean ± sd", "  ".join(f"{k} {m:.3f}±{sd:.3f}" for k, (m, sd) in summary(rows).items()))


if __name__ == "__main__":
    main()
