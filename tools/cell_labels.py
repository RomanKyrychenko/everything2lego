"""Builder labels from per-cell part evidence (cell_scorer.py's model), in the form sampler_demo --semantic reads:
the target's surface cells grouped into areas (colour-only grid regions, grid_regions.py, up to --area-max cells,
largest first, non-overlapping), and per area a weight for every part -- the mean of its cells' probabilities over
all parts over its mean over the whole target to the power --lift (1: a part likely everywhere -- a common brick
-- earns nothing for being likely here too; 0: the plain mean; the watermill builds chose 1 -- CELLP2 over 0.125 and 0.625, whose areas fill with 1 x 1 round bricks and lose every window; area_bench.py's rankings (best 0.125 at top 30, 0.625 at top 5) did not predict the renders),
its --top parts kept, scaled so the best is 1.  No rendering: the target, its normals and geometry (recess, extent, the
neighbours' parts), and real-set priors.

    tools/timed.sh cell-labels .venv-arm/bin/python tools/cell_labels.py examples/pixellabs-watermill-3425.glb \
        --model /Volumes/PRO-G40/lego_data/analysis/cell_bench/model.npz --out labels_cells.npz
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh"); ap.add_argument("--model", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--studs", type=int, default=32); ap.add_argument("--shell", type=int, default=3)
    ap.add_argument("--area-max", type=int, default=30); ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--regions", type=Path, default=None, help="region_evidence.py --mesh output for this mesh: the spot features from its rendered regions")
    ap.add_argument("--lift", type=float, default=1.0, help="area score = mean P(part) / (its mean over the target) ** lift")
    a = ap.parse_args()
    from cell_scorer import cell_probs, extent, parts_table, recess, surface_rows
    from grid_regions import partition, regions
    from sampler_demo import target_from_mesh
    from everything2lego.surface_dir import target_normals
    t0 = time.time()
    t = target_from_mesh(a.mesh, a.studs, a.shell)
    H, X, Z = t.want.shape
    r = dict(want=t.want, visible=t.visible, colour=t.colour,
             normal=target_normals(t.bits, t.want, sigma=1.5).reshape(H, X, Z, 3))
    cells, cands = regions(r, 0.0, 0.0)                      # colour only: the best grid regions (grid_regions.py)
    M = np.load(a.model, allow_pickle=True)
    P = parts_table()
    assert (M["ld"] == P["ld"]).all(), "model built on another part table"
    T = M["T"].item()
    d = surface_rows(r, P, geo=(recess(r["want"], r["normal"]), extent(r)))
    n = len(d["cells"])
    p = cell_probs(d, P, T, M["w1"], M["use"], M["w"], dict(np.load(a.regions)) if a.regions else None)
    lut = np.full((H, X, Z), -1); lut[tuple(d["cells"].T)] = np.arange(n)
    p = p[lut[tuple(cells.T)]]                               # regions' cells are surface cells with a normal
    areas = partition(cells, cands, a.area_max); taken = np.zeros(len(cells), bool)
    for c in areas:
        taken[c] = True
    cell_area = np.full((H, X, Z), -1, np.int32); cell_cat = np.full((H, X, Z), -1, np.int8)
    aff = np.zeros((len(areas), len(P["ld"])), np.float32)
    base = p.mean(0)
    for k, c in enumerate(areas):
        cell_area[tuple(cells[c].T)] = k; cell_cat[tuple(cells[c].T)] = 0
        m = p[c].mean(0) / base ** a.lift; keep = np.argsort(-m)[:a.top]
        aff[k, keep] = m[keep] / m[keep[0]]
    np.savez_compressed(a.out, cell_cat=cell_cat, categories=np.array(["any"]), cell_area=cell_area,
                        area_aff=aff.astype(np.float16), aff_parts=P["ld"])
    E = np.load("/Volumes/PRO-G40/lego_data/analysis/part_embed/google__siglip2-so400m-patch14-384.npz")
    top = np.bincount(aff.argmax(1), minlength=len(P["ld"]))
    print(f"{len(cells)} surface cells, {len(areas)} areas covering {taken.sum()} -> {a.out} ({time.time() - t0:.0f} s)")
    print("parts most often first in an area: " + "; ".join(f"{E['name'][i][:28]} {top[i]}" for i in np.argsort(-top)[:8]))
    sec = (aff > 0.5).sum(0); sec[np.argsort(-top)[:1]] = 0
    print("parts within half of the best in most areas: " + "; ".join(f"{E['name'][i][:28]} {sec[i]}" for i in np.argsort(-sec)[:8]))


if __name__ == "__main__":
    main()
