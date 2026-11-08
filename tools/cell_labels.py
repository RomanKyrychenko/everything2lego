"""Builder labels from per-cell part evidence (cell_scorer.py's model), in the form sampler_demo --semantic reads:
the target's surface cells grouped into areas (colour-only grid regions, grid_regions.py, up to --area-max cells,
largest first, non-overlapping), and per area a weight for every part -- the mean of its cells' probabilities over
all parts (plain ones included, so a special part earns only as much as it beats a plain brick there), its --top
parts kept, scaled so the best is 1.  No rendering: the target, its normals and geometry (recess, extent, the
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
    a = ap.parse_args()
    from cell_scorer import context, extent, features, parts_table, recess, surface_rows
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
    T = M["T"].item(); w1 = M["w1"]
    d = surface_rows(r, P, geo=(recess(r["want"], r["normal"]), extent(r)))
    n, D = len(d["cells"]), P["look"].shape[1]
    d["sel"] = np.arange(n)
    ctx = context(d, P, T, w1) if len(w1) else None
    ev = dict(names=np.array([]), name_emb=np.zeros((0, D)))
    p = np.empty((n, len(P["ld"])), np.float32)
    for s0 in range(0, n, 500):                              # no picture: appearance terms are 0
        rows = np.arange(s0, min(n, s0 + 500)); k = len(rows)
        q = dict(d, sel=rows, true=np.zeros(k, int), band=d["band_all"][rows], colour=d["colour_all"][rows],
                 tile=np.zeros((k, D), np.float32), seen=np.zeros(k, bool))
        z = features(q, ev, P, *T["band"], T, ctx=None if ctx is None else ctx[rows])[..., M["use"]] @ M["w"]
        z = np.exp(z - z.max(1, keepdims=True)); p[rows] = z / z.sum(1, keepdims=True)
    lut = np.full((H, X, Z), -1); lut[tuple(d["cells"].T)] = np.arange(n)
    p = p[lut[tuple(cells.T)]]                               # regions' cells are surface cells with a normal
    areas = partition(cells, cands, a.area_max); taken = np.zeros(len(cells), bool)
    for c in areas:
        taken[c] = True
    cell_area = np.full((H, X, Z), -1, np.int32); cell_cat = np.full((H, X, Z), -1, np.int8)
    aff = np.zeros((len(areas), len(P["ld"])), np.float32)
    for k, c in enumerate(areas):
        cell_area[tuple(cells[c].T)] = k; cell_cat[tuple(cells[c].T)] = 0
        m = p[c].mean(0); keep = np.argsort(-m)[:a.top]
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
