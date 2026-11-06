"""Builder labels from per-cell part evidence (cell_scorer.py's model), in the form sampler_demo --semantic reads:
the target's surface cells grouped into areas (colour-only grid regions, grid_regions.py, up to --area-max cells,
largest first, non-overlapping), and per area a weight for every part -- the mean of its cells' probabilities over
all parts (plain ones included, so a special part earns only as much as it beats a plain brick there), its --top
parts kept, scaled so the best is 1.  No rendering: the target, its normals, and real-set priors.

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
    from cell_scorer import FB, HB, features, parts_table
    from grid_regions import regions
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
    nrm = r["normal"][tuple(cells.T)]
    hb = np.minimum((cells[:, 0] / max(1, H - 1) * HB).astype(int), HB - 1)
    d = dict(true=np.zeros(len(cells), int), band=hb * FB + np.digitize(nrm[:, 0], [-0.8, -0.3, 0.3, 0.8]),
             colour=t.colour[tuple(cells.T)], tile=np.zeros((len(cells), P["look"].shape[1]), np.float32),
             seen=np.zeros(len(cells), bool))                # no picture: appearance terms are 0
    ev = dict(names=np.array([]), name_emb=np.zeros((0, P["look"].shape[1])))
    s = features(d, ev, P, M["prior_p"], M["prior_c"])[..., M["use"]] @ M["w"]
    p = np.exp(s - s.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    taken = np.zeros(len(cells), bool); areas = []
    for c in sorted((c for c in cands if len(c) <= a.area_max), key=len, reverse=True):
        if not taken[c].any():
            taken[c] = True; areas.append(c)
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
