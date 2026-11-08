"""Candidate regions on the sampler's stud grid instead of mesh faces: a few thousand surface cells, merged as selective
search merges an image, by colour and shape -- no rendering.

Leaves are single surface cells (wanted, visible, with a surface normal).  Adjacent regions (26-neighbourhood) merge
cheapest first; cost = colour distance (Lab / 20) + w_normal * (1 - cos between mean normals) + w_curv * |curvature
difference| + size (both small first).  Every merge of --min-cells..--max-cells that grew its larger half by half is a
candidate (as semantic_labels.regions).  A cell's curvature is 1 - the mean cosine of its normal with its neighbours'.

Evaluated on cell_bench.py's truth: each true part instance's best candidate overlap (IoU over surface cells), for
plain and special parts, the number of candidates and the seconds.

    tools/timed.sh grid-regions .venv-arm/bin/python tools/grid_regions.py /Volumes/PRO-G40/lego_data/analysis/cell_bench --source gen
"""
from __future__ import annotations

import argparse
import heapq
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]

OFFS = np.array([(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if (a, b, c) != (0, 0, 0)])


def surface(r: dict) -> np.ndarray:
    """(n, 3) surface cells (y, x, z): wanted, visible, with a normal."""
    r["normal"] = r["normal"].reshape(*r["want"].shape, 3)    # cell_bench files before the reshape stored it flat
    n = r["normal"].astype(np.float32)
    return np.argwhere(r["want"] & r["visible"] & (np.linalg.norm(n, axis=-1) > 0.5))


def neighbours(cells: np.ndarray, shape) -> tuple[np.ndarray, np.ndarray]:
    """Pairs (i < j) of 26-adjacent cells, as indices into `cells`."""
    lut = np.full(shape, -1, np.int64); lut[tuple(cells.T)] = np.arange(len(cells))
    a, b = [], []
    for o in OFFS[:13]:                                      # half the offsets: each pair once
        q = cells + o
        ok = ((q >= 0) & (q < np.array(shape))).all(1)
        j = np.full(len(cells), -1); j[ok] = lut[tuple(q[ok].T)]
        m = j >= 0
        a.append(np.flatnonzero(m)); b.append(j[m])
    return np.concatenate(a), np.concatenate(b)


def regions(r: dict, w_normal: float = 2.0, w_curv: float = 2.0, min_cells: int = 2, max_cells: int = 400) -> tuple:
    """-> (cells (n, 3), [candidate cell-index arrays])."""
    from everything2lego.colors import _palette_lab
    cells = surface(r)
    nrm = r["normal"].astype(np.float32)[tuple(cells.T)]
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    lab = _palette_lab()[np.maximum(r["colour"][tuple(cells.T)], 0)]
    ia, ib = neighbours(cells, r["want"].shape)
    deg = np.bincount(np.r_[ia, ib], minlength=len(cells))
    dots = (nrm[ia] * nrm[ib]).sum(1)
    curv = 1 - (np.bincount(ia, dots, len(cells)) + np.bincount(ib, dots, len(cells))) / np.maximum(deg, 1)
    m = len(cells)
    size = list(np.ones(m)); col = list(lab); nsum = list(nrm.copy()); cv = list(curv)
    nb = [set() for _ in range(m)]
    for x, y in zip(ia, ib):
        nb[x].add(y); nb[y].add(x)
    members = {i: [i] for i in range(m)}
    total = float(m)

    def cost(x, y):
        nx, ny = nsum[x] / np.linalg.norm(nsum[x]), nsum[y] / np.linalg.norm(nsum[y])
        return (np.linalg.norm(col[x] - col[y]) / 20 + w_normal * (1 - float(nx @ ny))
                + w_curv * abs(cv[x] - cv[y]) + (size[x] + size[y]) / total)
    heap = [(cost(x, y), x, y) for x in range(m) for y in nb[x] if x < y]
    heapq.heapify(heap)
    alive, out = set(range(m)), []
    while heap:
        _, x, y = heapq.heappop(heap)
        if x not in alive or y not in alive:
            continue
        z = len(size)
        s = size[x] + size[y]
        size.append(s); col.append((col[x] * size[x] + col[y] * size[y]) / s); nsum.append(nsum[x] + nsum[y])
        cv.append((cv[x] * size[x] + cv[y] * size[y]) / s)
        members[z] = members.pop(x) + members.pop(y)
        nb.append((nb[x] | nb[y]) - {x, y}); alive -= {x, y}; alive.add(z)
        for c in nb[z]:
            nb[c] -= {x, y}; nb[c].add(z)
            heapq.heappush(heap, (cost(z, c), z, c))
        if min_cells <= s <= max_cells and s >= 1.5 * max(size[x], size[y]):
            out.append(np.array(members[z]))
    return cells, out


def partition(cells: np.ndarray, cands: list, area_max: int = 30) -> list:
    """Non-overlapping areas from the candidates: up to `area_max` cells, largest first, none sharing a cell."""
    taken = np.zeros(len(cells), bool); areas = []
    for c in sorted((c for c in cands if len(c) <= area_max), key=len, reverse=True):
        if not taken[c].any():
            taken[c] = True; areas.append(c)
    return areas


def best_iou(cells: np.ndarray, cands: list, inst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per true instance (on the surface cells): its cell count and the best IoU of any candidate."""
    u, ci = np.unique(inst, return_inverse=True)
    n_i = np.bincount(ci, minlength=len(u)).astype(float)
    best = np.zeros(len(u))
    for c in cands:
        k = np.bincount(ci[c], minlength=len(u))
        hit = np.flatnonzero(k)
        best[hit] = np.maximum(best[hit], k[hit] / (len(c) + n_i[hit] - k[hit]))
    return u, n_i, best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bench", type=Path); ap.add_argument("--source", default="gen")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--variants", default="colour:0:0,normal:2:0,normal+curv:2:2",
                    help="name:w_normal:w_curv,...")
    a = ap.parse_args()
    E = np.load("/Volumes/PRO-G40/lego_data/analysis/part_embed/google__siglip2-so400m-patch14-384.npz")
    from part_match_bench import BASIC
    catof = dict(zip(E["ldraw"], E["category"]))
    files = sorted(a.bench.glob(f"[!.]*.{a.source}.npz"))[:a.n]
    for v in a.variants.split(","):
        name, wn, wc = v.split(":")
        t0 = time.time(); ncand = 0; rows = []
        for f in files:
            r = dict(np.load(f, allow_pickle=True))
            cells, cands = regions(r, float(wn), float(wc))
            ncand += len(cands)
            inst = r["inst"][tuple(cells.T)]; ok = (inst >= 0) & (r["share"][tuple(cells.T)] >= 0.5)
            inst = np.where(ok, inst, -1)
            u, n_i, best = best_iou(cells, cands, inst)
            for q, n, b in zip(u, n_i, best):
                if q < 0 or n < 2:
                    continue
                p = str(r["inst_part"][q]) if q < len(r["inst_part"]) else ""
                rows.append((n, b, catof.get(p, "") not in BASIC and p in catof))
        secs = time.time() - t0
        n, b, sp = (np.array(x) for x in zip(*rows))
        line = lambda m: f"inst {m.sum():5d}  mean IoU {b[m].mean():.3f}  IoU>=0.5 {(b[m] >= 0.5).mean():.3f}  cell-weighted {np.average(b[m], weights=n[m]):.3f}"
        print(f"{name:12s} {len(files)} sets, {ncand / len(files):6.0f} candidates/set, {secs / len(files):5.1f} s/set")
        print(f"   plain    {line(~sp)}\n   special  {line(sp)}", flush=True)


if __name__ == "__main__":
    main()
