"""Rebuild the worst regions of a finished build on their own.

A region is a connected patch of high local error (the native error map: shape, support and direction, what a
better cover of the patch can fix), boxed.  The target in that box is the mesh as it should be there; it is sampled
again with steps of its own, inside the whole build with every cell outside the box locked (Target.lock), so the
new parts sit on and under the parts they meet and every step is judged by the whole build's energy.  The new
parts are kept when that energy falls.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy import ndimage

FIXABLE = (0, 2, 3)          # native.ERR_LAYERS: shape, support, direction


def find_regions(err_cell: np.ndarray, want: np.ndarray, k: int = 8, q: float = 0.9,
                 max_side: tuple = (36, 16, 16)) -> list[tuple[slice, slice, slice]]:
    """The k worst patches, worst first: cells above the q-quantile of fixable error, joined across a cell, each
    boxed (at most max_side cells along y, x, z -- plates, studs, studs: 36 plates is 12 bricks -- around its worst
    cell)."""
    e = np.where(want, err_cell[..., list(FIXABLE)].sum(-1), 0.0)
    if not (e > 0).any():
        return []
    hot = ndimage.binary_dilation(e > np.quantile(e[e > 0], q)) & want
    lab, n = ndimage.label(hot)
    if n == 0:
        return []
    total = ndimage.sum(e, lab, np.arange(1, n + 1))
    worst = ndimage.maximum_position(e, lab, np.arange(1, n + 1))
    out = []
    for i in np.argsort(-total)[:k]:
        sl = ndimage.find_objects(lab == i + 1)[0]
        box = []
        for s, c, size, m in zip(sl, worst[i], want.shape, max_side):
            a, b = max(0, s.start - 1), min(size, s.stop + 1)
            if b - a > m:
                a = max(0, min(c - m // 2, size - m)); b = a + m
            box.append(slice(a, b))
        out.append(tuple(box))
    return out


_CELLS: dict = {}


def _cells(p) -> np.ndarray:
    """(n, 3) y, x, z cells a placement occupies (its yaw's cells, as the native walk maps it)."""
    if not _CELLS:
        from .sampler import vocabulary
        from .snot_options import with_sideways
        for v in with_sideways(vocabulary()):     # sideways yaws are codes of their own (1001...), never a yaw
            for o in v.orients:
                _CELLS.setdefault((v.part.ldraw, o.rot), o.cells)
    c = _CELLS.get((p.part.ldraw, p.rotation_deg))
    if c is None:   # shortcut: a part outside the vocabulary taken as its box; fine while the walk places only its own
        c = np.argwhere(np.ones((p.part.height_units, p.w_world, p.d_world), bool))
    return c + (p.y_plate, p.x_stud, p.z_stud)


def _inside(p, box) -> bool:
    c = _cells(p)
    return all(((c[:, k] >= s.start) & (c[:, k] < s.stop)).all() for k, s in enumerate(box))


def evaluate(target, models, placements, sample, **kw) -> dict:
    """The build's energy as it stands: every cell locked, so nothing is filled or moved."""
    return sample(replace(target, lock=np.ones_like(target.want)), models, iters=1, warmup=0,
                  init_placements=placements, **kw)[1]


def rebuild(target, models, placements, box, sample, iters: int, seed: int = 0, **kw):
    """(placements, info): the parts inside `box` built again, inside the whole build with every other cell locked,
    so each step is judged by the whole build's energy (a crop alone would weigh its own statistics against shape)."""
    lock = np.ones(target.want.shape, bool)
    lock[box] = False
    for p in placements:
        if not _inside(p, box):        # a part reaching into the box stays
            lock[tuple(_cells(p).T)] = True
    return sample(replace(target, lock=lock), models, iters=iters, seed=seed, init_placements=placements, **kw)


def rebuild_worst(target, models, placements, err_cell, sample, k: int = 8, iters: int = 200_000,
                  seed: int = 0, **kw):
    """Rebuild the k worst regions in turn, each kept when the whole build's energy falls; (placements, report)."""
    ev = evaluate(target, models, placements, sample, **kw)
    e0 = ev["energy"]
    print(f"regions: build energy {e0:.0f}, shape {ev['fidelity']:.0f}, unsupported {ev['unsupported']}, "
          f"{len(placements)} parts", flush=True)
    report = []
    for r, box in enumerate(find_regions(err_cell, target.want, k=k)):
        cand, ev = rebuild(target, models, placements, box, sample, iters, seed=seed + 7919 * (r + 1), **kw)
        kept = ev["energy"] < e0
        report.append({"box": [[s.start, s.stop] for s in box], "before": e0, "after": ev["energy"], "kept": kept})
        print(f"region {r + 1}: cells {[(s.start, s.stop) for s in box]}  energy {e0:.0f} -> {ev['energy']:.0f} "
              f"(shape {ev['fidelity']:.0f}, unsupported {ev['unsupported']}, {len(cand)} parts)  "
              f"{'kept' if kept else 'dropped'}", flush=True)
        if kept:
            placements, e0 = cand, ev["energy"]
    return placements, report
