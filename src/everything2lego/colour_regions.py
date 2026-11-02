"""Colour by regions: the target's showing cells grouped into colour regions as coarse as real sets' colour patches,
each region in one colour.

A mesh texture is read cell by cell: a mottled roof (orange and brown flecks) gives a mottled build.  A set designer
reads the area: one roof colour.  Here the showing cells (want & visible, colour known) are merged into regions on
their 6-neighbour graph by Ward's rule: the touching pair of regions whose merge adds the least colour variance
(|mean_a - mean_b|^2 * n_a n_b / (n_a + n_b), Lab) merges first -- flecks go into their surround long before two large
areas of different colour meet.  Merging stops at `density` regions per showing cell: real building sets' median is 3.8
per 100 showing cells (tools/learn/build_stats.py patches/100; our builds without this have 7-7.5).  Each region then takes the colour among its cells' colours with the least summed Lab distance to them.

    from everything2lego.colour_regions import regions
    target.colour = regions(target.colour, target.want, target.visible)
"""
from __future__ import annotations

import numpy as np

from .colors import _palette_lab

REAL_DENSITY = 0.038


def regions(colour: np.ndarray, want: np.ndarray, visible: np.ndarray, density: float = REAL_DENSITY) -> np.ndarray:
    """colour (H, X, Z) palette indices (-1 unknown) -> copy with every showing cell in its region's colour."""
    import heapq
    show = want & visible & (colour >= 0)
    idx = -np.ones(colour.shape, int); idx[show] = np.arange(show.sum())
    n = int(show.sum())
    if n < 2:
        return colour.copy()
    lab = _palette_lab(); col = colour[show]
    pairs = []
    for ax in range(3):
        a = np.moveaxis(idx, ax, 0); pairs.append(np.stack([a[:-1].ravel(), a[1:].ravel()], 1))
    e = np.concatenate(pairs); e = e[(e >= 0).all(1)]
    parent = np.arange(n); cnt = np.ones(n); tot = lab[col].astype(float).copy(); ver = np.zeros(n, int)
    nbrs: list[set] = [set() for _ in range(n)]
    for u, v in e:
        nbrs[u].add(v); nbrs[v].add(u)

    def ward(a: int, b: int) -> float:
        d = tot[a] / cnt[a] - tot[b] / cnt[b]
        return float(d @ d) * cnt[a] * cnt[b] / (cnt[a] + cnt[b])
    heap = [(ward(u, v), u, v, 0, 0) for u, v in e]; heapq.heapify(heap)
    left = n - max(1, int(round(density * n)))                                  # merges to make
    while heap and left > 0:
        _, a, b, va, vb = heapq.heappop(heap)
        if parent[a] != a or parent[b] != b or ver[a] != va or ver[b] != vb:
            continue                                                             # stale: a side merged since
        if cnt[a] < cnt[b]: a, b = b, a
        parent[b] = a; cnt[a] += cnt[b]; tot[a] += tot[b]; ver[a] += 1; left -= 1
        nbrs[a] |= nbrs[b]; nbrs[a] -= {a, b}
        for c in nbrs[b] - {a}:
            nbrs[c].discard(b); nbrs[c].add(a)
        nbrs[b] = set()
        for c in nbrs[a]:
            heapq.heappush(heap, (ward(a, c), a, c, ver[a], ver[c]) if a < c else (ward(c, a), c, a, ver[c], ver[a]))

    def find(u: int) -> int:
        while parent[u] != u:
            u = parent[u]
        return u
    root = np.array([find(u) for u in range(n)])
    out = colour.copy(); new = col.copy()
    for r in np.unique(root):
        m = root == r
        cs, k = np.unique(col[m], return_counts=True)
        cost = (k[None, :] * np.linalg.norm(lab[cs][:, None] - lab[cs][None], axis=-1)).sum(1)
        new[m] = cs[cost.argmin()]
    out[show] = new
    return out


if __name__ == "__main__":                                 # self-check: a speckled face becomes one colour, a real edge stays
    rng = np.random.default_rng(0)
    c = np.full((1, 20, 40), 0); c[0, :, 20:] = 1                               # two halves, palette 0 | palette 1
    speck = rng.random(c.shape) < 0.2; c[speck & (c == 0)] = 2                   # flecks of 2 in the left half
    want = np.ones(c.shape, bool); vis = want.copy()
    o = regions(c, want, vis, density=2 / 800)
    assert (o[0, :, :20] == 0).all() and (o[0, :, 20:] == 1).all(), np.unique(o, return_counts=True)
    print("ok: flecks", int(speck[0, :, :20].sum()), "-> 0; halves kept")
