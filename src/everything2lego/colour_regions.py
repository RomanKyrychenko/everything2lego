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

`regions` works on palette indices after each cell was snapped, so two palette colours either side of the mesh's real
colour come out as stripes.  `regions_by_part` merges first, on the continuous Lab (`cell_lab`, read from the mesh
before snapping), within each mesh part (part_labels.py), and snaps each region's mean once (--colour-by-part).
"""
from __future__ import annotations

import numpy as np

from .colors import _palette_lab

REAL_DENSITY = 0.038


def _ward(feat: np.ndarray, e: np.ndarray, group: np.ndarray, density: float) -> np.ndarray:
    """Ward agglomeration of n cells (feat (n, 3) Lab) over edges e (m, 2) joining cells of the same group: each group
    merged down to max(1, round(density * its cells)) regions.  -> root (n,) region id per cell."""
    import heapq
    n = len(feat)
    parent = np.arange(n); cnt = np.ones(n); tot = feat.astype(float).copy(); ver = np.zeros(n, int)
    _, g, gn = np.unique(group, return_inverse=True, return_counts=True)
    left = gn - np.maximum(1, np.round(density * gn)).astype(int)              # merges to make, per group
    nbrs: list[set] = [set() for _ in range(n)]
    for u, v in e:
        nbrs[u].add(v); nbrs[v].add(u)

    def ward(a: int, b: int) -> float:
        d = tot[a] / cnt[a] - tot[b] / cnt[b]
        return float(d @ d) * cnt[a] * cnt[b] / (cnt[a] + cnt[b])
    heap = [(ward(u, v), u, v, 0, 0) for u, v in e]; heapq.heapify(heap)
    todo = int(left.sum())
    while heap and todo > 0:
        _, a, b, va, vb = heapq.heappop(heap)
        if parent[a] != a or parent[b] != b or ver[a] != va or ver[b] != vb or left[g[a]] <= 0:
            continue                                                             # stale, or its group is done
        if cnt[a] < cnt[b]: a, b = b, a
        parent[b] = a; cnt[a] += cnt[b]; tot[a] += tot[b]; ver[a] += 1; left[g[a]] -= 1; todo -= 1
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
    return np.array([find(u) for u in range(n)])


def _edges(show: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(cell index grid, -1 off `show`; 6-neighbour edges between showing cells)."""
    idx = -np.ones(show.shape, int); idx[show] = np.arange(show.sum())
    pairs = []
    for ax in range(3):
        a = np.moveaxis(idx, ax, 0); pairs.append(np.stack([a[:-1].ravel(), a[1:].ravel()], 1))
    e = np.concatenate(pairs)
    return idx, e[(e >= 0).all(1)]


def regions(colour: np.ndarray, want: np.ndarray, visible: np.ndarray, density: float = REAL_DENSITY) -> np.ndarray:
    """colour (H, X, Z) palette indices (-1 unknown) -> copy with every showing cell in its region's colour."""
    show = want & visible & (colour >= 0)
    if show.sum() < 2:
        return colour.copy()
    lab = _palette_lab(); col = colour[show]
    _, e = _edges(show)
    root = _ward(lab[col], e, np.zeros(len(col), int), density)
    out = colour.copy(); new = col.copy()
    for r in np.unique(root):
        m = root == r
        cs, k = np.unique(col[m], return_counts=True)
        cost = (k[None, :] * np.linalg.norm(lab[cs][:, None] - lab[cs][None], axis=-1)).sum(1)
        new[m] = cs[cost.argmin()]
    out[show] = new
    return out


def cell_lab(mesh, mesh_to_voxel: np.ndarray, shape: tuple, n: int = 400_000) -> np.ndarray:
    """(H, X, Z, 3) continuous Lab per cell: the mean colour of `n` surface samples of the (aligned) mesh falling in the
    cell, mapped by MeshTarget.mesh_to_voxel; cells with no sample take the nearest sampled cell's.  The voxeliser
    snaps each cell's colour to the palette at once; this is the colour before snapping, read over the cell's area."""
    import trimesh
    from scipy.ndimage import distance_transform_edt
    from .colors import _rgb_to_lab
    from .semantic_v2.mesh_target import _sample_mesh_colors
    pts, fid = trimesh.sample.sample_surface(mesh, n, seed=0)
    lab = _rgb_to_lab(_sample_mesh_colors(mesh, pts, fid).astype(np.float32) / 255.0)
    v = np.floor(np.c_[pts, np.ones(len(pts))] @ np.asarray(mesh_to_voxel, float).T)[:, :3].astype(int)
    c = v[:, [1, 0, 2]]                                                          # (x, y, z) -> grid (y, x, z)
    ok = ((c >= 0) & (c < np.array(shape))).all(1)
    flat = np.ravel_multi_index(c[ok].T, shape); N = int(np.prod(shape))
    k = np.bincount(flat, minlength=N)
    out = np.stack([np.bincount(flat, lab[ok, j], minlength=N) for j in range(3)], 1) / np.maximum(k, 1)[:, None]
    out = out.reshape(*shape, 3); has = k.reshape(shape) > 0
    if has.any():
        _, ii = distance_transform_edt(~has, return_indices=True)
        out = out[tuple(ii)]
    return out.astype(np.float32)


def regions_by_part(lab: np.ndarray, colour: np.ndarray, want: np.ndarray, visible: np.ndarray,
                    part: np.ndarray | None = None, density: float = REAL_DENSITY,
                    allowed: np.ndarray | None = None, link: float = 6.0) -> np.ndarray:
    """Merge, then snap: each part's showing cells (part (H, X, Z) int, -1 = the unlabelled rest; None = one part) are
    merged by Ward's rule on their continuous Lab (lab (H, X, Z, 3)), only across neighbours of the same part, to
    max(1, round(density * n)) regions per part; each region takes the palette colour nearest its mean Lab among
    `allowed` palette indices (default: the colours the target's showing cells already have); regions whose means are
    within `link` (Lab) of each other, anywhere, snap as one (0 = off).  -> new colour."""
    show = want & visible & (colour >= 0)
    if show.sum() < 2:
        return colour.copy()
    pal = _palette_lab()
    if allowed is None:
        allowed = np.unique(colour[show])
    allowed = np.asarray(allowed, int)
    grp = np.zeros(int(show.sum()), int) if part is None else np.asarray(part)[show].astype(int)
    idx, e = _edges(show)
    e = e[grp[e[:, 0]] == grp[e[:, 1]]]
    f = lab[show].astype(float)
    root = _ward(f, e, grp, density)
    _, r, rn = np.unique(root, return_inverse=True, return_counts=True)
    mean = np.stack([np.bincount(r, f[:, j]) for j in range(3)], 1) / rn[:, None]
    # one material lit differently on two faces (the two halves of a roof, ~5 apart) lands either side of a palette
    # boundary: regions within `link` of a larger one's group mean (any part, not only touching) share its mean
    lead, gm, gn = -np.ones(len(rn), int), [], []
    for i in np.argsort(-rn):
        dd = [np.linalg.norm(mean[i] - m) for m in gm]
        if dd and min(dd) <= link:
            j = int(np.argmin(dd)); gm[j] = (gm[j] * gn[j] + mean[i] * rn[i]) / (gn[j] + rn[i]); gn[j] += rn[i]
        else:
            j = len(gm); gm.append(mean[i].copy()); gn.append(rn[i])
        lead[i] = j
    mean = np.array(gm)[lead]
    near = allowed[((mean[:, None] - pal[allowed][None]) ** 2).sum(-1).argmin(1)]
    out = colour.copy(); out[show] = near[r]
    return out


if __name__ == "__main__":                                 # self-check: a speckled face becomes one colour, a real edge stays
    rng = np.random.default_rng(0)
    c = np.full((1, 20, 40), 0); c[0, :, 20:] = 1                               # two halves, palette 0 | palette 1
    speck = rng.random(c.shape) < 0.2; c[speck & (c == 0)] = 2                   # flecks of 2 in the left half
    want = np.ones(c.shape, bool); vis = want.copy()
    o = regions(c, want, vis, density=2 / 800)
    assert (o[0, :, :20] == 0).all() and (o[0, :, 20:] == 1).all(), np.unique(o, return_counts=True)
    print("ok: flecks", int(speck[0, :, :20].sum()), "-> 0; halves kept")
    # merge, then snap, per part: part 0 speckled around one colour -> one colour; part 1 two colours half/half -> two
    from .colors import palette
    P = _palette_lab(); a, b, d = 0, 1, 2                                         # three palette colours
    part = np.zeros((1, 20, 40), int); part[0, :, 20:] = 1
    lab = np.empty((1, 20, 40, 3)); lab[..., :] = P[a]
    lab[0, :, :20] += rng.normal(0, 6, (20, 20, 3))                              # speckle, in Lab, around colour a
    lab[0, :10, 20:] = P[b]; lab[0, 10:, 20:] = P[d]                             # part 1: b on top, d below
    col = np.full((1, 20, 40), a)
    o = regions_by_part(lab, col, want, vis, part, allowed=np.array([a, b, d]))
    assert (o[0, :, :20] == a).all() and (o[0, :10, 20:] == b).all() and (o[0, 10:, 20:] == d).all(), np.unique(o[0, :, :20], return_counts=True)
    print("ok: per part, speckle -> one colour, half/half part keeps two;", len(palette()), "palette colours")
