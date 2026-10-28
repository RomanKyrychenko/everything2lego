"""Surface direction: does each exposed face of a part point the way the target's surface does there?

The shape term counts filled sub-samples, so a plate staircase and a slope fit a 37 deg roof about equally; what
tells them apart is the direction their faces point.  Here
    target   each cell on the outer surface gets the mesh's outward direction (gradient of the slightly smoothed
             sub-sample occupancy); cells deeper in get none (nobody sees them)
    option   each cell of a part gets its faces: area-weighted outward directions from the part's slightly smoothed
             sub-sample shape (so a slope's face is slanted, not a staircase of samples), binned into 26 directions
and a placed part costs, over its faces in surface cells, area * (1 - cos) for faces turned outward (cos > 0); faces
turned inward are hidden and free.  Per unit of a 37 deg roof: a plate staircase ~0.4, a 33 deg slope ~0.005.
The cost depends only on the part and its place, so the engine keeps it like the shape term.
"""
from __future__ import annotations

import numpy as np

from .sampler import SUB, unpack

SAMPLE = np.array([2.0, 5.0, 5.0])                       # LDU per sub-sample (y, x, z)
STUD_TOP = 400.0                                          # LDU^2: area unit (one stud's top)
DIRS = np.array([(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if (a, b, c) != (0, 0, 0)], float)
DIRS /= np.linalg.norm(DIRS, axis=1, keepdims=True)


def _grad_out(F: np.ndarray, sigma: float) -> np.ndarray:
    """Outward area-weighted normals (..., 3) of a sample occupancy (y, x, z): -grad of the smoothed field x sample volume."""
    from scipy.ndimage import gaussian_filter
    G = np.stack(np.gradient(gaussian_filter(F.astype(float), sigma), *SAMPLE), -1)
    return -G * SAMPLE.prod()


def target_normals(bits: np.ndarray, want: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """(H*X*Z*3,) float32: unit outward direction per target cell on the outer surface, zeros elsewhere."""
    from scipy.ndimage import binary_dilation, label
    H, X, Z = bits.shape
    T = unpack(bits)
    G = _grad_out(T, sigma).reshape(H, SUB, X, SUB, Z, SUB, 3).sum((1, 3, 5))
    lab, _ = label(np.pad(~want, 1, constant_values=True))
    outside = (lab == lab[0, 0, 0])[1:-1, 1:-1, 1:-1]
    surf = want & binary_dilation(outside, structure=np.ones((3, 3, 3), bool))
    n = np.linalg.norm(G, axis=-1, keepdims=True)
    N = np.where(surf[..., None] & (n > 1e-9), G / np.maximum(n, 1e-9), 0.0)
    return N.astype(np.float32).reshape(-1)


def option_faces(o, sigma: float = 1.0) -> np.ndarray:
    """(m, 7) float32 rows (cy, cx, cz, gx..gz in (y, x, z) order, |g|): a part's faces per cell, binned by direction,
    area in stud tops.  Faces just outside the box are given to the box cell next to them."""
    h, W, D = o.size
    g = np.zeros(o.size, np.uint64); g[tuple(o.cells.T)] = o.bits
    F = np.pad(unpack(g), 2)
    G = _grad_out(F, sigma)
    mag = np.linalg.norm(G, axis=-1)
    idx = np.argwhere(mag > 1e-3 * mag.max()) if mag.max() > 0 else np.zeros((0, 3), int)
    if not len(idx):
        return np.zeros((0, 7), np.float32)
    cell = np.clip((idx - 2) // SUB, 0, np.array(o.size) - 1)
    vec = G[tuple(idx.T)]
    b = np.argmax(vec @ DIRS.T, axis=1)
    occupied = {tuple(c) for c in o.cells.tolist()}
    acc: dict = {}
    for c, bb, v in zip(map(tuple, cell.tolist()), b, vec):
        if c not in occupied:
            continue                                      # a face over an empty corner of the box: no cell of the part there
        a = acc.setdefault((c, int(bb)), np.zeros(3)); a += v
    rows = [(*c, *(v / STUD_TOP), np.linalg.norm(v) / STUD_TOP) for (c, _), v in acc.items() if np.linalg.norm(v) > 1e-6]
    return np.array(rows, np.float32).reshape(-1, 7)


def tables(opts) -> tuple[np.ndarray, np.ndarray]:
    """For every option (sampler order): start offsets (n_opt + 1,) int32 and the stacked face rows (m, 7) float32."""
    rows, start = [], [0]
    for _, o in opts:
        r = option_faces(o); rows.append(r); start.append(start[-1] + len(r))
    return np.array(start, np.int32), (np.concatenate(rows) if rows else np.zeros((0, 7))).astype(np.float32)


def cost(faces: np.ndarray, normal_at) -> float:
    """Reference cost of one placement: sum over faces of |g| - g.n where g.n > 0 (normal_at(cy, cx, cz) -> 3-vector)."""
    s = 0.0
    for cy, cx, cz, gy, gx, gz, m in faces:
        n = normal_at(int(cy), int(cx), int(cz)); d = gy * n[0] + gx * n[1] + gz * n[2]
        if d > 0: s += m - d
    return s


if __name__ == "__main__":                                # self-check: on a 37 deg roof a slope beats a plate step
    import sys
    sys.path.insert(0, "src")
    from .sampler import select_vocabulary
    V = {v.part.ldraw: v for v in select_vocabulary("building", 2000)}
    th = np.radians(37)
    roof = lambda d: np.array([np.cos(th), 0.0, -np.sin(th)]) if d else None   # noqa: E731 (y up, facing -z)
    slope = next(o for o in V["4286.dat"].orients if o.rot == 0)
    plate = V["3623.dat"].orients[0]
    for name, o in (("33 deg slope 3x1", slope), ("plate 1x3", plate)):
        f = option_faces(o)
        c = cost(f, lambda cy, cx, cz: roof(True))
        area = f[:, 6].sum()
        print(f"{name:18s} faces {len(f):3d}  cost {c:6.3f}  per face area {c / max(area, 1e-9):.3f}")
