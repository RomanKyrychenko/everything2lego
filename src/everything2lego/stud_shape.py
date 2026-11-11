"""Stud-scale shape of a part or of a region of the target, by one function, so the two can be compared: what survives
a generated mesh is shape at the scale of a stud (a cone, an arch, a slope, a round tower), not a part's fine detail.

Per cell, from the 4x4x4 sub-sample shapes (sampler bits) of the whole thing the cell is in (the part's own cells, or
the whole target: a region's cut edges inside a wall are not faces), the outward faces (surface_dir._grad_out) binned
by how steep they are -- up, slope up, side, slope down, down -- and, for the faces that are not flat tops or
bottoms, how they turn about the vertical: (cos 4a, sin 4a) area-weighted, so walls square to the grid sum to a long
vector and a round surface to a short one.  A thing's descriptor (DIM values, yaw-invariant):

    height (plates), long and short side (studs), fill (share of the box's sub-samples), the five steepness shares,
    roundness (1 - the turn vector's length over the side area)

shortcut: at 4 sub-samples a stud, a thing 1 stud across reads round whether it is or not (its corners outweigh its
sides); roundness tells shapes apart from 2 studs across.  Finer sub-samples would fix it, at 8x the memory.
"""
from __future__ import annotations

import numpy as np

from .sampler import SUB, unpack
from .surface_dir import _grad_out

DIM = 10
EDGES = (0.8, 0.3, -0.3, -0.8)                          # up | slope up | side | slope down | down, by the face's y


def cell_faces(bits: np.ndarray, sigma: float = 0.6) -> np.ndarray:
    """(H, X, Z, 8) per cell: area of its outward faces in the five steepness bins, then the side-turn vector
    (sum w cos 4a, sum w sin 4a) and its weight sum w (faces not within 37 deg of up or down)."""
    H, X, Z = bits.shape
    F = np.pad(unpack(bits), SUB)
    G = _grad_out(F, sigma)[SUB:-SUB, SUB:-SUB, SUB:-SUB]
    w = np.linalg.norm(G, axis=-1)
    e = np.where(w > 1e-9, G[..., 0] / np.maximum(w, 1e-9), 0.0)            # (y, x, z): the face's up component
    b = np.digitize(-e, [-x for x in EDGES])                               # 0 up .. 4 down
    out = np.zeros((H * SUB, X * SUB, Z * SUB, 8))
    for k in range(5):
        out[..., k] = np.where(b == k, w, 0.0)
    side = (np.abs(e) < EDGES[0]) & (w > 1e-9)
    a = np.arctan2(G[..., 2], G[..., 1])
    out[..., 5] = np.where(side, w * np.cos(4 * a), 0.0)
    out[..., 6] = np.where(side, w * np.sin(4 * a), 0.0)
    out[..., 7] = np.where(side, w, 0.0)
    return out.reshape(H, SUB, X, SUB, Z, SUB, 8).sum((1, 3, 5))


def describe(cells: np.ndarray, bits: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """(DIM,) the descriptor of the thing occupying `cells` (n, 3) (y, x, z) with shapes `bits` (n,) and per-cell
    `faces` (n, 8) from cell_faces."""
    lo, hi = cells.min(0), cells.max(0) + 1
    h, w, d = hi - lo
    fill = sum(bin(int(x)).count("1") for x in bits) / (h * w * d * SUB ** 3)
    f = faces.sum(0)
    area = f[:5].sum()
    steep = f[:5] / area if area > 0 else np.zeros(5)
    round_ = 1 - np.hypot(f[5], f[6]) / f[7] if f[7] > 0 else 0.0
    return np.array([h, max(w, d), min(w, d), fill, *steep, round_], np.float32)


def part_descriptor(o) -> np.ndarray:
    """The descriptor of a vocabulary part in one orientation (sampler.Orient)."""
    g = np.zeros(o.size, np.uint64); g[tuple(o.cells.T)] = o.bits
    return describe(o.cells, o.bits, cell_faces(g)[tuple(o.cells.T)])
