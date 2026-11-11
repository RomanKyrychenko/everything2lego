"""A learned proposal distribution for the sampler: P(token | context), a
small neural network trained on the official models
(tools/learn/proposal_net.py) to say which part comes next.

The context of the cell (y, x, z) where a part's first cell goes — the
cells are filled in (y, x, z) order, as the sampler fills them — is

  target   how full the object makes each of the window's cells (0..1:
           a sloped surface half-fills its cells)
  placed   the window's cells already built, before (y, x, z) in fill order
  kinds    how many of those built cells each part kind holds (/ window size)
  object   the kind of object (one-hot: building, road_vehicle, ...)
  static   the wider view, fixed for a target (static_context): the
           object's mean fill in 3 x 3 x 3 blocks around the cell, the
           cell's height in the object, its depth inside it

over a window of WY x WX x WZ cells around the cell.  The sampler's Rust
core (native/sampler) computes the same features and the same network;
change one, change both.
"""
from __future__ import annotations

from functools import lru_cache
from importlib.resources import files
from typing import Optional

import numpy as np

WY, WX, WZ = (-1, 4), (-2, 4), (-2, 4)          # window offsets [lo, hi) around the cell
SHAPE = (WY[1] - WY[0], WX[1] - WX[0], WZ[1] - WZ[0])
NWIN = SHAPE[0] * SHAPE[1] * SHAPE[2]

# window cells before the cell in fill order (y, then x, then z)
_d = np.indices(SHAPE).reshape(3, -1).T + np.array([WY[0], WX[0], WZ[0]])
BEFORE = ((_d[:, 0] < 0) | ((_d[:, 0] == 0) & ((_d[:, 1] < 0) | ((_d[:, 1] == 0) & (_d[:, 2] < 0))))).reshape(SHAPE)


BLOCK = (3, 4, 4)                               # the wider view's blocks: (plates, studs, studs)
BLOCKS = [(by, bx, bz) for by in (-3, 0, 3) for bx in (-6, -2, 2) for bz in (-6, -2, 2)]
# longer-range blocks for whole-target context: object-scale footprint samples
# at ±12 studs, ±9 plates.  Set via E2L_WIDE_CONTEXT=1 to enable (requires
# a from-scratch retrain since n_inputs changes).
import os as _os
if _os.environ.get("E2L_WIDE_CONTEXT") == "1":
    BLOCKS = BLOCKS + [(by, bx, bz) for by in (-9, 9) for bx in (-12, 12) for bz in (-12, 12)]
NSTATIC = len(BLOCKS) + 2


def n_inputs(n_kinds: int, n_objects: int) -> int:
    return 2 * NWIN + n_kinds + 1 + n_objects + NSTATIC


def static_context(fill: np.ndarray) -> np.ndarray:
    """(H, X, Z, NSTATIC) float32: per cell, the object's mean fill in each
    block of BLOCKS (offsets of the block's minimum corner; outside the
    grid is empty), the cell's height in the object (0 bottom .. 1 top) and
    its depth inside it (distance to the outside, studs, capped at 8, / 8)."""
    from scipy.ndimage import distance_transform_edt
    H, X, Z = fill.shape
    S = np.zeros((H + 1, X + 1, Z + 1))
    S[1:, 1:, 1:] = fill.cumsum(0).cumsum(1).cumsum(2)
    y, x, z = np.meshgrid(np.arange(H), np.arange(X), np.arange(Z), indexing="ij")
    out = np.zeros((H, X, Z, NSTATIC), np.float32)
    for n, (by, bx, bz) in enumerate(BLOCKS):
        y0, y1 = np.clip(y + by, 0, H), np.clip(y + by + BLOCK[0], 0, H)
        x0, x1 = np.clip(x + bx, 0, X), np.clip(x + bx + BLOCK[1], 0, X)
        z0, z1 = np.clip(z + bz, 0, Z), np.clip(z + bz + BLOCK[2], 0, Z)
        tot = (S[y1, x1, z1] - S[y0, x1, z1] - S[y1, x0, z1] - S[y1, x1, z0]
               + S[y0, x0, z1] + S[y0, x1, z0] + S[y1, x0, z0] - S[y0, x0, z0])
        out[..., n] = tot / (BLOCK[0] * BLOCK[1] * BLOCK[2])
    out[..., -2] = y / max(1, H - 1)
    depth = distance_transform_edt(fill >= 0.5, sampling=(0.4, 1.0, 1.0))
    out[..., -1] = np.minimum(depth, 8.0) / 8.0
    return out


def features(fill: np.ndarray, placed_kind: np.ndarray, static: np.ndarray, y: int, x: int, z: int,
             object_index: int, n_kinds: int, n_objects: int) -> np.ndarray:
    """fill: (H, X, Z) 0..1, how full the object makes each cell;
    placed_kind: (H, X, Z) int, the kind index of the part built in each
    cell, -1 where none (kind n_kinds: a part of no known kind);
    static: static_context(fill)."""
    target = fill
    H, X, Z = target.shape
    t = np.zeros(SHAPE, np.float32)
    k = np.full(SHAPE, -1, np.int64)
    y0, x0, z0 = y + WY[0], x + WX[0], z + WZ[0]
    sy = slice(max(0, y0), min(H, y + WY[1]))
    sx = slice(max(0, x0), min(X, x + WX[1]))
    sz = slice(max(0, z0), min(Z, z + WZ[1]))
    dst = (slice(sy.start - y0, sy.stop - y0), slice(sx.start - x0, sx.stop - x0), slice(sz.start - z0, sz.stop - z0))
    t[dst] = target[sy, sx, sz]
    k[dst] = placed_kind[sy, sx, sz]
    k[~BEFORE] = -1
    placed = (k >= 0).astype(np.float32)
    bag = np.bincount(k[k >= 0], minlength=n_kinds + 1)[:n_kinds + 1].astype(np.float32) / NWIN
    obj = np.zeros(n_objects, np.float32)
    if object_index >= 0:
        obj[object_index] = 1.0
    return np.concatenate([t.ravel(), placed.ravel(), bag, obj, static[y, x, z]])


class Net:
    """A one-hidden-layer network: input -> ReLU(W1 x + b1) -> W2 h + b2 ->
    log-softmax over `tokens` (+ "other": any part not in the vocabulary)."""

    def __init__(self, d: dict):
        self.W1, self.b1, self.W2, self.b2 = (np.asarray(d[k], np.float32) for k in ("W1", "b1", "W2", "b2"))
        self.mu, self.sd = np.asarray(d["mu"], np.float32), np.asarray(d["sd"], np.float32)
        self.tokens = [str(t) for t in d["tokens"]]
        self.kinds = [str(k) for k in d["kinds"]]
        self.objects = [str(o) for o in d["objects"]]
        self.token_index = {t: i for i, t in enumerate(self.tokens)}
        self.kind_index = {k: i for i, k in enumerate(self.kinds)}
        self.object_index = {o: i for i, o in enumerate(self.objects)}

    def logprobs(self, x: np.ndarray) -> np.ndarray:
        h = np.maximum(0.0, ((x - self.mu) / self.sd) @ self.W1 + self.b1)
        z = h @ self.W2 + self.b2
        z = z - z.max(axis=-1, keepdims=True)
        return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))

    def other(self) -> int:
        return len(self.tokens)


@lru_cache(maxsize=1)
def load() -> Optional[Net]:
    """The trained network (data/proposal_net.npz), or None when there is none."""
    f = files("everything2lego.data").joinpath("proposal_net.npz")
    if not f.is_file():
        return None
    with f.open("rb") as fh:
        return Net(dict(np.load(fh, allow_pickle=False)))
