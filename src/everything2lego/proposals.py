"""A learned proposal distribution for the sampler: P(token | context), a
small neural network trained on the official models
(tools/learn/proposal_net.py) to say which part comes next.

The context of the cell (y, x, z) where a part's first cell goes — the
cells are filled in (y, x, z) order, as the sampler fills them — is

  target   the window's cells that are part of the object (0/1)
  placed   the window's cells already built, before (y, x, z) in fill order
  kinds    how many of those built cells each part kind holds (/ window size)
  object   the kind of object (one-hot: building, road_vehicle, ...)

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


def n_inputs(n_kinds: int, n_objects: int) -> int:
    return 2 * NWIN + n_kinds + 1 + n_objects


def features(target: np.ndarray, placed_kind: np.ndarray, y: int, x: int, z: int,
             object_index: int, n_kinds: int, n_objects: int) -> np.ndarray:
    """target: (H, X, Z) bool; placed_kind: (H, X, Z) int, the kind index of
    the part built in each cell, -1 where none (kind n_kinds: a part of no
    known kind)."""
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
    return np.concatenate([t.ravel(), placed.ravel(), bag, obj])


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
