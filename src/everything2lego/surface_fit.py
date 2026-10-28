"""Surface fit: the sampler's shape term weighted by distance to the target's surface (an engine-side stand-in for
the two-way surface distance tools/ldr_to_mesh.py --compare measures).

Per sub-sample of every cell (the 4x4x4 masks, 5 LDU across, 2 LDU up):
    w_out  LEGO where the target is empty: 1 + alpha * (distance to the target, studs) outside the model; ~0 in a
           hollow the model encloses (hidden: holes and fill inside don't matter)
    w_in   the target covered: 1 within `band` studs of the outer surface, `deep` further in (a missing wall costs,
           an empty interior barely does)
The engine adds w_out and subtracts w_in for each sample a part fills, so the term stays incremental.
"""
from __future__ import annotations

import numpy as np

from .sampler import SUB, unpack

STUD, SAMPLE = 20.0, (2.0, 5.0, 5.0)                       # LDU; sample spacing (y, x, z)


def weights(bits: np.ndarray, alpha: float = 1.0, band: float = 1.0, deep: float = 0.05, hollow: float = 0.02):
    """bits (H, X, Z) uint64 target masks -> (w_out, w_in) float32 arrays of H*X*Z*64, in the engine's order."""
    from scipy.ndimage import distance_transform_edt, label
    T = unpack(bits)                                       # (H*4, X*4, Z*4) bool, the target's samples
    empty = ~T
    lab, _ = label(np.pad(empty, 1, constant_values=True))   # empty samples joined to the grid's border = outside
    outside = (lab == lab[0, 0, 0])[1:-1, 1:-1, 1:-1]
    d_out = distance_transform_edt(~T, sampling=SAMPLE) / STUD          # studs to the nearest target sample
    d_in = distance_transform_edt(~outside, sampling=SAMPLE) / STUD     # studs from the outside, inward
    w_out = np.where(outside, 1.0 + alpha * d_out, hollow)
    w_in = np.where(d_in <= band, 1.0, deep)
    H, X, Z = bits.shape
    order = lambda a: a.reshape(H, SUB, X, SUB, Z, SUB).transpose(0, 2, 4, 1, 3, 5).reshape(-1)   # noqa: E731  (pack's bit order)
    return order(w_out).astype(np.float32), order(w_in).astype(np.float32)


if __name__ == "__main__":                                 # self-check on a box target
    from .sampler import FULL
    b = np.zeros((6, 6, 6), np.uint64); b[1:4, 1:5, 1:5] = FULL
    wo, wi = weights(b)
    wo, wi = wo.reshape(6, 6, 6, 64), wi.reshape(6, 6, 6, 64)
    assert wi[2, 2, 2].min() == 1.0 or wi[2, 2, 2].max() == 1.0          # interior cell: near the surface at this size
    assert wo[0, 0, 0].max() > wo[0, 1, 1].max() >= 1.0                  # further out costs more
    assert wo[2, 2, 2].max() <= 1.0 and wi[5, 5, 5].max() <= 1.0
    print("ok: outside cost", round(float(wo[0, 1, 1].mean()), 2), "->", round(float(wo[0, 0, 0].mean()), 2))
