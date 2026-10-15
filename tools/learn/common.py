"""The official models as the learning code sees them: every part's grid
box and token, and the object's real shape — each cell's filled share,
from the parts' measured shapes (part_shape masks) carried into the
model by each part's own placement matrix, so a slope fills the cells it
really fills, however it is turned.  Parts without a measured shape count
as their box.

Grid: (y plates up, x studs, z studs), cell (0, 0, 0) at the model's
minimum corner.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "tools" / "partsdb"))
sys.path.insert(0, str(REPO / "tools" / "set_study"))
sys.path.insert(0, str(REPO / "src"))
import features as F  # noqa: E402
from build import DB  # noqa: E402

from everything2lego.sampler import FULL, SUB, popcount, unpack  # noqa: E402

STUD, PLATE = 20.0, 8.0


@lru_cache(maxsize=1)
def _shapes() -> dict:
    """part_num -> (sample points (n, 3) in the part's LDraw frame, LDU)."""
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    out = {}
    for pn, w, d, h, x0, z0, y0, bits in con.execute("SELECT part_num, w, d, h, x0, z0, y0, bits FROM part_shape"):
        fine = unpack(np.array(json.loads(bits), dtype=np.uint64))       # (h*4, w*4, d*4), y from the bottom
        iy, ix, iz = np.nonzero(fine)
        x = x0 + (ix + 0.5) * STUD / SUB
        z = z0 + (iz + 0.5) * STUD / SUB
        y = y0 + h * PLATE - (iy + 0.5) * PLATE / SUB                      # LDraw y points down
        out[pn] = np.stack([x, y, z], 1)
    return out


@lru_cache(maxsize=256)
def model(name: str):
    """(part_nums, tokens, i0, i1, bits) of an official model: each part's
    grid box [i0, i1) (y, x, z), and each cell's shape as a 64-bit mask of
    4x4x4 samples (sampler.pack's bit order).  None when it has no parts."""
    from ldraw_read import flatten, part_bbox
    from patterns import to_model_frame
    f = F.OMR / f"{name}.mpd"
    if not f.exists():
        f = F.OMR / f"{name}.ldr"
    parts = [p for p in flatten(f) if part_bbox(p.name) is not None]
    if not parts:
        return None
    parts = to_model_frame(parts)
    pns, toks, lo, hi = [], [], [], []
    for p in parts:
        a_, b_ = part_bbox(p.name)
        a_ = a_.copy()
        if b_[1] > 0:
            a_[1] = max(a_[1], 0.0)                                       # studs above the body don't count
        c = np.array([[x, y, z, 1] for x in (a_[0], b_[0]) for y in (a_[1], b_[1]) for z in (a_[2], b_[2])])
        w = (p.M @ c.T).T[:, :3]
        a, b = w.min(0), w.max(0)
        lo.append([-b[1] / PLATE, a[0] / STUD, a[2] / STUD])
        hi.append([-a[1] / PLATE, b[0] / STUD, b[2] / STUD])
        pn = F._CTX["ldmap"].get(p.name, p.name[:-4])
        pns.append(pn)
        toks.append(F._CTX["token"].get(pn, "other0x0h0"))
    lo, hi = np.array(lo), np.array(hi)
    i0 = np.floor(lo + 0.25).astype(int)
    i1 = np.maximum(np.ceil(hi - 0.25).astype(int), i0 + 1)
    base = i0.min(0)
    i0, i1 = i0 - base, i1 - base
    shape = tuple(i1.max(0))
    bits = np.zeros(shape, np.uint64)
    shapes = _shapes()
    for p, pn, a, b in zip(parts, pns, i0, i1):
        pts = shapes.get(pn)
        if pts is None:
            bits[a[0]:b[0], a[1]:b[1], a[2]:b[2]] = FULL
            continue
        w = (p.M[:3, :3] @ pts.T).T + p.M[:3, 3]
        gf = np.floor(np.stack([-w[:, 1] / PLATE, w[:, 0] / STUD, w[:, 2] / STUD], 1) * SUB).astype(int)
        g, sub = gf // SUB - base, gf % SUB
        ok = ((g >= 0) & (g < np.array(shape))).all(1)
        bit = (sub[ok, 0] * SUB * SUB + sub[ok, 1] * SUB + sub[ok, 2]).astype(np.uint64)
        np.bitwise_or.at(bits, tuple(g[ok].T), np.left_shift(np.uint64(1), bit))
    return pns, toks, i0, i1, bits


def target(name: str, kind: str | None, degrade: float = 0.0, seed: int = 0):
    """The official model's own shape as a sampler target; `degrade` > 0
    perturbs it as a reconstruction would (sampler targets come from
    meshes, not from the model): that share of its surface cells flipped,
    and each part-full cell's samples redrawn."""
    from scipy.ndimage import binary_dilation, binary_erosion
    from everything2lego import sampler
    got = model(name)
    if got is None:
        return None
    bits = got[4].copy()
    want = popcount(bits) > 1
    if degrade > 0:
        rnd = np.random.default_rng(seed)
        surface = want & ~binary_erosion(want)
        outside = binary_dilation(want) & ~want
        flip = (surface | outside) & (rnd.random(want.shape) < degrade)
        bits = np.where(flip & ~want, FULL, np.where(flip & want, np.uint64(0), bits))
        # and each part-full cell's samples redrawn: a reconstruction does not know them
        part = (popcount(bits) > 0) & (popcount(bits) < SUB ** 3) & (rnd.random(want.shape) < degrade)
        noise = rnd.integers(0, 2 ** 63, want.shape, dtype=np.uint64) | rnd.integers(0, 2 ** 63, want.shape, dtype=np.uint64) << np.uint64(1)
        bits = np.where(part, bits ^ (noise & np.uint64(0x00000000FFFF0000)), bits)
        want = popcount(bits) > 1
    visible = want & ~binary_erosion(want)
    return sampler.Target(want, np.full(want.shape, -1), visible, bits=bits, kind=kind), got
