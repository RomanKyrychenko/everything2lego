"""A part's real surface, as the sampler's surface-distance term charges it.

The sampler's own part shapes are sub-sample occupancies (4x4x4 per cell); a thin part (an antenna, a plant leaf) fills
almost none of them, so a face list made from them gives it almost no surface.  Here each option is placed at the
grid's origin, written as the LDraw lines the model would have (slope_courses.expand, io_ldraw.placement_to_line),
and its library triangles are sampled: points about every 5 LDU, each carrying its share of the area, binned into
the sub-samples they fall in.  Rows (sy, sx, sz, area): the sub-sample (box-local, y up) and the area there in stud
tops (400 LDU^2).  The engine charges each row by how far its sub-sample stands off the target.
"""
from __future__ import annotations

import hashlib
import pickle
import sys
from pathlib import Path

import numpy as np

from .sampler import SUB

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path.home() / ".cache" / "everything2lego" / "part_surface.pkl"
STUD_TOP = 400.0
SPACING = 5.0                                              # LDU between surface points
_MEM: dict | None = None


def _triangles(line: str) -> np.ndarray:
    """(n, 3, 3) LDU triangles of one LDraw type-1 line (or several, newline-separated)."""
    if str(ROOT / "tools") not in sys.path:
        sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "tools" / "set_study")]
    from ldr_to_mesh import _matrix, _part
    out = []
    for ln in line.splitlines():
        t = ln.split()
        if len(t) < 15 or t[0] != "1":
            continue
        v, _ = _part(" ".join(t[14:]).lower())
        if len(v):
            M = _matrix(t[2:14]); out.append(v @ M[:3, :3].T + M[:3, 3])
    return np.concatenate(out) if out else np.zeros((0, 3, 3))


def surface_rows(lines: str, size: tuple[int, int, int]) -> np.ndarray:
    """(m, 4) float32 rows (sy, sx, sz, area in stud tops) of the parts in `lines`, placed with their box's minimum
    corner at cell (0, 0, 0); sub-samples outside the box are clipped onto it."""
    tri = _triangles(lines)
    if not len(tri):
        return np.zeros((0, 4), np.float32)
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    n = np.maximum(1, np.ceil(area / SPACING ** 2)).astype(int)
    rng = np.random.default_rng(0)
    k = np.repeat(np.arange(len(tri)), n)
    r1, r2 = rng.random(len(k)), rng.random(len(k))
    s = np.sqrt(r1)
    p = (1 - s)[:, None] * tri[k, 0] + (s * (1 - r2))[:, None] * tri[k, 1] + (s * r2)[:, None] * tri[k, 2]
    w = (area / n)[k] / STUD_TOP
    # LDU -> sub-samples: x, z studs of 20 LDU centred on 20 * i; y plates of 8 LDU, LDraw y down from the box bottom
    cell = np.stack([-p[:, 1] / 8.0, (p[:, 0] + 10.0) / 20.0, (p[:, 2] + 10.0) / 20.0], 1)
    q = np.clip(np.floor(cell * SUB).astype(int), 0, np.array(size) * SUB - 1)
    key, inv = np.unique(q, axis=0, return_inverse=True)
    acc = np.bincount(inv.ravel(), weights=w, minlength=len(key))
    return np.column_stack([key, acc]).astype(np.float32)


PROTRUDE_LDU = 1.0                                         # geometry this far past the box counts as reaching out


def protrusion_rows(lines: str, size: tuple[int, int, int], studs: frozenset, cells: frozenset) -> np.ndarray:
    """(m, 3) int32 sub-samples (box-local, may lie outside the box) that the part's geometry reaches past the cells
    it claims (`cells`, (y, x, z)): a side stud, a tooth, a clip, a handle, out of its box or in a box cell left free.  Ordinary top studs (on the `studs` (x, z) cells, up to 4.5 LDU over the
    top) are left out: they go into the part above."""
    tri = _triangles(lines)
    if not len(tri):
        return np.zeros((0, 3), np.int32)
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    n = np.maximum(1, np.ceil(area / 1.0)).astype(int)                    # dense: protrusions are small
    rng = np.random.default_rng(0)
    k = np.repeat(np.arange(len(tri)), n)
    r1, r2 = rng.random(len(k)), rng.random(len(k)); s = np.sqrt(r1)
    p = (1 - s)[:, None] * tri[k, 0] + (s * (1 - r2))[:, None] * tri[k, 1] + (s * r2)[:, None] * tri[k, 2]
    ldu = np.stack([-p[:, 1], p[:, 0] + 10.0, p[:, 2] + 10.0], 1)       # y up from the box bottom; x, z from its corner
    hi = np.array([size[0] * 8.0, size[1] * 20.0, size[2] * 20.0])
    out = ((ldu < -PROTRUDE_LDU) | (ldu > hi + PROTRUDE_LDU)).any(1)
    inner = np.clip(ldu, PROTRUDE_LDU, hi - PROTRUDE_LDU)                # points well inside a box cell, by that cell
    cell = np.floor(inner / np.array([8.0, 20.0, 20.0])).astype(int)
    unclaimed = np.array([tuple(c) not in cells for c in cell], bool) if len(cell) else np.zeros(0, bool)
    inside = ((ldu >= PROTRUDE_LDU) & (ldu <= hi - PROTRUDE_LDU)).all(1)
    out |= inside & unclaimed
    cx, cz = np.floor(ldu[:, 1] / 20.0).astype(int), np.floor(ldu[:, 2] / 20.0).astype(int)
    on_stud = np.array([(a, b) in studs for a, b in zip(cx, cz)], bool) if len(ldu) else np.zeros(0, bool)
    stud = (ldu[:, 0] > hi[0]) & (ldu[:, 0] <= hi[0] + 4.5) & on_stud
    stud &= (ldu[:, 1] >= -PROTRUDE_LDU) & (ldu[:, 1] <= hi[1] + PROTRUDE_LDU) & (ldu[:, 2] >= -PROTRUDE_LDU) & (ldu[:, 2] <= hi[2] + PROTRUDE_LDU)
    q = np.floor(ldu[out & ~stud] / np.array([8.0, 20.0, 20.0]) * SUB).astype(np.int32)
    return np.unique(q, axis=0) if len(q) else np.zeros((0, 3), np.int32)


def protrusion_tables(opts) -> tuple[np.ndarray, np.ndarray]:
    """For every option: start offsets (n_opt + 1,) int32 and stacked protrusion sub-samples (m, 3) int32, cached."""
    from .colors import palette
    from .io_ldraw import placement_to_line
    from .slope_courses import expand
    global _MEM
    if _MEM is None:
        try:
            _MEM = pickle.loads(CACHE.read_bytes())
        except (OSError, pickle.UnpicklingError, EOFError):
            _MEM = {}
    rows, start, new = [], [0], False
    black = next(i for i, c in enumerate(palette()) if c.code == 0)
    for v, o in opts:
        lines = "\n".join(placement_to_line(p) for p in expand(v.part, black, 0, 0, 0, o))
        key = "prot2:" + hashlib.sha1((lines + repr(tuple(o.size)) + repr(sorted(o.top)) + repr(o.cells.tolist())).encode()).hexdigest()
        if key not in _MEM:
            _MEM[key] = protrusion_rows(lines, tuple(o.size), frozenset(o.top), frozenset(map(tuple, o.cells.tolist()))); new = True
        rows.append(_MEM[key]); start.append(start[-1] + len(_MEM[key]))
    if new:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        tmp = CACHE.with_suffix(f".{id(rows)}.tmp"); tmp.write_bytes(pickle.dumps(_MEM)); tmp.replace(CACHE)
    return np.array(start, np.int32), (np.concatenate(rows) if rows else np.zeros((0, 3))).astype(np.int32)


def tables(opts) -> tuple[np.ndarray, np.ndarray]:
    """For every option (sampler order): start offsets (n_opt + 1,) int32 and stacked rows (m, 4) float32, cached by
    the option's LDraw lines and box."""
    from .colors import palette
    from .io_ldraw import placement_to_line
    from .slope_courses import expand
    global _MEM
    if _MEM is None:
        try:
            _MEM = pickle.loads(CACHE.read_bytes())
        except (OSError, pickle.UnpicklingError, EOFError):
            _MEM = {}
    rows, start, new = [], [0], False
    black = next(i for i, c in enumerate(palette()) if c.code == 0)
    for v, o in opts:
        lines = "\n".join(placement_to_line(p) for p in expand(v.part, black, 0, 0, 0, o))
        key = hashlib.sha1((lines + repr(tuple(o.size))).encode()).hexdigest()
        if key not in _MEM:
            _MEM[key] = surface_rows(lines, tuple(o.size)); new = True
        rows.append(_MEM[key]); start.append(start[-1] + len(_MEM[key]))
    if new:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        tmp = CACHE.with_suffix(f".{id(rows)}.tmp"); tmp.write_bytes(pickle.dumps(_MEM)); tmp.replace(CACHE)
    return np.array(start, np.int32), (np.concatenate(rows) if rows else np.zeros((0, 4))).astype(np.float32)


if __name__ == "__main__":                                 # self-check: a 2x4 brick's surface sits in its box; an antenna has some
    from .sampler import select_vocabulary
    V = {v.part.ldraw: v for v in select_vocabulary("building", 3000)}
    for name in ("3001.dat", "3957a.dat", "3957.dat", "4589.dat"):
        if name not in V:
            continue
        v = V[name]; o = v.orients[0]
        st, r = tables([(v, o)])
        print(f"{name:10s} box {o.size}  rows {len(r)}  area {r[:, 3].sum():.2f} stud tops  y range {r[:, 0].min():.0f}-{r[:, 0].max():.0f}")
        assert len(r) and (r[:, :3] >= 0).all() and (r[:, :3] < np.array(o.size) * SUB).all()
