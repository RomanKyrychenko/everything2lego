"""facade — a measured front for the voxel model.

A single-image 3D generator gives the whole object but only *plausibly*;
a metric depth model (MapAnything) gives the photographed side exactly —
window reveals, a bay's projection, the depth of a porch, the true
skyline — and nothing else.  This module fuses the two:

  1. register the facade's point cloud to the model's front surface
     (which side faces the camera, scale, offsets, residual yaw) by ICP
     in isotropic millimetres, trying each side of the model;
  2. read the measured front off the registered points, column by
     column of the voxel grid: the front depth per (u, y) cell and the
     object's top per column;
  3. override the solid volume with it.

Input: an npz as written by tools/facade_depth.py — `pts3d` (H, W, 3)
metric points in the camera frame (x right, y down, z into the scene)
and `mask` (H, W) bool for the object's pixels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

STUD_MM = 8.0
PLATE_MM = 3.2


@dataclass
class Facade:
    pts: np.ndarray        # (H, W, 3) metric, camera frame
    mask: np.ndarray       # (H, W) bool

    @classmethod
    def load(cls, path: str) -> "Facade":
        d = np.load(path)
        pts = d["pts3d"] if "pts3d" in d else d["0_pts3d"]
        mask = d["mask"] if "mask" in d else d["0_mask"]
        return cls(pts=np.asarray(pts).squeeze(), mask=np.asarray(mask).squeeze().astype(bool))

    def points(self, max_points: int = 20000, seed: int = 0) -> np.ndarray:
        P = self.pts[self.mask]
        P = P[np.isfinite(P).all(axis=1)]
        if len(P) > max_points:
            rng = np.random.default_rng(seed)
            P = P[rng.choice(len(P), max_points, replace=False)]
        return P.astype(np.float64)


@dataclass
class FacadeFit:
    axis: str              # "x" | "z": the horizontal axis of the facade in the grid
    sign: int              # +1: the camera looks from the +depth side (front at high coordinate)
    mirror: bool
    T: np.ndarray          # (4, 4) camera metres → model millimetres
    rms_mm: float
    inliers: float         # share of facade points within the inlier distance after ICP


# ---------------------------------------------------------------------------
# Model front surface
# ---------------------------------------------------------------------------

def _grid_to_mm(cells: np.ndarray) -> np.ndarray:
    """(N, 3) voxel (x, y, z) → mm (x, y, z)."""
    return cells * np.array([STUD_MM, PLATE_MM, STUD_MM])


def model_front_points(solid: np.ndarray, axis: str, sign: int) -> np.ndarray:
    """Centres of the first solid cell seen from the camera side, for every
    (u, y) column of the view, as mm points on the face toward the camera."""
    H, X, Z = solid.shape
    pts = []
    if axis == "x":                                   # camera looks along z
        for y in range(H):
            sl = solid[y]                             # (X, Z)
            if not sl.any():
                continue
            for u in np.flatnonzero(sl.any(axis=1)):
                zs = np.flatnonzero(sl[u])
                d = zs.max() + 1 if sign > 0 else zs.min()
                pts.append((u + 0.5, y + 0.5, d))
    else:                                             # camera looks along x
        for y in range(H):
            sl = solid[y]
            if not sl.any():
                continue
            for u in np.flatnonzero(sl.any(axis=0)):
                xs = np.flatnonzero(sl[:, u])
                d = xs.max() + 1 if sign > 0 else xs.min()
                pts.append((d, y + 0.5, u + 0.5))
    return _grid_to_mm(np.array(pts, dtype=np.float64))


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def _initial_transform(P: np.ndarray, Q: np.ndarray, axis: str, sign: int, mirror: bool,
                       scale: float) -> np.ndarray:
    """Camera (x right, y down, z forward) → model mm for one side.  The
    facade's horizontal axis goes to the grid's `axis`, its up to +y, its
    depth toward the camera to the side the camera is on; translation
    aligns the horizontal centre, the ground and the median front."""
    R = np.zeros((3, 3))
    h = 0 if axis == "x" else 2                       # model column for the facade's horizontal
    dcol = 2 if axis == "x" else 0                    # model column for depth
    R[h, 0] = -scale if mirror else scale             # camera x → horizontal
    R[1, 1] = -scale                                  # camera y (down) → model y (up)
    R[dcol, 2] = -sign * scale                        # camera z (away) → away from the camera side
    Pm = P @ R.T
    t = np.zeros(3)
    t[h] = np.median(Q[:, h]) - np.median(Pm[:, h])
    t[1] = Q[:, 1].min() - np.percentile(Pm[:, 1], 1)
    t[dcol] = np.median(Q[:, dcol]) - np.median(Pm[:, dcol])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _umeyama(A: np.ndarray, B: np.ndarray, with_scale: bool) -> np.ndarray:
    """Similarity (or rigid) transform mapping A onto B, least squares."""
    ma, mb = A.mean(axis=0), B.mean(axis=0)
    A0, B0 = A - ma, B - mb
    U, S, Vt = np.linalg.svd(A0.T @ B0 / len(A))
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1.0, 1.0, d])
    R = (U @ D @ Vt).T
    s = (S * np.diag(D)).sum() / (A0 ** 2).sum() * len(A) if with_scale else 1.0
    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3] = mb - s * R @ ma
    return T


def _apply(T: np.ndarray, P: np.ndarray) -> np.ndarray:
    return P @ T[:3, :3].T + T[:3, 3]


def register(facade: Facade, solid: np.ndarray, *, iters: int = 20, inlier_mm: float = 24.0,
             allow_scale: bool = True) -> FacadeFit:
    """Try every side of the model (± x, ± z, mirrored or not), ICP each,
    keep the one with the most inliers (then the lowest RMS).  Scale is
    initialised from the object's width in both and refined by ICP only
    within ±15%, since the facade and the model should agree on size up
    to the generator's error."""
    from scipy.spatial import cKDTree
    P = facade.points()
    width_m = float(np.percentile(P[:, 0], 99) - np.percentile(P[:, 0], 1))
    best: Optional[FacadeFit] = None
    for axis in ("x", "z"):
        occ_xz = solid.any(axis=0)
        us = np.flatnonzero(occ_xz.any(axis=1 if axis == "x" else 0))
        width_mm = float(us.max() + 1 - us.min()) * STUD_MM
        scale0 = width_mm / max(1e-6, width_m)
        for sign in (+1, -1):
            Q = model_front_points(solid, axis, sign)
            tree = cKDTree(Q)
            for mirror in (False, True):
                T = _initial_transform(P, Q, axis, sign, mirror, scale0)
                for it in range(iters):
                    Pm = _apply(T, P)
                    dist, idx = tree.query(Pm)
                    keep = dist < (inlier_mm * (3.0 if it < 3 else 1.5))
                    if keep.sum() < 50:
                        break
                    T_step = _umeyama(Pm[keep], Q[idx[keep]], with_scale=allow_scale)
                    # keep the scale within ±15% of the width-based estimate
                    s_total = np.cbrt(abs(np.linalg.det((T_step @ T)[:3, :3])))
                    if not (0.85 * scale0 <= s_total <= 1.15 * scale0):
                        T_step = _umeyama(Pm[keep], Q[idx[keep]], with_scale=False)
                    T = T_step @ T
                Pm = _apply(T, P)
                dist, _ = tree.query(Pm)
                inl = float((dist < inlier_mm).mean())
                rms = float(np.sqrt(np.mean(np.minimum(dist, inlier_mm * 2) ** 2)))
                fit = FacadeFit(axis=axis, sign=sign, mirror=mirror, T=T, rms_mm=rms, inliers=inl)
                if best is None or (fit.inliers, -fit.rms_mm) > (best.inliers, -best.rms_mm):
                    best = fit
    assert best is not None
    return best


# ---------------------------------------------------------------------------
# The measured front, and applying it
# ---------------------------------------------------------------------------

def measured_front(facade: Facade, fit: FacadeFit, grid_shape: tuple[int, int, int]
                   ) -> tuple[np.ndarray, np.ndarray]:
    """From the registered points: `front[u, y]` — the depth coordinate
    (grid units) of the front face per view column, taking the point
    nearest the camera in each cell; `top[u]` — the object's top plate per
    column.  NaN where nothing was measured."""
    H, X, Z = grid_shape
    P = facade.pts[facade.mask]
    P = P[np.isfinite(P).all(axis=1)].astype(np.float64)
    M = _apply(fit.T, P) / np.array([STUD_MM, PLATE_MM, STUD_MM])
    h = 0 if fit.axis == "x" else 2
    dcol = 2 if fit.axis == "x" else 0
    U = X if fit.axis == "x" else Z
    D = Z if fit.axis == "x" else X
    u = np.floor(M[:, h]).astype(int)
    y = np.floor(M[:, 1]).astype(int)
    d = M[:, dcol]
    ok = (u >= 0) & (u < U) & (y >= 0) & (y < H) & (d > -2) & (d < D + 2)
    u, y, d = u[ok], y[ok], d[ok]
    front = np.full((U, H), -np.inf if fit.sign > 0 else np.inf)
    if fit.sign > 0:
        np.maximum.at(front, (u, y), d)
        front[np.isneginf(front)] = np.nan
    else:
        np.minimum.at(front, (u, y), d)
        front[np.isposinf(front)] = np.nan
    top = np.full(U, -1.0)
    np.maximum.at(top, u, y.astype(float))
    top[top < 0] = np.nan
    return front, top


def smoothed_front(solid: np.ndarray, facade: Facade, fit: FacadeFit
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The measured front map, smoothed, plus the skyline and the model's
    own front map: (front, top, model_front), each in grid units.

    A facade is made of planes; the measurement is made of pixels.  A
    median over 3 studs x 3 courses keeps a bay or a porch (studs wide,
    storeys tall) and drops the per-pixel jitter that would otherwise
    turn every wall into ribbons on neighbouring lines."""
    from scipy.ndimage import median_filter
    H, X, Z = solid.shape
    front, top = measured_front(facade, fit, solid.shape)
    U = X if fit.axis == "x" else Z
    model_front = np.full((U, H), np.nan)
    for u in range(U):
        for y in range(H):
            c = solid[y, u, :] if fit.axis == "x" else solid[y, :, u]
            idx = np.flatnonzero(c)
            if len(idx):
                model_front[u, y] = idx.max() + 1 if fit.sign > 0 else idx.min()
    filled = np.where(np.isnan(front), model_front, front)
    filled = np.where(np.isnan(filled), np.nanmedian(filled), filled)
    smooth = median_filter(filled, size=(3, 9), mode="nearest")
    return np.where(np.isnan(front), np.nan, smooth), top, model_front


def apply_facade(solid: np.ndarray, facade: Facade, fit: FacadeFit, *,
                 max_shift: int = 10, min_cells: int = 3) -> tuple[np.ndarray, int]:
    """Override the model's front with the measured one.

    Skyline first: columns are cut down to the measured top (a generator
    that stretched a tower).  Then for each (u, y) cell column with a
    measurement the front face moves to the measured depth: cells in
    front of it are cleared, cells behind it up to the old front are
    filled.  Columns thinner than `min_cells` after the move keep their
    old front (a measurement through a window reveal must not slice a
    wall away).  Shifts are capped at `max_shift` studs.  Returns the new
    volume and the number of cells changed."""
    H, X, Z = solid.shape
    front, top, model_front = smoothed_front(solid, facade, fit)
    out = solid.copy()
    U = X if fit.axis == "x" else Z
    D = Z if fit.axis == "x" else X
    changed = 0

    def column(u: int, y: int) -> np.ndarray:
        return out[y, u, :] if fit.axis == "x" else out[y, :, u]

    # a column's measured top is the max over a small neighbourhood (a
    # single sampled pixel column misses the apex of a thin feature)
    top_s = top.copy()
    for u in range(U):
        lo, hi = max(0, u - 1), min(U, u + 2)
        vals = top[lo:hi]
        if np.isfinite(vals).any():
            top_s[u] = np.nanmax(vals)
    for u in range(U):
        if np.isnan(top_s[u]):
            continue
        t = int(np.clip(round(top_s[u]), 0, H - 1))
        sl = out[:, u, :] if fit.axis == "x" else out[:, :, u]
        changed += int(sl[t + 1:].sum())
        sl[t + 1:] = False

    for u in range(U):
        for y in range(H):
            f = front[u, y]
            if np.isnan(f):
                continue
            c = column(u, y)
            idx = np.flatnonzero(c)
            if len(idx) == 0:
                continue
            have = int(idx.max()) if fit.sign > 0 else int(idx.min())
            # the face sits on a cell boundary: for the +side a face at
            # coordinate f means the last solid cell is round(f) - 1
            want = int(round(f)) - 1 if fit.sign > 0 else int(round(f))
            want = int(np.clip(want, have - max_shift, have + max_shift))
            want = int(np.clip(want, 0, D - 1))
            if want == have:
                continue
            back = int(idx.min()) if fit.sign > 0 else int(idx.max())
            if (fit.sign > 0 and want - back + 1 < min_cells) or \
                    (fit.sign < 0 and back - want + 1 < min_cells):
                continue
            if fit.sign > 0:
                if want > have:
                    c[have + 1:want + 1] = True
                else:
                    c[want + 1:have + 1] = False
            else:
                if want < have:
                    c[want:have] = True
                else:
                    c[have:want] = False
            changed += abs(want - have)
    return out, changed


def wall_recess(fit: FacadeFit, raw_front: np.ndarray, *, axis_x: bool, other: int,
                origin: int, length: int, y0: int, height: int,
                min_depth: float = 0.6, max_depth: float = 2.5) -> Optional[np.ndarray]:
    """For a wall on the camera side, the (length, height) mask of its
    cells that the UNSMOOTHED measurement puts between `min_depth` and
    `max_depth` studs behind the wall's face — window reveals, a recessed
    door; deeper is not a reveal but something else behind the wall line
    (the roof above the eave, the gap beside a bay).  None when the wall
    is not on the measured side or not at the measured front (its face
    more than a stud off the measurement's median there)."""
    if axis_x != (fit.axis == "x"):
        return None
    U, H = raw_front.shape
    face = other + 1 if fit.sign > 0 else other
    us = np.arange(origin, origin + length)
    ys = np.arange(y0, y0 + height)
    ok_u = (us >= 0) & (us < U)
    ok_y = (ys >= 0) & (ys < H)
    if not ok_u.any() or not ok_y.any():
        return None
    sub = np.full((length, height), np.nan)
    sub[np.ix_(ok_u, ok_y)] = raw_front[np.ix_(us[ok_u], ys[ok_y])]
    if np.isfinite(sub).mean() < 0.3:
        return None
    med = np.nanmedian(sub)
    if abs(med - face) > 1.0:
        return None
    depth = np.nan_to_num((face - sub) if fit.sign > 0 else (sub - face), nan=0.0)
    return (depth >= min_depth) & (depth <= max_depth)
