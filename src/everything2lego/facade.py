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
  3. carve the solid volume with it: every cell projected into the
     photo is kept, emptied or filled by what its pixel measured.

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
# background pixels are only trusted on a registration at least this good:
# a matte with holes puts "nothing here" rays through the object
SKYLINE_MIN_INLIERS = 0.9


def solid_volume(occ: np.ndarray) -> np.ndarray:
    from .semantic_v2.massing import solid_volume as _impl
    return _impl(occ)


@dataclass
class Facade:
    pts: np.ndarray        # (H, W, 3) metric, camera frame
    mask: np.ndarray       # (H, W) bool
    conf: Optional[np.ndarray] = None   # (H, W) the depth model's confidence, when saved
    rgb: Optional[np.ndarray] = None    # (H, W, 3) uint8 the photo, on the point map's grid

    @classmethod
    def load(cls, path: str) -> "Facade":
        d = np.load(path)
        pts = d["pts3d"] if "pts3d" in d else d["0_pts3d"]
        mask = d["mask"] if "mask" in d else d["0_mask"]
        conf = d["conf"] if "conf" in d else (d["0_conf"] if "0_conf" in d else None)
        rgb = d["rgb"] if "rgb" in d else None
        return cls(pts=np.asarray(pts).squeeze(), mask=np.asarray(mask).squeeze().astype(bool),
                   conf=None if conf is None else np.asarray(conf).squeeze().astype(np.float32),
                   rgb=None if rgb is None else np.asarray(rgb).astype(np.uint8))

    @property
    def valid(self) -> np.ndarray:
        """Object pixels with a measured depth.  The matte is the object's
        silhouette; a depth model returns nothing (z = 0) on some of it —
        dark timber, glass — and those pixels must not become points at
        the camera."""
        z = self.pts[..., 2]
        return self.mask & np.isfinite(self.pts).all(axis=-1) & (z > 0.1)

    def points(self, max_points: int = 20000, seed: int = 0) -> np.ndarray:
        P = self.pts[self.valid]
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
    R[h, 0] = scale                                   # camera x → horizontal
    R[1, 1] = -scale                                  # camera y (down) → model y (up)
    R[dcol, 2] = -sign * scale                        # camera z (away) → away from the camera side
    # a camera is right-handed (right × down = forward): the horizontal
    # flips with the side it looks from, or the heading is a reflection
    if (np.linalg.det(R) < 0) != mirror:
        R[h, 0] = -R[h, 0]
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


def model_surface_points(solid: np.ndarray) -> np.ndarray:
    """The centres of every exposed cell FACE of the solid, in mm — the
    whole skin, for a photo taken from any direction.  Faces, not cell
    centres: a measured surface lies on the boundary between a solid and
    an empty cell, and registering it to centres would pull it half a
    cell into the model."""
    pad = np.pad(solid, 1)
    out = []
    for axis, step in ((1, 1), (1, -1), (0, 1), (0, -1), (2, 1), (2, -1)):   # x±, y±, z±
        nb = np.roll(pad, -step, axis=axis)[1:-1, 1:-1, 1:-1]
        face = solid & ~nb
        cells = np.argwhere(face).astype(np.float64)   # (y, x, z)
        pts = np.c_[cells[:, 1] + 0.5, cells[:, 0] + 0.5, cells[:, 2] + 0.5]
        g = {1: 0, 0: 1, 2: 2}[axis]                   # grid column of that axis
        pts[:, g] += 0.5 * step
        out.append(pts)
    return _grid_to_mm(np.vstack(out))


def _dominant_wall(P: np.ndarray, *, tol_mm: float = 6.0, trials: int = 400, seed: int = 0,
                   min_share: float = 0.4) -> Optional[tuple[np.ndarray, np.ndarray, float]]:
    """RANSAC for the VERTICAL plane most of `P` (N, 3, model mm, y up)
    lies on — (horizontal unit normal, centroid, inlier share), or None
    when no wall holds 40% of the points.  Vertical by construction:
    walls are, and a free plane through a facade with a roof, a ledge or
    a step seen at a grazing angle tilts to catch them."""
    if len(P) < 50:
        return None
    rng = np.random.default_rng(seed)
    best, best_n = None, 0
    xz = P[:, [0, 2]]
    for _ in range(trials):
        a, b = xz[rng.choice(len(P), 2, replace=False)]
        t = b - a
        nt = np.linalg.norm(t)
        if nt < 1e-6:
            continue
        n2 = np.array([-t[1], t[0]]) / nt
        d = np.abs((xz - a) @ n2)
        cnt = int((d < tol_mm).sum())
        if cnt > best_n:
            best, best_n = (n2, a), cnt
    if best is None or best_n < min_share * len(P):
        return None
    n2, a = best
    inl = (np.abs((xz - a) @ n2) < tol_mm)
    c2 = xz[inl].mean(axis=0)
    # refine: the smallest principal axis of the inliers' footprint
    _, _, Vt = np.linalg.svd(xz[inl] - c2, full_matrices=False)
    n2 = Vt[-1]
    n = np.array([n2[0], 0.0, n2[1]])
    c = np.array([c2[0], P[inl, 1].mean(), c2[1]])
    return n, c, best_n / len(P)


def _rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The smallest rotation taking unit vector a onto unit vector b."""
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = float(a @ b)
    if s < 1e-12:
        return np.eye(3) if c > 0 else -np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / s ** 2)


def _snap_facade_plane(T: np.ndarray, P: np.ndarray, *, max_deg: float = 12.0
                       ) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Buildings' walls lie on the grid: the facade's dominant measured
    plane, once registered, should face a grid axis exactly.  A frontal
    photo cannot tell a camera yaw from a front the generator skewed, so
    ICP splits the difference — this resolves it the way a set designer
    would: the big measured wall is flat and square.  Rotates T about
    the plane's centroid so its normal meets the nearest horizontal axis
    (a yaw about the vertical); leaves T alone when the facade has no
    dominant wall or the correction is large (not a wall at all)."""
    Pm = _apply(T, P)
    plane = _dominant_wall(Pm)
    if plane is None:
        return T, None
    n, c, _ = plane
    on_wall = np.abs((Pm - c) @ n) < 6.0
    axes = [np.array(v, float) for v in ((1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1))]
    target = max(axes, key=lambda a: abs(float(a @ n)))
    if float(target @ n) < 0:
        n = -n
    ang = np.degrees(np.arccos(np.clip(float(target @ n), -1, 1)))
    if ang > max_deg:
        return T, None
    if ang < 0.05:
        return T, on_wall
    R = _rotation_between(n, target)
    S = np.eye(4)
    S[:3, :3] = R
    S[:3, 3] = c - R @ c
    return S @ T, on_wall


def _refine_frame_by_silhouette(solid: np.ndarray, facade: Facade, fit: FacadeFit,
                                K: tuple[float, float, float, float]) -> FacadeFit:
    """ICP seats surfaces but is blind to edges: a photo's outline fixes
    where the object starts and stops — its height, its width, where it
    stands — better than any surface match.  Coordinate descent on the
    vertical and horizontal offsets and the scale, maximising the
    silhouette overlap."""
    best = fit
    best_iou = silhouette_iou(solid, facade, fit, K)
    s0 = float(np.cbrt(abs(np.linalg.det(fit.T[:3, :3]))))
    right = fit.T[:3, 0] / s0
    # the point the scale pivots on: the facade's ground centre
    Pm = _apply(fit.T, facade.points(max_points=5000))
    pivot = np.array([np.median(Pm[:, 0]), np.percentile(Pm[:, 1], 1), np.median(Pm[:, 2])])

    def moved(T, dx_mm, dy_mm, ds):
        T2 = T.copy()
        T2[:3, 3] += right * dx_mm + np.array([0.0, dy_mm, 0.0])
        if ds != 0.0:
            Sc = np.eye(4)
            Sc[:3, :3] *= 1.0 + ds
            Sc[:3, 3] = pivot - (1.0 + ds) * pivot
            T2 = Sc @ T2
        return T2

    for step_mm, step_s in ((PLATE_MM * 2, 0.02), (PLATE_MM, 0.01), (PLATE_MM * 0.5, 0.005)):
        improved = True
        while improved:
            improved = False
            for dx, dy, ds in ((0, step_mm, 0), (0, -step_mm, 0), (step_mm, 0, 0), (-step_mm, 0, 0),
                               (0, 0, step_s), (0, 0, -step_s)):
                cand = FacadeFit(best.axis, best.sign, best.mirror, moved(best.T, dx, dy, ds), best.rms_mm, best.inliers)
                iou = silhouette_iou(solid, facade, cand, K)
                if iou > best_iou + 1e-4:
                    best, best_iou, improved = cand, iou, True
    return best


def _heading(T: np.ndarray) -> tuple[str, int]:
    """(axis, sign) of the grid side a camera transform looks at: the
    facade's horizontal runs along `axis`; sign +1 = the camera is on the
    high side of the other axis."""
    f = T[:3, :3] @ np.array([0.0, 0.0, 1.0])            # camera forward, model frame
    if abs(f[2]) >= abs(f[0]):
        return "x", (1 if f[2] < 0 else -1)
    return "z", (1 if f[0] < 0 else -1)


def _initial_transform_yaw(P: np.ndarray, Q: np.ndarray, yaw: float, width_m: float
                           ) -> tuple[np.ndarray, float]:
    """Camera (x right, y down, z forward) → model mm with the camera
    looking along -z turned by `yaw` about the vertical; the scale from
    the model's extent across that view over the facade's width; the
    facade's near side seated on the model's near side, centred across
    the view, standing on the model's ground."""
    c, s_ = np.cos(yaw), np.sin(yaw)
    Ry = np.array([[c, 0, s_], [0, 1, 0], [-s_, 0, c]])
    base = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])   # right→x, down→-y, forward→-z
    R = Ry @ base
    right, fwd = R[:, 0], R[:, 2]
    proj = Q @ right
    scale = float(np.percentile(proj, 99) - np.percentile(proj, 1)) / max(width_m, 1e-6)
    Pm = P @ (scale * R).T
    t = np.zeros(3)
    t += right * (np.median(Q @ right) - np.median(Pm @ right))
    t += fwd * (np.percentile(Q @ fwd, 5) - np.percentile(Pm @ fwd, 5))
    t[1] = Q[:, 1].min() - np.percentile(Pm[:, 1], 1)
    T = np.eye(4)
    T[:3, :3] = scale * R
    T[:3, 3] = t
    return T, scale


def register(facade: Facade, solid: np.ndarray, *, iters: int = 20, inlier_mm: float = 24.0,
             allow_scale: bool = True) -> FacadeFit:
    """Register the facade's point cloud to the model's whole skin by ICP
    in isotropic millimetres, starting from each of eight headings (four
    sides, four corners — a camera is never a mirror, so reflections are
    not tried).  Scale is pinned by the object's width across the view.
    The heading kept is the one whose model silhouette and relief agree
    with the photo."""
    from scipy.spatial import cKDTree
    P = facade.points()
    width_m = float(np.percentile(P[:, 0], 99) - np.percentile(P[:, 0], 1))
    Q = model_surface_points(solid)
    tree = cKDTree(Q)
    K = camera_intrinsics(facade)
    best: Optional[FacadeFit] = None
    best_score = -1.0
    for yaw_deg in range(0, 360, 45):
        # every heading the photo could be from: face-on to a side, or a
        # ¾ view onto a corner (ICP does not turn 45° on its own)
        T, scale0 = _initial_transform_yaw(P, Q, np.radians(yaw_deg), width_m)
        for it in range(iters):
            Pm = _apply(T, P)
            dist, idx = tree.query(Pm)
            # anneal: wide to catch, then tight so a majority face
            # wins over a part the generator misplaced
            band = 3.0 if it < 3 else (1.5 if it < iters - 5 else 0.4)
            keep = dist < inlier_mm * band
            if keep.sum() < 50:
                break
            T_step = _umeyama(Pm[keep], Q[idx[keep]], with_scale=allow_scale)
            # scale is pinned by the object's width, not by ICP (a
            # frontal view barely constrains it): the model's extent
            # across the camera's current heading over the facade's
            # measured width, ±8%
            Tn = T_step @ T
            s_total = np.cbrt(abs(np.linalg.det(Tn[:3, :3])))
            right = Tn[:3, 0] / s_total
            proj = Q @ right
            prior = float(np.percentile(proj, 99) - np.percentile(proj, 1)) / max(width_m, 1e-6)
            if not (0.92 * prior <= s_total <= 1.08 * prior):
                T_step = _umeyama(Pm[keep], Q[idx[keep]], with_scale=False)
                Tn = T_step @ T
                s_rigid = np.cbrt(abs(np.linalg.det(Tn[:3, :3])))
                s_want = float(np.clip(s_rigid, 0.92 * prior, 1.08 * prior))
                if abs(s_want - s_rigid) > 1e-9:
                    c = Pm[keep].mean(axis=0)
                    Sc = np.eye(4)
                    Sc[:3, :3] *= s_want / s_rigid
                    Sc[:3, 3] = c - (s_want / s_rigid) * c
                    Tn = Sc @ Tn
            T = Tn
        # square the measured wall to the grid, then re-seat it in
        # depth robustly: the median residual, so the part of the
        # front most of the measurements agree with wins over a
        # part the generator misplaced
        T, on_wall = _snap_facade_plane(T, P)
        if on_wall is not None and on_wall.sum() >= 50:
            for _ in range(5):
                Pm = _apply(T, P)
                dist, idx = tree.query(Pm)
                keep = on_wall & (dist < inlier_mm * 0.75)
                if keep.sum() < 50:
                    break
                step = np.median(Q[idx[keep]] - Pm[keep], axis=0)
                if np.linalg.norm(step) < 0.1:
                    break
                T = T.copy()
                T[:3, 3] += step
        Pm = _apply(T, P)
        dist, _ = tree.query(Pm)
        inl = float((dist < inlier_mm).mean())
        rms = float(np.sqrt(np.mean(np.minimum(dist, inlier_mm * 2) ** 2)))
        axis, sign = _heading(T)
        fit = FacadeFit(axis=axis, sign=sign, mirror=False, T=T, rms_mm=rms, inliers=inl)
        if K:
            # the outline fixes the frame (height, width, where it
            # stands) that surface matching leaves loose
            fit = _refine_frame_by_silhouette(solid, facade, fit, K)
            Pm = _apply(fit.T, P)
            dist, _ = tree.query(Pm)
            fit.inliers = float((dist < inlier_mm).mean())
            fit.rms_mm = float(np.sqrt(np.mean(np.minimum(dist, inlier_mm * 2) ** 2)))
            rms, inl = fit.rms_mm, fit.inliers
        # the heading whose model outline AND relief match the
        # photo: the facade's own ICP inliers are fooled by a fit
        # shrunk onto a patch of the model, and a symmetric outline
        # by the wrong side
        if K:
            score = (silhouette_iou(solid, facade, fit, K) * depth_agreement(solid, facade, fit, K)
                     * max(0.0, 1.0 - rms / (2.0 * inlier_mm)))
        else:
            score = inl * max(0.0, 1.0 - rms / (2.0 * inlier_mm))
        if best is None or score > best_score:
            best, best_score = fit, score
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
    P = facade.pts[facade.valid].astype(np.float64)
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


def camera_intrinsics(facade: Facade) -> Optional[tuple[float, float, float, float]]:
    """(fx, fy, cx, cy) of the pinhole that produced the point map, read
    off the measured pixels (x/z is linear in the column, y/z in the row).
    None when there is too little to fit."""
    z = facade.pts[..., 2]
    valid = facade.mask & np.isfinite(z) & (z > 0.1)
    if valid.sum() < 100:
        return None
    vv, uu = np.nonzero(valid)
    ax, bx = np.polyfit(uu, facade.pts[valid][:, 0] / z[valid], 1)   # x/z = (u - cx) / fx
    ay, by = np.polyfit(vv, facade.pts[valid][:, 1] / z[valid], 1)
    return 1.0 / ax, 1.0 / ay, -bx / ax, -by / ay


def _project_cells(centres: np.ndarray, facade: Facade, fit: FacadeFit,
                   K: tuple[float, float, float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Grid-unit cell centres → (u, v, z_camera, seen): the pixel each
    lands on through the registered camera, its depth, and whether it is
    inside the photo and in front of the camera."""
    fx, fy, cx, cy = K
    Hh, Ww = facade.mask.shape
    scale_mm = np.array([STUD_MM, PLATE_MM, STUD_MM])
    P_cam = _apply(np.linalg.inv(fit.T), centres * scale_mm)
    zc = P_cam[:, 2]
    infront = zc > 0.05
    u = np.full(len(zc), -1)
    v = np.full(len(zc), -1)
    u[infront] = np.round(fx * P_cam[infront, 0] / zc[infront] + cx).astype(int)
    v[infront] = np.round(fy * P_cam[infront, 1] / zc[infront] + cy).astype(int)
    seen = infront & (u >= 0) & (u < Ww) & (v >= 0) & (v < Hh)
    return u, v, zc, seen


def silhouette_iou(solid: np.ndarray, facade: Facade, fit: FacadeFit,
                   K: Optional[tuple[float, float, float, float]] = None) -> float:
    """How well the model's outline, seen through the registered camera,
    matches the photo's matte: intersection over union of the two
    silhouettes.  The one number that says a registration is right —
    ICP residuals alone are fooled by a facade shrunk onto a patch of a
    bigger model."""
    K = K or camera_intrinsics(facade)
    if K is None:
        return 0.0
    from scipy.ndimage import binary_dilation
    cells = np.argwhere(solid)                        # (y, x, z)
    centres = np.c_[cells[:, 1] + 0.5, cells[:, 0] + 0.5, cells[:, 2] + 0.5]
    u, v, _, seen = _project_cells(centres, facade, fit, K)
    sil = np.zeros(facade.mask.shape, bool)
    sil[v[seen], u[seen]] = True
    sil = binary_dilation(sil, iterations=1)         # cells are wider than a pixel
    inter = (sil & facade.mask).sum()
    union = (sil | facade.mask).sum()
    return float(inter / max(union, 1))


def depth_agreement(solid: np.ndarray, facade: Facade, fit: FacadeFit,
                    K: Optional[tuple[float, float, float, float]] = None, *, tol_studs: float = 1.5) -> float:
    """The share of the matte's pixels whose measured depth agrees (within
    `tol_studs`) with the model's nearest cell on that pixel, seen through
    the registered camera.  With the silhouette, what decides between
    headings: a symmetric outline seen from the wrong side has the wrong
    relief."""
    K = K or camera_intrinsics(facade)
    if K is None:
        return 0.0
    cells = np.argwhere(solid)
    centres = np.c_[cells[:, 1] + 0.5, cells[:, 0] + 0.5, cells[:, 2] + 0.5]
    u, v, zc, seen = _project_cells(centres, facade, fit, K)
    Hh, Ww = facade.mask.shape
    zbuf = np.full(Hh * Ww, np.inf)
    flat = v[seen] * Ww + u[seen]
    np.minimum.at(zbuf, flat, zc[seen])
    zbuf = zbuf.reshape(Hh, Ww)
    zm = facade.pts[..., 2]
    ok = facade.mask & np.isfinite(zm) & (zm > 0.1)
    if ok.sum() == 0:
        return 0.0
    s_mm_per_m = float(np.cbrt(abs(np.linalg.det(fit.T[:3, :3]))))
    # a cell is wider than a pixel: spread each cell's depth over its footprint
    from scipy.ndimage import minimum_filter
    cell_px = K[0] * STUD_MM / s_mm_per_m / max(float(np.median(zm[ok])), 1e-6)
    zbuf = minimum_filter(zbuf, size=int(np.ceil(cell_px)) + 1, mode="nearest")
    tol = tol_studs * STUD_MM / s_mm_per_m
    agree = ok & np.isfinite(zbuf) & (np.abs(zbuf - zm) < tol)
    return float(agree.sum() / ok.sum())


def seen_mask(cells: np.ndarray, facade: Facade, fit: FacadeFit,
              K: Optional[tuple[float, float, float, float]] = None, *, tol_studs: float = 1.0
              ) -> np.ndarray:
    """Which of `cells` (N, 3 grid (x, y, z) centres) the photo actually
    shows: each lands on an object pixel whose measured depth is the
    cell's own (within `tol_studs`), not something in front of it.  What
    the photo shows has the photo's colours; the rest of a reconstruction
    was inpainted."""
    K = K or camera_intrinsics(facade)
    if K is None or len(cells) == 0:
        return np.zeros(len(cells), bool)
    u, v, zc, seen = _project_cells(np.asarray(cells, np.float64), facade, fit, K)
    out = np.zeros(len(cells), bool)
    if not seen.any():
        return out
    ui, vi = u[seen], v[seen]
    zm = facade.pts[vi, ui, 2].astype(np.float64)
    s_mm_per_m = float(np.cbrt(abs(np.linalg.det(fit.T[:3, :3]))))
    tol = tol_studs * STUD_MM / s_mm_per_m
    out[np.flatnonzero(seen)] = facade.mask[vi, ui] & np.isfinite(zm) & (np.abs(zm - zc[seen]) < tol)
    return out


def _blurred_photo(facade: Facade, fit: FacadeFit, K) -> Optional[np.ndarray]:
    """The photo averaged over a cell's footprint (a cell covers several
    pixels: its colour is their mean, not one pixel of mortar)."""
    if facade.rgb is None:
        return None
    from scipy.ndimage import uniform_filter
    s_mm_per_m = float(np.cbrt(abs(np.linalg.det(fit.T[:3, :3]))))
    zs_ok = facade.pts[..., 2][facade.valid]
    cell_px = K[0] * STUD_MM / s_mm_per_m / max(float(np.nanmedian(zs_ok)), 1e-6)
    size = max(1, int(round(cell_px)))
    return uniform_filter(facade.rgb.astype(np.float32), size=(size, size, 1), mode="nearest")


def seen_rgb(cells: np.ndarray, facade: Facade, fit: FacadeFit,
             K: Optional[tuple[float, float, float, float]] = None, *, tol_studs: float = 1.0,
             photo: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
    """(seen mask, (n_seen, 3) uint8 photo colours) for `cells` (N, 3 grid
    (x, y, z) centres): what the photo shows of them, and in what colour."""
    K = K or camera_intrinsics(facade)
    empty = (np.zeros(len(cells), bool), np.zeros((0, 3), np.uint8))
    if K is None or facade.rgb is None or len(cells) == 0:
        return empty
    photo = photo if photo is not None else _blurred_photo(facade, fit, K)
    u, v, zc, inside = _project_cells(np.asarray(cells, np.float64), facade, fit, K)
    seen = seen_mask(cells, facade, fit, K, tol_studs=tol_studs)
    if not seen.any():
        return empty
    rgb = np.clip(photo[v[seen], u[seen]], 0, 255).astype(np.uint8)
    return seen, rgb


def photo_colours(solid: np.ndarray, palette_idx: np.ndarray, facade: Facade, fit: FacadeFit,
                  *, tol_studs: float = 1.0) -> tuple[np.ndarray, int]:
    """The photo is the truth wherever it sees the model: every surface
    cell of `solid` the photo shows (see `seen_mask`) takes the palette
    colour of its own pixel.  A reconstruction's colours are the photo's
    after a projection, a decimation and a quantisation; sunlit brick
    comes out of that as a patchwork of grey, black and trim colour.
    Cells the photo does not show keep the reconstruction's colour.
    Returns the recoloured palette index grid and the cells changed."""
    if facade.rgb is None:
        return palette_idx, 0
    K = camera_intrinsics(facade)
    if K is None:
        return palette_idx, 0
    from scipy.ndimage import binary_erosion
    from .colors import snap_rgb_to_palette
    skin = solid & ~binary_erosion(solid, iterations=1, border_value=0)
    cells = np.argwhere(skin)                                    # (y, x, z)
    centres = np.c_[cells[:, 1] + 0.5, cells[:, 0] + 0.5, cells[:, 2] + 0.5]
    u, v, zc, inside = _project_cells(centres, facade, fit, K)
    if not inside.any():
        return palette_idx, 0
    ui, vi = u[inside], v[inside]
    zm = facade.pts[vi, ui, 2].astype(np.float64)
    s_mm_per_m = float(np.cbrt(abs(np.linalg.det(fit.T[:3, :3]))))
    tol = tol_studs * STUD_MM / s_mm_per_m
    seen = facade.mask[vi, ui] & np.isfinite(zm) & (np.abs(zm - zc[inside]) < tol)
    if not seen.any():
        return palette_idx, 0
    photo = _blurred_photo(facade, fit, K)
    rgb = np.clip(photo[vi[seen], ui[seen]], 0, 255)
    snapped = snap_rgb_to_palette(rgb.astype(np.uint8)).astype(palette_idx.dtype)
    c = cells[np.flatnonzero(inside)[seen]]
    out = palette_idx.copy()
    changed = int((out[c[:, 0], c[:, 1], c[:, 2]] != snapped).sum())
    out[c[:, 0], c[:, 1], c[:, 2]] = snapped
    return out, changed


def _column_tops(facade: Facade, fit: FacadeFit, K) -> np.ndarray:
    """Per image column, the height (model plates) of the object's top
    edge: the topmost pixel with a depth, rebuilt on its own pixel ray
    (the pixel's position is exact; only its depth is noisy, and that is
    a local median).  +inf where the column has no object."""
    fx, fy, cx, cy = K
    valid = facade.valid
    Hh, Ww = valid.shape
    tops = np.full(Ww, np.inf)
    cols = np.flatnonzero(valid.any(axis=0))
    if len(cols) == 0:
        return tops
    rows = np.argmax(valid[:, cols], axis=0)
    z = np.where(valid, facade.pts[..., 2], np.nan)
    zs = np.empty(len(cols))
    for k, (r, c) in enumerate(zip(rows, cols)):
        win = z[r:min(Hh, r + 3), max(0, c - 1):c + 2]
        zs[k] = np.nanmedian(win)
    P = np.c_[(cols - cx) / fx * zs, (rows - cy) / fy * zs, zs]
    tops[cols] = _apply(fit.T, P)[:, 1] / PLATE_MM
    return tops


def square_walls(facade: Facade, fit: FacadeFit, *, max_deg: float = 20.0,
                 min_share: float = 0.1, band_studs: float = 3.0) -> tuple[Facade, list[float]]:
    """Manhattan regularisation of a measured facade.  A single photo's
    depth bends angles — a ¾ view of a box comes back with its two walls
    meeting at 78°, not 90° — while the set it becomes has square walls.
    Up to two dominant vertical walls are found in the registered points;
    each is turned to its nearest grid axis (when within `max_deg`) about
    their common corner (or its own centre when alone).  Every pixel of a
    wall — the face and what lies within `band_studs` of it, window
    reveals included — keeps its offset from the wall and its own pixel
    ray: only its depth along that ray changes.  Returns the corrected
    facade and the corrections applied (degrees)."""
    K = camera_intrinsics(facade)
    if K is None:
        return facade, []
    fx, fy, cx, cy = K
    valid = facade.valid
    vv, uu = np.nonzero(valid)
    if len(vv) < 200:
        return facade, []
    P = _apply(fit.T, facade.pts[vv, uu].astype(np.float64))
    walls = []
    remaining = np.ones(len(P), bool)
    for _ in range(2):
        idx = np.flatnonzero(remaining)
        if len(idx) < 100:
            break
        w = _dominant_wall(P[idx], min_share=min_share * len(P) / len(idx))
        if w is None:
            break
        n, c, _ = w
        on = np.abs((P - c) @ n) < 6.0
        walls.append((n, c, on))
        remaining &= ~on
    if not walls:
        return facade, []
    axes = [np.array(v, float) for v in ((1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1))]
    pivot = None
    if len(walls) == 2:
        (n1, c1, _), (n2, c2, _) = walls
        A = np.array([[n1[0], n1[2]], [n2[0], n2[2]]])
        if abs(np.linalg.det(A)) > 0.2:                      # not parallel: they meet
            xz = np.linalg.solve(A, [n1 @ c1, n2 @ c2])
            pivot = np.array([xz[0], 0.0, xz[1]])
    scale = float(np.cbrt(abs(np.linalg.det(fit.T[:3, :3]))))
    R = fit.T[:3, :3]
    t0 = fit.T[:3, 3]
    band = band_studs * STUD_MM
    pts = facade.pts.copy()
    ray = np.c_[(uu - cx) / fx, (vv - cy) / fy, np.ones(len(uu))]
    dm = ray @ R.T                                           # model-frame ray per unit camera depth
    best_dist = np.full(len(P), np.inf)
    assign = np.full(len(P), -1)
    corrections: list[float] = []
    targets = []
    for k, (n, c, on) in enumerate(walls):
        target = max(axes, key=lambda a: abs(float(a @ n)))
        if float(target @ n) < 0:
            target = -target
        ang = float(np.degrees(np.arccos(np.clip(target @ n, -1, 1))))
        targets.append((target, ang))
        # this wall's pixels: within the band of its plane, inside its extent
        along = np.cross([0.0, 1.0, 0.0], n)
        s_on = (P[on] - c) @ along
        lo, hi = np.percentile(s_on, 1) - STUD_MM, np.percentile(s_on, 99) + STUD_MM
        top = np.percentile(P[on, 1], 99) + STUD_MM
        d = (P - c) @ n
        sa = (P - c) @ along
        mine = (np.abs(d) < band) & (sa >= lo) & (sa <= hi) & (P[:, 1] <= top)
        closer = mine & (np.abs(d) < best_dist)
        assign[closer] = k
        best_dist[closer] = np.abs(d[closer])
    for k, (n, c, on) in enumerate(walls):
        target, ang = targets[k]
        if ang > max_deg or ang < 0.2:
            continue
        sel = assign == k
        if not sel.any():
            continue
        piv = pivot if pivot is not None else c
        offset = (P[sel] - c) @ n                            # relief relative to the wall
        # the corrected wall: normal `target`, through the pivot's trace of the old wall
        anchor = piv + n * ((c - piv) @ n)
        rhs = offset + target @ anchor - target @ t0
        den = dm[sel] @ target
        ok = np.abs(den) > 1e-6
        z = np.where(ok, rhs / np.where(ok, den, 1.0), np.nan)
        good = ok & (z > 0.1)
        sv, su = vv[sel][good], uu[sel][good]
        pts[sv, su] = ray[sel][good] * z[good, None]
        corrections.append(ang)
    out = Facade(pts=pts, mask=facade.mask, conf=facade.conf, rgb=facade.rgb)
    return out, corrections


def carve_facade(solid: np.ndarray, facade: Facade, fit: FacadeFit, *,
                 background: Optional[bool] = None, margin: float = 0.75, fill_depth: float = 2.0,
                 reach: float = 6.0) -> tuple[np.ndarray, int]:
    """Space carving with the measured depth map, from wherever the photo
    was taken.  Every cell of the grid is projected into the photo through
    the registered camera:

      * it lands on an OBJECT pixel nearer than the measured surface (by
        more than `margin` studs): the cell is in front of the real
        surface — emptied;
      * on an object pixel just behind the surface (within `fill_depth`
        studs), where the model had matter within `reach` studs further
        along the ray: filled, so a surface the generator placed too deep
        comes forward (nothing is created in thin air);
      * on a BACKGROUND pixel, nearer than whatever that pixel measured
        (the sky measures nothing: anything on its ray goes): emptied — a
        tower the generator built taller than the photo shows is cut by
        the sky above it;
      * outside the photo, or behind the camera: unknown — kept.

    Background pixels are only trusted (`background`) on a registration
    with SKYLINE_MIN_INLIERS or more inliers, since a matte with holes
    puts background rays through the object.  Works for a frontal photo
    and a ¾ view alike.  Returns the new volume and the cells changed."""
    H, X, Z = solid.shape
    if background is None:
        background = fit.inliers >= SKYLINE_MIN_INLIERS
    K = camera_intrinsics(facade)
    if K is None:
        return solid.copy(), 0
    Hh, Ww = facade.mask.shape
    scale_mm = np.array([STUD_MM, PLATE_MM, STUD_MM])
    s_mm_per_m = float(np.cbrt(abs(np.linalg.det(fit.T[:3, :3]))))
    cam_grid = fit.T[:3, 3] / scale_mm

    # every cell centre, through the camera
    yy, xx, zz = np.meshgrid(np.arange(H), np.arange(X), np.arange(Z), indexing="ij")
    centres = np.c_[xx.ravel() + 0.5, yy.ravel() + 0.5, zz.ravel() + 0.5]
    u, v, zc, seen = _project_cells(centres, facade, fit, K)
    ui, vi = u[seen], v[seen]
    on_object = facade.mask[vi, ui]
    zmeas = facade.pts[vi, ui, 2].astype(np.float64)
    zmeas = np.where(np.isfinite(zmeas), zmeas, 0.0)
    margin_m = margin * STUD_MM / s_mm_per_m
    fill_m = fill_depth * STUD_MM / s_mm_per_m

    flat_in = solid.ravel().copy()
    out = flat_in.copy()
    idx = np.flatnonzero(seen)
    zs = zc[seen]

    measured = on_object & (zmeas > 0.1)
    # the measured top of the object in each image column, in model
    # plates (the topmost matte pixel with a depth), and how far above it
    # a cell may sit: half a cell plus one pixel's height at that depth
    col_top = _column_tops(facade, fit, K)
    px_plates = np.where(measured, zmeas, 0.0) / K[1] * s_mm_per_m / PLATE_MM
    cell_y = centres[idx, 1]
    above_top = cell_y > col_top[ui] + 0.5 + px_plates
    # 1. in front of the measured surface: empty
    clear = measured & (zs < zmeas - margin_m)
    # 2. just behind it, with matter in reach and not above the column's
    #    measured top: solid
    behind = measured & (zs >= zmeas) & (zs <= zmeas + fill_m) & ~above_top
    if behind.any():
        bidx = idx[behind]
        c = centres[bidx]
        d = c - cam_grid
        d /= np.maximum(np.linalg.norm(d, axis=1)[:, None], 1e-9)
        far = c + d * reach
        fxg = np.clip(np.floor(far[:, 0]).astype(int), 0, X - 1)
        fyg = np.clip(np.floor(far[:, 1]).astype(int), 0, H - 1)
        fzg = np.clip(np.floor(far[:, 2]).astype(int), 0, Z - 1)
        has_matter = flat_in[bidx] | solid[fyg, fxg, fzg]
        fill_idx = bidx[has_matter]
    else:
        fill_idx = np.zeros(0, int)
    # 2b. behind the measured surface but higher than the column's top:
    #     from below a camera cannot see the top of a tower, so a model
    #     rising behind the front's top edge agrees with the photo; a set
    #     designer reads the visible height as THE height.  (What hides
    #     behind something at least as high — the back of a flat roof, a
    #     wall behind a porch — stays.)
    clear |= measured & (zs >= zmeas) & above_top
    # 3. the background: nothing up to what the pixel measured; the sky
    #    (no depth, or above the object's outline in that column — a depth
    #    model gives the sky a confident-looking but meaningless depth,
    #    often NEARER than the object) all the way.
    if background and (~facade.mask).any():
        tops = np.full(Ww, Hh)
        cols = np.flatnonzero(facade.mask.any(axis=0))
        tops[cols] = np.argmax(facade.mask[:, cols], axis=0)
        bgpix = ~on_object
        # sky: above the object's outline in its column.  A background
        # pixel BELOW the outline without a depth is unknown, not sky — a
        # dark wall the matte missed has no depth either, and carving
        # along its whole ray would cut a wedge out of the wall.
        sky = bgpix & (vi < tops[ui])
        if facade.conf is not None:
            lo = float(np.nanpercentile(facade.conf[facade.mask], 5))
            sky |= bgpix & (facade.conf[vi, ui] < lo)
        clear |= sky | (bgpix & (zmeas > 0.1) & (zs < zmeas))
    out[idx[clear]] = False
    out[fill_idx] = True
    out = out.reshape(solid.shape)
    # 4. what the carving cut loose cannot stand: matter above the photo's
    #    top edge, or behind an edge, that only hung on the carved-away
    #    part goes with it.  Pieces on the ground stay.
    from scipy.ndimage import label
    lab, n = label(out)
    if n > 1:
        sizes = np.bincount(lab.ravel())
        grounded = set(np.unique(lab[0])) - {0}
        keep = grounded or {int(np.argmax(sizes[1:])) + 1}
        out &= np.isin(lab, list(keep))
    changed = int((out != solid).sum())
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
