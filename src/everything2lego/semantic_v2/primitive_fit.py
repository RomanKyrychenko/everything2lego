"""Primitive fitting — RANSAC sphere and cylinder over mesh faces.

Given a set of face centroids and face normals, try to fit a sphere or a
cylinder. Returns a fit-quality score and the geometric parameters.

Used by RegionDetector on faces that weren't captured by `trimesh.facets`
(curved / organic geometry).

Two independent fitters:
    fit_sphere(centroids, normals) → SphereFit or None
    fit_cylinder(centroids, normals) → CylinderFit or None

Both use a linear least-squares closed-form for robustness (no random
sampling) — a sphere has 4 free parameters (cx, cy, cz, r), a cylinder
has 5 (axis direction unit vector + point on axis + radius).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SphereFit:
    center: np.ndarray  # (3,) float32
    radius: float
    residual_mm: float  # mean |distance - radius|


@dataclass
class CylinderFit:
    axis: np.ndarray    # (3,) float32 unit vector
    center: np.ndarray  # point on axis
    radius: float
    residual_mm: float


def fit_sphere(centroids: np.ndarray, normals: np.ndarray | None = None
                ) -> SphereFit | None:
    """Algebraic least-squares sphere fit (Kasa's method).

    Minimise Σ (|p - c|² - r²)² over c ∈ ℝ³, r ∈ ℝ.  Linearised as
    Ax = b with x = [cx, cy, cz, k] and k = r² - |c|².
    """
    if len(centroids) < 4:
        return None
    P = np.asarray(centroids, dtype=np.float64)
    A = np.concatenate([2 * P, np.ones((len(P), 1))], axis=1)
    b = (P * P).sum(axis=1)
    try:
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    c = x[:3]
    k = x[3]
    r2 = k + float((c * c).sum())
    if r2 <= 0:
        return None
    r = float(np.sqrt(r2))
    distances = np.linalg.norm(P - c, axis=1)
    residual = float(np.abs(distances - r).mean())
    return SphereFit(center=c.astype(np.float32), radius=r, residual_mm=residual)


def fit_cylinder(centroids: np.ndarray, normals: np.ndarray
                    ) -> CylinderFit | None:
    """Cylinder fit via normal-space PCA.

    For a cylindrical surface, all outward normals are perpendicular to the
    cylinder axis. So the axis is the eigenvector of the normal-covariance
    matrix with the SMALLEST eigenvalue.  Given the axis, project centroids
    onto the plane ⊥ axis and fit a circle for the radius + center.
    """
    if len(centroids) < 6:
        return None
    N = np.asarray(normals, dtype=np.float64)
    N = N / (np.linalg.norm(N, axis=1, keepdims=True) + 1e-9)
    cov = N.T @ N / len(N)
    try:
        vals, vecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return None
    axis = vecs[:, 0]  # smallest eigenvalue
    axis /= (np.linalg.norm(axis) + 1e-9)

    # Project centroids to plane ⊥ axis.
    P = np.asarray(centroids, dtype=np.float64)
    P0 = P.mean(axis=0)
    proj = P - np.outer((P - P0) @ axis, axis)  # (N, 3)
    # In that plane, fit a 2-D circle. Build a 2-D basis (u, v) ⊥ axis.
    tmp = (np.array([1.0, 0.0, 0.0])
            if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0]))
    u = np.cross(axis, tmp); u /= (np.linalg.norm(u) + 1e-9)
    v = np.cross(axis, u);   v /= (np.linalg.norm(v) + 1e-9)
    uv = np.stack([(proj - P0) @ u, (proj - P0) @ v], axis=1)  # (N, 2)
    # Kasa 2-D circle fit.
    A = np.concatenate([2 * uv, np.ones((len(uv), 1))], axis=1)
    b = (uv * uv).sum(axis=1)
    try:
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cu, cv, k = x
    r2 = k + cu * cu + cv * cv
    if r2 <= 0:
        return None
    r = float(np.sqrt(r2))
    residuals = np.abs(np.linalg.norm(uv - np.array([cu, cv]), axis=1) - r)
    residual = float(residuals.mean())
    center_3d = (P0 + cu * u + cv * v).astype(np.float32)
    return CylinderFit(axis=axis.astype(np.float32),
                        center=center_3d,
                        radius=r,
                        residual_mm=residual)
