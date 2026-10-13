"""facade: register a measured front to the model and override its front
and skyline — on a synthetic block whose generator 'stretched the tower'."""

import numpy as np

from everything2lego.facade import Facade, apply_facade, register


def _solid(tower_top: int) -> np.ndarray:
    H, X, Z = 60, 30, 20
    s = np.zeros((H, X, Z), bool)
    s[0:30, 2:28, 2:14] = True            # main block, front face at z=14 (camera on +z)
    s[0:tower_top, 12:18, 2:16] = True    # tower, 2 studs proud of the front
    return s


def _facade_from(solid: np.ndarray, *, cam_dist_m: float = 10.0, mm_per_m: float = 50.0,
                 noise_mm: float = 2.0) -> Facade:
    """A camera-frame point map of the +z front of `solid`: x right, y
    down, z away — a slight yaw so the fit has something to recover."""
    H, X, Z = solid.shape
    pts, mask = [], []
    rng = np.random.default_rng(0)
    yaw = np.deg2rad(4.0)
    rows = []
    for y in range(H - 1, -1, -1):               # image rows top→bottom
        row, mrow = [], []
        for x in range(X):
            zs = np.flatnonzero(solid[y, x])
            if len(zs) == 0:
                row.append((0.0, 0.0, 0.0)); mrow.append(False)
                continue
            front = zs.max() + 1
            X_mm, Y_mm, Zf = (x + 0.5) * 8.0, (y + 0.5) * 3.2, front * 8.0
            # camera looks along -z of the model: camera z = distance - model z
            cx = (X_mm - X * 4.0) / mm_per_m
            cy = -(Y_mm - H * 1.6) / mm_per_m
            cz = cam_dist_m - Zf / mm_per_m
            cx, cz = cx * np.cos(yaw) - cz * np.sin(yaw), cx * np.sin(yaw) + cz * np.cos(yaw)
            row.append((cx + rng.normal(0, noise_mm / 1000), cy, cz + rng.normal(0, noise_mm / 1000)))
            mrow.append(True)
        pts.append(row); mask.append(mrow)
    return Facade(pts=np.array(pts, np.float32), mask=np.array(mask))


def test_register_finds_the_photographed_side_and_scale():
    solid = _solid(tower_top=45)
    fac = _facade_from(solid)
    fit = register(fac, solid)
    assert fit.axis == "x" and fit.sign == 1 and not fit.mirror
    assert fit.inliers > 0.9 and fit.rms_mm < 6.0
    scale = np.cbrt(abs(np.linalg.det(fit.T[:3, :3])))
    assert abs(scale - 50.0) / 50.0 < 0.05


def test_apply_cuts_a_stretched_tower_to_the_measured_skyline_and_keeps_the_front():
    truth = _solid(tower_top=45)
    fac = _facade_from(truth)
    stretched = _solid(tower_top=58)                 # the generator's tower, 13 plates too tall
    fit = register(fac, stretched)
    fixed, changed = apply_facade(stretched, fac, fit)
    assert changed > 0
    tower_cols = fixed[:, 12:18, :].any(axis=(1, 2))
    top = int(np.flatnonzero(tower_cols).max())
    assert 43 <= top <= 46                           # measured 45 (± a plate of sampling)
    # the main block's front stays where it was measured: z = 14
    fronts = [np.flatnonzero(fixed[15, x]).max() + 1 for x in range(3, 11)]
    assert all(f == 14 for f in fronts), fronts
    # and the tower is still 2 studs proud
    assert np.flatnonzero(fixed[15, 15]).max() + 1 == 16
