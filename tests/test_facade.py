"""facade: register a measured front to the model and override its front
and skyline — on a synthetic block whose generator 'stretched the tower'."""

import numpy as np

from everything2lego.facade import Facade, carve_facade, register, silhouette_iou


def _solid(tower_top: int) -> np.ndarray:
    H, X, Z = 60, 30, 20
    s = np.zeros((H, X, Z), bool)
    s[0:30, 2:28, 2:14] = True            # main block, front face at z=14 (camera on +z)
    s[0:tower_top, 8:14, 2:16] = True     # tower (off-centre, as real houses are), 2 studs proud
    return s


TOWER = slice(8, 14)
TOWER_FRONT_X = 11
PLAIN = range(16, 27)                     # main-block columns clear of the tower


def _facade_from(solid: np.ndarray, *, cam_dist_m: float = 10.0, mm_per_m: float = 50.0,
                 noise_mm: float = 2.0, yaw_deg: float = 4.0, px: int = 4,
                 orbit: bool = False) -> Facade:
    """A metric camera-frame point map of `solid` (x right, y down, z
    away) as a pinhole camera on the +z side would measure it, by
    marching a ray through the voxels for every pixel — `px` pixels per
    stud at the front, a slight yaw so the fit has something to recover,
    the background a wall far behind."""
    H, X, Z = solid.shape
    S = mm_per_m
    # camera: at the model's horizontal centre, mid-height, cam_dist in front of z=0, yawed
    yaw = np.deg2rad(yaw_deg)
    Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    if orbit:
        # walked around the object: the camera orbits its centre
        centre = np.array([X * 4.0, H * 1.6, Z * 4.0])
        C = centre + Ry @ np.array([0.0, 0.0, cam_dist_m * S - Z * 4.0])   # mm, model frame
    else:
        # panned on the spot in front of the +z side
        C = np.array([X * 4.0, H * 1.6, cam_dist_m * S])
    # camera axes in the model frame: right, down, forward (toward -z)
    right, down, fwd = Ry @ np.array([1.0, 0, 0]), np.array([0, -1.0, 0]), Ry @ np.array([0, 0, -1.0])
    # the frame is half again as wide and tall as the object, so the yawed
    # view keeps the whole object with background around it
    W, Hh = int(X * px * 1.5), int(H * 3.2 / 8.0 * px * 1.5)
    f = (cam_dist_m * S - 14 * 8.0) / 8.0 * px                  # focal: px pixels per stud at the front
    rng = np.random.default_rng(0)
    vv, uu = np.mgrid[0:Hh, 0:W]
    dirs = (fwd[None, None] + ((uu + 0.5 - W / 2) / f)[..., None] * right[None, None]
            + ((vv + 0.5 - Hh / 2) / f)[..., None] * down[None, None])
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    hit = np.zeros((Hh, W), bool)
    pts_model = np.zeros((Hh, W, 3))
    t = np.full((Hh, W), 0.0)
    step = 1.0
    for _ in range(int(cam_dist_m * S * 1.2 / step)):
        t += step
        p = C[None, None] + dirs * t[..., None]
        gx, gy, gz = np.floor(p[..., 0] / 8.0).astype(int), np.floor(p[..., 1] / 3.2).astype(int), np.floor(p[..., 2] / 8.0).astype(int)
        inside = (gx >= 0) & (gx < X) & (gy >= 0) & (gy < H) & (gz >= 0) & (gz < Z)
        now = np.zeros((Hh, W), bool)
        now[inside] = solid[gy[inside], gx[inside], gz[inside]]
        new = now & ~hit
        pts_model[new] = p[new]
        hit |= new
        if hit.all():
            break
    far = C[None, None] + dirs * (cam_dist_m * S * 2.5)
    pts_model[~hit] = far[~hit]
    pts_model[hit] += rng.normal(0, noise_mm, pts_model[hit].shape)
    rel = pts_model - C
    cam = np.stack([rel @ right, rel @ down, rel @ fwd], axis=-1) / S
    return Facade(pts=cam.astype(np.float32), mask=hit)


def test_register_finds_the_photographed_side_and_scale():
    solid = _solid(tower_top=45)
    fac = _facade_from(solid)
    fit = register(fac, solid)
    assert fit.axis == "x" and fit.sign == 1 and not fit.mirror
    assert fit.inliers > 0.9 and fit.rms_mm < 6.0
    scale = np.cbrt(abs(np.linalg.det(fit.T[:3, :3])))
    assert abs(scale - 50.0) / 50.0 < 0.05
    assert silhouette_iou(solid, fac, fit) > 0.8


def _check_cut(fixed: np.ndarray, truth: np.ndarray) -> None:
    tower_cols = fixed[:, TOWER, :].any(axis=(1, 2))
    top = int(np.flatnonzero(tower_cols).max())
    assert 43 <= top <= 46                           # measured 45 (± a plate of sampling)
    # the main block's front stays where it was measured: z = 14
    fronts = [np.flatnonzero(fixed[15, x]).max() + 1 for x in PLAIN]
    assert all(f == 14 for f in fronts), fronts
    # and the tower is still 2 studs proud
    assert np.flatnonzero(fixed[15, TOWER_FRONT_X]).max() + 1 == 16
    # the rest of the block is untouched (back, sides, interior)
    assert (fixed ^ truth).sum() < 0.05 * truth.sum()


def test_carve_cuts_the_tower_by_projection_and_keeps_the_front():
    """The 3D carve: every cell projected into the photo — the sky above
    the tower empties it, the measured front stays, nothing else moves."""
    truth = _solid(tower_top=45)
    fac = _facade_from(truth)
    stretched = _solid(tower_top=58)
    fit = register(fac, stretched)
    fixed, changed = carve_facade(stretched, fac, fit)
    assert changed > 0
    _check_cut(fixed, truth)


def test_carve_brings_a_sunken_front_forward_and_cuts_a_proud_one():
    """The generator put the main block's front 2 studs too deep on the
    left and 2 studs too proud on the right: the measured front wins."""
    truth = _solid(tower_top=45)
    fac = _facade_from(truth)
    wrong = truth.copy()
    wrong[0:30, 2:8, 12:14] = False                  # left: sunk to z=12
    wrong[0:30, 22:28, 14:16] = True                 # right: proud to z=16
    fit = register(fac, wrong)
    assert fit.axis == "x" and fit.sign == 1
    fixed, _ = carve_facade(wrong, fac, fit)
    left = [np.flatnonzero(fixed[15, x]).max() + 1 for x in range(3, 7)]
    right = [np.flatnonzero(fixed[15, x]).max() + 1 for x in range(23, 27)]
    assert all(f == 14 for f in left), left
    # a proud face is cut back to within the carve's half-stud ambiguity
    assert all(14 <= f <= 15 for f in right), right
    # and the rest of the front (which was right) stays put
    mid = [np.flatnonzero(fixed[15, x]).max() + 1 for x in range(16, 21)]
    assert all(f == 14 for f in mid), mid


def test_register_is_not_fooled_by_a_part_the_generator_misplaced():
    """A whole side proud by 2 studs must not make the fit shrink the
    facade onto a patch of the model or pick another heading."""
    truth = _solid(tower_top=45)
    fac = _facade_from(truth)
    wrong = truth.copy()
    wrong[0:30, 22:28, 14:16] = True
    fit = register(fac, wrong)
    assert fit.axis == "x" and fit.sign == 1
    scale = np.cbrt(abs(np.linalg.det(fit.T[:3, :3])))
    assert abs(scale - 50.0) / 50.0 < 0.06


def test_carve_drops_what_it_cuts_loose():
    """A cap above the photo's top edge is unseen — but once the tower
    under it is carved away it floats, and goes."""
    truth = _solid(tower_top=45)
    fac = _facade_from(truth)
    fac = Facade(pts=fac.pts[3:], mask=fac.mask[3:])  # crop the top 3 rows of the photo
    stretched = _solid(tower_top=58)
    fit = register(fac, stretched)
    fixed, _ = carve_facade(stretched, fac, fit)
    from scipy.ndimage import label
    _, n = label(fixed)
    assert n == 1
    assert int(np.flatnonzero(fixed[:, TOWER, :].any(axis=(1, 2))).max()) <= 46


def test_dominant_colour_reads_the_material_not_a_vote_of_snapped_cells():
    """Sunlit and shaded brick snap to two palette entries; a grey base
    snaps to one.  Voting after snapping lets the grey win; the material
    is the brick."""
    from everything2lego.colors import dominant_colour, palette, snap_rgb_to_palette
    rng = np.random.default_rng(0)
    brick = np.array([150, 120, 90]) + rng.normal(0, 14, (600, 3))
    grey = np.array([100, 102, 100]) + rng.normal(0, 4, (450, 3))
    rgb = np.clip(np.vstack([brick, grey]), 0, 255).astype(np.uint8)
    pal = palette()
    votes = np.bincount(snap_rgb_to_palette(rgb), minlength=len(pal))
    got = dominant_colour(rgb)
    grey_idx = snap_rgb_to_palette(np.array([[100, 102, 100]], np.uint8))[0]
    assert votes.argmax() == grey_idx                # the snapped vote says grey
    assert got != grey_idx
    assert pal[got].name in ("Dark Tan", "Medium Nougat", "Light Brown", "Nougat")


def test_photo_colours_the_seen_front_and_leaves_the_back():
    truth = _solid(tower_top=45)
    fac = _facade_from(truth)
    H, W = fac.mask.shape
    fac.rgb = np.zeros((H, W, 3), np.uint8)
    fac.rgb[...] = (200, 40, 40)                       # a red front
    fit = register(fac, truth)
    from everything2lego.facade import photo_colours
    from everything2lego.colors import snap_rgb_to_palette
    pi = np.full(truth.shape, -1, np.int16)
    pi[truth] = 0
    out, changed = photo_colours(truth, pi, fac, fit)
    red = snap_rgb_to_palette(np.array([[200, 40, 40]], np.uint8))[0]
    assert changed > 0
    assert out[15, 20, 13] == red                      # the main block's front face
    assert out[15, 20, 2] == 0                         # its back: unseen, unchanged


def test_square_walls_restores_a_right_angle_a_three_quarter_view_bent():
    """A ¾ view of a box, its depth bent the way a wrong focal length
    bends it (the two walls meet at ~80°): after square_walls they meet
    square again, and the fit still registers both sides."""
    from everything2lego.facade import _apply, _dominant_wall, square_walls
    box = np.zeros((40, 30, 30), bool)
    box[0:30, 4:26, 4:26] = True
    fac = _facade_from(box, yaw_deg=40.0, cam_dist_m=12.0, orbit=True)
    bent = fac.pts.copy()
    bent[..., 0] *= 0.8                                   # a focal-length error
    fac = Facade(pts=bent, mask=fac.mask)
    fit = register(fac, box)

    def angle(f):
        P = _apply(fit.T, f.points(max_points=20000))
        n1, c1, _ = _dominant_wall(P, min_share=0.1)
        rest = P[np.abs((P - c1) @ n1) > 10]
        n2, _, _ = _dominant_wall(rest, min_share=0.1)
        return float(np.degrees(np.arccos(abs(n1 @ n2))))

    before = angle(fac)
    fixed, corrections = square_walls(fac, fit)
    after = angle(fixed)
    assert abs(before - 90) > 4, before
    assert corrections
    assert abs(after - 90) < 1.5, after
