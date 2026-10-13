"""Per-region palette lock: planar regions are owned cell-by-cell by the
nearest plane, lock to their dominant colour(s), and the colour target is
snapped so the lock is attainable."""

from types import SimpleNamespace as NS

import numpy as np

from everything2lego.semantic_v2 import (SoftAssembly, _build_region_cell_masks,
                                         _snap_target_to_region_palette)
from everything2lego.semantic_v2.regions import RegionLabel

EYE = NS(mesh_to_voxel=np.eye(4, dtype=np.float32))


def _plane(label, x_lo, x_hi, plane_x, **extra):
    """A planar region whose box spans x∈[x_lo, x_hi] and whose plane is
    x = plane_x (normal +x)."""
    return NS(label=label,
              bounds=(np.array([x_lo, 0, 0], np.float32), np.array([x_hi, 9, 9], np.float32)),
              plane_normal=np.array([1.0, 0, 0]), plane_origin=np.array([plane_x, 0, 0]),
              **extra)


def _grid():
    Y = X = Z = 10
    occ = np.ones((Y, X, Z), bool)
    pi = np.full((Y, X, Z), 3)
    rng = np.random.default_rng(0)
    r = rng.random((Y, 4, Z))                               # wall x∈[0,4): 80/15/5 dither
    pi[:, :4, :] = np.where(r < 0.80, 3, np.where(r < 0.95, 5, 7))
    pi[:, 6:, :] = rng.integers(0, 3, (Y, 4, Z)) + 10        # detail x∈[6,10): three colours
    return occ, pi


def test_planar_region_locks_to_dominant_shade_and_detail_gets_no_lock():
    occ, pi = _grid()
    wall = _plane(RegionLabel.PLANAR_WALL, 0, 3.0, 0.5)
    detail = NS(label=RegionLabel.DETAIL,
                bounds=(np.array([6, 0, 0], np.float32), np.array([9, 9, 9], np.float32)))
    masks = _build_region_cell_masks(EYE, [wall, detail], pi, occ)
    assert len(masks) == 1                                   # detail regions are unrestricted
    mask, allowed = masks[0]
    assert allowed == frozenset({3})
    assert not mask[:, 6:, :].any()


def test_two_tone_planar_region_keeps_both():
    occ, pi = _grid()
    pi[:, :4, :] = np.where(np.arange(4)[None, :, None] % 2 == 0, 3, 5)   # 50/50 by column
    (_, allowed), = _build_region_cell_masks(EYE, [_plane(RegionLabel.PLANAR_ROOF, 0, 3.0, 0.5)], pi, occ)
    assert allowed == frozenset({3, 5})


def test_snapped_target_is_attainable_under_the_lock():
    occ, pi = _grid()
    pi[0, 0, 0] = -1                                         # an empty cell inside the wall
    masks = _build_region_cell_masks(EYE, [_plane(RegionLabel.PLANAR_WALL, 0, 3.0, 0.5)], pi, occ)
    (mask, allowed), = masks
    assert allowed == frozenset({3})
    snapped = _snap_target_to_region_palette(pi, masks)
    inside = snapped[mask]
    assert set(np.unique(inside[inside >= 0])) == {3}
    assert snapped[0, 0, 0] == -1
    assert np.array_equal(snapped[~mask], pi[~mask])


def test_overlapping_boxes_are_owned_by_nearest_plane():
    """Boxes overlap on x∈[2,4).  The plane at x=0.5 owns x∈{0,1,2}, the one
    at x=4.8 owns x∈{3,4,5}: masks are disjoint and each locks to its own
    shade despite dither in the overlap."""
    Y = X = Z = 10
    occ = np.ones((Y, X, Z), bool)
    pi = np.where(np.arange(X)[None, :, None] < 3, 3, 5) * np.ones((Y, 1, Z), int)
    pi[:, 2, :5] = 5                                         # 1/6 of A's cells wrong
    pi[:, 3, :5] = 3                                         # 1/6 of B's cells wrong
    a = _plane(RegionLabel.PLANAR_WALL, 0, 3.0, 0.5)         # box x∈[0,4)
    b = _plane(RegionLabel.PLANAR_WALL, 2, 5.0, 4.8)         # box x∈[2,6)
    masks = _build_region_cell_masks(EYE, [a, b], pi, occ)
    assert len(masks) == 2
    (ma, aa), (mb, ab) = masks
    assert not (ma & mb).any()
    assert set(np.nonzero(ma)[1]) == {0, 1, 2}
    assert set(np.nonzero(mb)[1]) == {3, 4, 5}
    assert (aa, ab) == (frozenset({3}), frozenset({5}))
    snapped = _snap_target_to_region_palette(pi, masks)
    assert (snapped[:, :3, :] == 3).all() and (snapped[:, 3:6, :] == 5).all()
    # The lock in the assembly agrees with the ownership.
    sa = SoftAssembly.from_placements((Y, X, Z), [], 12, per_region_palette=masks)
    logits = sa.color_logits.detach()
    assert logits[0, 2, 0, 3] > -1e9 and logits[0, 2, 0, 5] < -1e9   # owned by A: only 3
    assert logits[0, 3, 0, 5] > -1e9 and logits[0, 3, 0, 3] < -1e9   # owned by B: only 5
    assert logits[0, 8, 0, 7] > -1e9                                   # outside all boxes: free


def test_plane_is_derived_from_points_when_region_lacks_it():
    Y = X = Z = 6
    occ = np.ones((Y, X, Z), bool)
    pi = np.full((Y, X, Z), 3)
    reg = NS(label=RegionLabel.PLANAR_ROOF,
             bounds=(np.zeros(3, np.float32), np.array([5, 5, 5], np.float32)),
             points=np.array([[0, 2.0, 0], [5, 2.0, 5]]),
             normals=np.array([[0, 1.0, 0], [0, 1.0, 0]]))
    masks = _build_region_cell_masks(EYE, [reg], pi, occ)
    assert len(masks) == 1 and masks[0][1] == frozenset({3})
