"""Structural tests for the InteriorVolumeFiller (semantic_v2.interior).

Small synthetic occupancy grids, no mesh loading.
"""

import numpy as np

from everything2lego.bricks import catalog
from everything2lego.semantic_v2 import interior
from everything2lego.semantic_v2.interior import InteriorConfig, InteriorVolumeFiller
from everything2lego.semantic_v2.mesh_target import MeshTarget
from everything2lego.voxelize import VoxelGrid


def _target(occ: np.ndarray) -> MeshTarget:
    """A MeshTarget whose only meaningful field is the voxel grid."""
    grid = VoxelGrid(np.where(occ, 0, -1).astype(np.int16))
    empty = np.zeros((0, 3), dtype=np.float32)
    return MeshTarget(mesh=None, max_studs=max(occ.shape[1:]),
                      surface_points=empty, surface_normals=empty,
                      surface_colors=empty.astype(np.uint8),
                      voxel_grid=grid, mesh_to_voxel=np.eye(4))


def _coverage(placements, shape):
    counts = np.zeros(shape, dtype=int)
    for p in placements:
        counts[p.y_plate:p.y_plate + p.part.height_units,
               p.x_stud:p.x_stud + p.w_world,
               p.z_stud:p.z_stud + p.d_world] += 1
    return counts


def _x_seams(placements, y, X):
    """(x-boundary, z) pairs where a 3-plate brick on course `y` ends."""
    seams = set()
    for p in placements:
        if p.y_plate != y or p.part.height_units != 3:
            continue
        for z in range(p.z_stud, p.z_stud + p.d_world):
            if p.x_stud > 0:
                seams.add((p.x_stud, z))
            if p.x_stud + p.w_world < X:
                seams.add((p.x_stud + p.w_world, z))
    return seams


def _shared_seam_fraction(placements, shape):
    Y, X, _ = shape
    fractions = []
    for y in range(0, Y - 3, 3):
        below, above = _x_seams(placements, y, X), _x_seams(placements, y + 3, X)
        if below:
            fractions.append(len(below & above) / len(below))
    assert fractions, "need at least two brick courses"
    return float(np.mean(fractions))


def test_consecutive_courses_do_not_share_seams():
    """Before the running bond was added every course was tiled by the same
    greedy scan, so on a solid 9x20x12 (and 12x17x13) grid the fraction of
    x-boundaries shared between course y and y+3 was 1.0 -- every brick sat
    exactly on the brick below it.  Staggering the scan per course gives 0.0.
    """
    for shape in [(9, 20, 12), (12, 17, 13)]:
        remaining = np.ones(shape, dtype=bool)
        placements = interior._promote_interior_bricks(remaining, 0)
        assert _shared_seam_fraction(placements, shape) < 0.25


def test_2x6_brick_is_used_for_a_2x6_run():
    remaining = np.zeros((3, 6, 2), dtype=bool)
    remaining[:] = True
    placements = interior._promote_interior_bricks(remaining, 0)
    assert [p.part.ldraw for p in placements] == ["2456.dat"]
    assert not remaining.any()
    # In a wider slab 2x6 still shows up where a 2x6 solid run is left over.
    remaining = np.ones((6, 17, 13), dtype=bool)
    placements = interior._promote_interior_bricks(remaining, 0)
    assert any(p.part.ldraw == "2456.dat" for p in placements)


def test_interior_brick_list_is_area_descending():
    areas = [w * d for _, (w, d) in interior._INTERIOR_BRICKS]
    assert areas == sorted(areas, reverse=True)
    known = {p.ldraw for p in catalog()}
    assert all(ldraw in known for ldraw, _ in interior._INTERIOR_BRICKS)


def test_fill_covers_every_target_cell_exactly_once():
    occ = np.ones((7, 11, 9), dtype=bool)
    occ[:, 4:6, 3:5] = False           # a hole through the volume
    occ[5:, :3] = False                # an uneven top
    placements = InteriorVolumeFiller().fill(_target(occ), [])
    counts = _coverage(placements, occ.shape)
    assert np.array_equal(counts, occ.astype(int))
    assert all(p.part.shape in ("brick", "plate", "brick_tall") for p in placements)


def test_hollow_keeps_2x2_columns_and_course_aligned_tall_bricks():
    occ = np.ones((33, 24, 20), dtype=bool)
    cfg = InteriorConfig(hollow=True, shell_thickness=2, floor_spacing=12,
                         column_spacing=8, foundation_plates=3)
    keep = interior._hollow_with_supports(
        occ.copy(), occ, shell_thickness=2, floor_spacing=12,
        column_spacing=8, foundation_plates=3)
    assert (keep <= occ).all() and keep.sum() < occ.sum()
    # Interior column at (8, 8) spans a 2x2 footprint, nothing wider.
    assert keep[6, 8:10, 8:10].all()
    assert not keep[6, 10, 8] and not keep[6, 8, 10] and not keep[6, 7, 8]
    assert keep[12].all() and keep[:3].all()      # floor slab + foundation

    placements = InteriorVolumeFiller(cfg).fill(_target(occ), [])
    assert np.array_equal(_coverage(placements, occ.shape), keep.astype(int))
    tall = [p for p in placements if p.part.height_units == 15]
    assert tall and all(p.y_plate % 3 == 0 for p in tall)
    assert any(p.part.ldraw == "30360.dat" for p in tall)
