"""Roof refinement: the slope family follows the measured pitch, long parts
cover a face, ridges are tiled, corners get corner slopes, and every swap
keeps the voxel volume and a single supported assembly."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from everything2lego import pack, stability, voxelize
from everything2lego.io_ldraw import _ROT

COLOR = 4
FOUNDATION = 2  # plates: one slab under everything, so connectivity is the roof's doing


def _grid_from_tops(tops: list[int], depth: int) -> voxelize.VoxelGrid:
    """Solid columns; `tops[x]` is the top plate index of column x above the
    foundation slab, the same for every z."""
    tops_arr = np.asarray(tops) + FOUNDATION
    arr = np.full((int(tops_arr.max()) + 1, len(tops), depth), -1, dtype=np.int16)
    for x, t in enumerate(tops_arr):
        arr[:t + 1, x, :] = COLOR
    return voxelize.VoxelGrid(palette_idx=arr, metadata={})


def _cells(placements: list[pack.Placement]) -> set[tuple[int, int, int]]:
    return {(p.y_plate + dy, p.x_stud + dx, p.z_stud + dz)
            for p in placements
            for dy in range(p.part.height_units)
            for dx in range(p.w_world) for dz in range(p.d_world)}


def _pack_both(grid: voxelize.VoxelGrid, **kw):
    kw.setdefault("foundation_layers", FOUNDATION)
    plain = pack.pack(grid, use_slopes=False, **kw)
    roofed = pack.pack(grid, use_slopes=True, **kw)
    return plain, roofed


def _assert_sound(grid: voxelize.VoxelGrid, plain, roofed) -> None:
    """Volume-preserving and still one supported assembly."""
    occupied = {tuple(c) for c in np.argwhere(grid.palette_idx >= 0)}
    assert _cells(roofed) == _cells(plain) == occupied
    report = stability.check(roofed)
    assert report.ok, (report.components, len(report.floating), report.overlaps)


def _slopes(placements, shape="slope"):
    return [p for p in placements if p.part.shape == shape]


def _angle(p: pack.Placement) -> int:
    import re
    m = re.search(r"(?:Slope|Sloped) (\d+)", p.part.name)
    assert m, p.part.name
    return int(m.group(1))


def test_one_plate_per_stud_gets_cheese_not_45():
    grid = _grid_from_tops([0, 1, 2, 3, 4, 5], depth=4)
    plain, roofed = _pack_both(grid)
    slopes = _slopes(roofed)
    assert slopes, "a shallow plate staircase should still get sloped parts"
    assert all(p.part.ldraw == "54200.dat" for p in slopes), Counter(p.part.ldraw for p in slopes)
    assert all(_angle(p) <= 33 for p in slopes)
    # Each cheese sits on the step's edge column with its low side downhill (-x).
    assert all(pack.slope_world_low(p) == "-x" for p in slopes)
    assert len(slopes) >= 4 * 4
    _assert_sound(grid, plain, roofed)


def test_one_brick_per_two_studs_gets_33_degree_parts():
    grid = _grid_from_tops([2, 2, 5, 5, 8, 8, 11, 11], depth=4)
    plain, roofed = _pack_both(grid)
    slopes = _slopes(roofed)
    thirty_three = [p for p in slopes if _angle(p) == 33]
    assert len(thirty_three) >= 2
    for p in thirty_three:
        assert p.w_world == 3, "a 33° slope is three studs long along its run"
        assert pack.slope_world_low(p) == "-x"
    # The step edges that have three studs behind them are 33°, not 45°.
    at_edges = {p.x_stud: _angle(p) for p in slopes if pack.slope_world_low(p) == "-x"}
    assert at_edges[2] == 33 and at_edges[4] == 33
    assert not any(_angle(p) == 30 for p in slopes)
    _assert_sound(grid, plain, roofed)


def test_one_brick_per_stud_gets_45_degree_parts():
    grid = _grid_from_tops([2, 5, 8, 11, 14, 17], depth=4)
    plain, roofed = _pack_both(grid)
    slopes = _slopes(roofed)
    assert len(slopes) >= 4
    assert all(_angle(p) == 45 for p in slopes), Counter(p.part.name for p in slopes)
    assert all(p.w_world == 2 and pack.slope_world_low(p) == "-x" for p in slopes)
    # The whole face is a few long parts, not many 1-wide ones.
    assert all(p.d_world == 4 for p in slopes)
    _assert_sound(grid, plain, roofed)


@pytest.mark.parametrize("rise,expected", [(6, {"3678b.dat"}), (9, {"3684.dat"})])
def test_steep_runs_get_tall_slopes(rise, expected):
    """Two (three) bricks of rise per stud: 65° (75°) slopes that replace the
    stack of bricks at each step."""
    grid = _grid_from_tops([rise - 1 + rise * k for k in range(4)], depth=2)
    plain, roofed = _pack_both(grid)
    slopes = _slopes(roofed)
    assert {p.part.ldraw for p in slopes} == expected, Counter(p.part.ldraw for p in slopes)
    assert len(slopes) == 2
    for p in slopes:
        assert p.part.height_units == rise
        assert pack.slope_world_low(p) == "-x"
    _assert_sound(grid, plain, roofed)


def test_gable_roof_ridge_is_tiled_and_faces_are_long_slopes():
    grid = _grid_from_tops([2, 5, 8, 11, 11, 8, 5, 2], depth=4)
    plain, roofed = _pack_both(grid)
    top = int(grid.palette_idx.shape[0]) - 1
    ridge_top = [p for p in roofed
                 if p.y_plate + p.part.height_units - 1 == top]
    assert ridge_top and all(p.part.shape == "tile" for p in ridge_top), \
        Counter(p.part.name for p in ridge_top)
    assert {(x, z) for p in ridge_top for x in range(p.x_stud, p.x_stud + p.w_world)
            for z in range(p.z_stud, p.z_stud + p.d_world)} == {(x, z) for x in (3, 4) for z in range(4)}
    # No studded brick is left on the ridge course.
    assert not any(p.part.shape == "brick" and p.y_plate + 2 == top for p in roofed)
    slopes = _slopes(roofed)
    assert Counter(pack.slope_world_low(p) for p in slopes) == {"+x": 2, "-x": 2}
    assert all(p.part.ldraw == "3037.dat" for p in slopes), Counter(p.part.ldraw for p in slopes)
    _assert_sound(grid, plain, roofed)


def test_seams_alternate_between_courses():
    """A face six studs long is two 2x4 + 2x2 courses whose seam moves."""
    grid = _grid_from_tops([2, 5, 8, 11, 14], depth=6)
    plain, roofed = _pack_both(grid)
    by_course: dict[int, list[int]] = {}
    for p in _slopes(roofed):
        by_course.setdefault(p.y_plate, []).append(p.z_stud + p.d_world)
    seams = {y: sorted(v)[:-1] for y, v in by_course.items()}
    assert len(seams) >= 3
    assert len({tuple(s) for s in seams.values()}) >= 2, seams
    _assert_sound(grid, plain, roofed)


def test_outside_corner_gets_double_convex_and_valley_gets_double_concave():
    # Hip: a 2x2 tower on a corner of a slab (outside corner).
    arr = np.full((8, 6, 6), -1, dtype=np.int16)
    arr[0:2] = COLOR
    arr[2:5, 0:6, 0:6] = COLOR
    arr[5:8, 0:2, 0:2] = COLOR
    grid = voxelize.VoxelGrid(palette_idx=arr, metadata={})
    plain, roofed = _pack_both(grid)
    corners = _slopes(roofed, "slope_corner")
    assert [c.part.ldraw for c in corners] == ["3045.dat"]
    assert pack.slope_world_low(corners[0]) == "+x+z"
    _assert_sound(grid, plain, roofed)

    # Valley: two 45° gables meeting in an L.
    ridge, xa, zb, half = 14, 6, 6, 4
    arr = np.full((ridge + 1, 12, 12), -1, dtype=np.int16)
    for x in range(12):
        for z in range(12):
            h = -1
            if abs(x - xa) <= half and z <= zb:
                h = max(h, ridge - 3 * abs(x - xa))
            if abs(z - zb) <= half and x <= xa:
                h = max(h, ridge - 3 * abs(z - zb))
            if h >= 0:
                arr[:h + 1, x, z] = COLOR
    arr = np.pad(arr, ((FOUNDATION, 0), (0, 0), (0, 0)), constant_values=-1)
    arr[:FOUNDATION][:, arr[FOUNDATION] >= 0] = COLOR
    grid = voxelize.VoxelGrid(palette_idx=arr, metadata={})
    plain, roofed = _pack_both(grid)
    valleys = [c for c in _slopes(roofed, "slope_corner") if c.part.ldraw == "3046.dat"]
    assert valleys, Counter(p.part.ldraw for p in roofed)
    for v in valleys:
        assert pack.slope_world_low(v) == "-x-z"
        # The low corner lies on the valley diagonal.
        assert v.x_stud - xa == v.z_stud - zb
    assert not any(c.part.ldraw == "3676.dat" for c in _slopes(roofed, "slope_corner"))
    _assert_sound(grid, plain, roofed)


def test_corner_rotation_cycle_matches_ldraw_matrices():
    """`_CORNER_LOW_CYCLE` is the (+x, +z) corner carried through `_ROT`."""
    for i, world in enumerate(pack._CORNER_LOW_CYCLE):
        m = _ROT[i * 90]
        wx = m[0] + m[2]      # image of the (1, 0, 1) corner direction, x component
        wz = m[6] + m[8]      # z component
        assert world == ("+x" if wx > 0 else "-x") + ("+z" if wz > 0 else "-z")
    corner = next(p for p in pack.parts_by_shape("slope_corner") if p.ldraw == "3045.dat")
    assert [pack._slope_rotation(corner, (2, 2), d) for d in pack._CORNER_LOW_CYCLE] == [0, 90, 180, 270]


def test_slope_families_have_the_geometry_their_names_state():
    fams = pack._slope_families()
    assert fams["45"].parts[1].ldraw == "3040.dat" and fams["45"].parts[4].ldraw == "3037.dat"
    assert fams["33"].parts[2].ldraw == "3298.dat" and fams["33"].parts[1].ldraw == "4286.dat"
    assert fams["cheese"].parts[1].ldraw == "54200.dat"
    assert fams["65"].parts[2].ldraw == "3678b.dat" and fams["75"].parts[2].ldraw == "3684.dat"
    for fam in fams.values():
        for width, part in fam.parts.items():
            assert part.footprint == (fam.run, width) and part.height_units == fam.height, part
