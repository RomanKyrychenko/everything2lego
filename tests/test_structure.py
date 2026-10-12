"""The stability model (which cells offer and take studs) and the support
repair that uses it."""

import numpy as np

from everything2lego import pipeline, stability, voxelize
from everything2lego.bricks import catalog
from everything2lego.colors import palette
from everything2lego.pack import Placement


def part(ldraw):
    return next(p for p in catalog() if p.ldraw == ldraw)


def place(ldraw, x, z, y, w=None, d=None, rot=0, color=4):
    p = part(ldraw)
    w = p.footprint[0] if w is None else w
    d = p.footprint[1] if d is None else d
    return Placement(p, color, x, z, y, w, d, rot)


def test_tiles_and_cheese_slopes_offer_no_studs():
    base = place("3001.dat", 0, 0, 0)                   # 2x4 brick
    tile = place("3068b.dat", 0, 0, 3)                  # 2x2 tile on it
    on_tile = place("3003.dat", 0, 0, 4)                # 2x2 brick on the tile
    r = stability.check([base, tile, on_tile])
    assert [p.part.ldraw for p in r.floating] == ["3003.dat"]
    cheese = place("54200.dat", 0, 0, 3)
    on_cheese = place("3024.dat", 0, 0, 5)
    r = stability.check([base, cheese, on_cheese])
    assert [p.part.ldraw for p in r.floating] == ["3024.dat"]


def test_slope_offers_studs_only_on_its_high_side():
    base = place("3001.dat", 0, 0, 0, 4, 2, 90)         # 4 along x, 2 along z
    slope = place("3040.dat", 0, 0, 3)                  # 2x1, low toward +x: high cell x=0
    on_high = place("3024.dat", 0, 0, 6)
    on_low = place("3024.dat", 1, 0, 6)
    r = stability.check([base, slope, on_high, on_low])
    assert [p.x_stud for p in r.floating] == [1]
    assert stability.top_studs(slope) == {(0, 0)}
    turned = place("3040.dat", 0, 0, 3, 1, 2, 90)       # low toward -z: high cell z=1
    assert stability.top_studs(turned) == {(0, 1)}


def test_inverted_slope_grips_only_under_its_high_side():
    base = place("3001.dat", 0, 0, 0, 4, 2, 90)
    inv = place("3665.dat", 0, 0, 3)                    # 2x1 inverted, low toward +x
    assert stability.bottom_cells(inv) == {(0, 0)}
    assert not stability.check([base, inv]).floating
    off = place("3665.dat", 1, 0, 3, rot=180)           # high side at x=2, over the base
    assert not stability.check([base, off]).floating
    hanging = place("3665.dat", 3, 0, 3, rot=180)       # high side at x=4, past the base
    assert stability.check([base, hanging]).floating


def test_weak_parts_are_listed():
    base = place("3005.dat", 0, 0, 0)
    plate = place("3710.dat", 0, 0, 3)                  # 1x4 plate on one stud
    r = stability.check([base, plate])
    assert r.weak_connections == 1 and r.weak == [plate]


def test_best_gap_prefers_a_short_peg_on_the_model_over_a_column_to_the_ground():
    occ = np.zeros((10, 6, 1), dtype=bool)
    occ[:9, 0, 0] = True                                # a wall at x=0
    occ[:5, 3, 0] = True                                # a stub at x=3, 4 cells short
    p = place("3010.dat", 0, 0, 9)                      # 1x4 across x=0..3 at y=9, not touching
    x, z, y_lo, y_hi, lands = pipeline.best_gap(occ, p)
    assert (x, z, y_lo, y_hi, lands) == (3, 0, 5, 9, True)
    occ[:, 3, 0] = False                                # no stub: column from the centre cell
    occ[:, 0, 0] = False
    x, z, y_lo, y_hi, lands = pipeline.best_gap(occ, p)
    assert (x, y_lo, y_hi, lands) == (1, 0, 9, False)


def test_brace_grounds_every_floating_part_without_overlaps():
    base = place("3001.dat", 0, 0, 0, 4, 2, 90)
    high = place("3001.dat", 0, 0, 12, 4, 2, 90)        # floating 9 plates up
    out, added = pipeline.brace([base, high])
    r = stability.check(out)
    assert added > 0 and not r.floating and not r.overlaps and r.components == 1


def test_resolve_overlaps_keeps_first_and_refills_the_rest():
    a = place("3001.dat", 0, 0, 0, 4, 2, 90)
    b = place("3010.dat", 2, 0, 0)                      # 1x4 along x=2..5 overlapping a at x=2,3
    out, dropped = pipeline.resolve_overlaps([a, b])
    assert dropped == 1 and not stability.check(out).overlaps
    occ = pipeline.occupancy(out)
    assert occ[0:3, 0:6, 0].all() and occ[0:3, 4:6, 0].all()


def test_assemble_supports_an_overhang_with_a_grounded_column():
    grid = np.full((6, 6, 2), -1, dtype=np.int16)
    grid[0:3, 0:2, :] = 4                               # a post
    grid[3:6, 0:6, :] = 4                               # a beam hanging out 4 studs
    result = pipeline.assemble(voxelize.VoxelGrid(palette_idx=grid, metadata={}))
    assert result.report["geometrically_connected"]
    assert result.report["added_support_voxels"] > 0
    grey = next(i for i, c in enumerate(palette()) if c.code == 71)
    assert any(p.color_idx == grey and p.y_plate >= 2 for p in result.placements)


def test_cantilevered_plate_is_reported_but_not_a_failure():
    base = place("3005.dat", 0, 0, 0)
    hanging = place("3460.dat", 0, 0, 3)                # 1x8 plate held at its end stud
    r = stability.check([base, hanging])
    assert r.cantilever_count == 1 and r.cantilevers == [hanging]
    assert r.ok and r.warnings["cantilevers"] == 1
    centred = place("3460.dat", -3, 0, 3)               # the same plate held at its middle
    assert stability.check([base, centred]).cantilever_count == 0
    two_out = place("3710.dat", -1, 0, 3)               # 1x4 whose centre is 1.5 out
    assert stability.check([base, two_out]).cantilever_count == 0


def test_running_seam_is_reported_with_its_line_and_range():
    courses = [place("3004.dat", x, 0, y) for y in (0, 3, 6) for x in (0, 2)]
    r = stability.check(courses)
    assert r.running_seam_count == 1
    seam = r.running_seams[0]
    assert (seam.axis, seam.x, seam.z, seam.y_lo, seam.y_hi, seam.courses) == ("x", 1, 0, 0, 8, 3)
    assert len(seam.parts) == 6 and seam.line == "x=1|2 z=0..0 y=0..8"
    assert not r.floating and r.components == 2    # two columns, nothing ties them
    bridged = courses[:2] + [place("3010.dat", 0, 0, 3)] + courses[4:]
    r = stability.check(bridged)
    assert r.running_seam_count == 0 and r.ok
    two_courses = stability.check(courses[:4])
    assert two_courses.running_seam_count == 0
    # One plane, several studs long: one seam with a span.
    rows = [place("3006.dat", 0, z, y, 10, 2, 0) for y in (0, 3, 6) for z in (0, 2)]
    seams = stability.check(rows).running_seams
    assert len(seams) == 1 and seams[0].axis == "z" and seams[0].span == 10


def test_group_hanging_by_one_stud_is_reported():
    base = place("3001.dat", 0, 0, 0, 4, 2, 90)
    post = place("3005.dat", 3, 0, 3)                   # 1x1 brick at the base's corner
    shelf = place("3020.dat", 3, 0, 6, 4, 2, 90)        # 2x4 plate on the post's one stud
    load = place("3003.dat", 5, 0, 7)                   # 2x2 brick on the shelf
    r = stability.check([base, post, shelf, load])
    assert r.ok and r.lift_group_count >= 1
    assert any(set(g) == {shelf, load} for g in r.lift_groups)
    assert r.warnings["single_stud_groups"] == r.lift_group_count
    # Two studs under the shelf and nothing hangs.
    post2 = place("3004.dat", 2, 0, 3)
    shelf2 = place("3020.dat", 2, 0, 6, 4, 2, 90)
    load2 = place("3003.dat", 4, 0, 7)
    assert stability.check([base, post2, shelf2, load2]).lift_group_count == 0
    # A 1x1 column is a single stud by nature, not a weak joint.
    column = [base, post, place("3024.dat", 3, 0, 6), place("3070b.dat", 3, 0, 7)]
    assert stability.check(column).lift_group_count == 0
