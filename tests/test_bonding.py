"""Bonding: running seams in stacked courses are broken by re-merging the
parts on either side, without changing what is covered or in what colour."""

import numpy as np

from everything2lego import bonding, pipeline, stability
from everything2lego.bricks import catalog
from everything2lego.pack import Placement


def part(ldraw):
    return next(p for p in catalog() if p.ldraw == ldraw)


def place(ldraw, x, z, y, w=None, d=None, rot=0, color=4):
    p = part(ldraw)
    w = p.footprint[0] if w is None else w
    d = p.footprint[1] if d is None else d
    return Placement(p, color, x, z, y, w, d, rot)


def wall(ldraw, length, courses, *, color=4, step=None):
    """Identical courses of `ldraw` bricks laid end to end along x."""
    p = part(ldraw)
    step = p.footprint[0] if step is None else step
    return [place(ldraw, x, 0, y, step, 1, 0, color)
            for y in range(0, 3 * courses, 3) for x in range(0, length, step)]


def colour_cells(placements):
    """{(y, x, z): colour} for every cell; asserts no cell is claimed twice."""
    cells = {}
    for p in placements:
        for y in range(p.y_plate, p.y_plate + p.part.height_units):
            for x in range(p.x_stud, p.x_stud + p.w_world):
                for z in range(p.z_stud, p.z_stud + p.d_world):
                    assert (y, x, z) not in cells
                    cells[(y, x, z)] = p.color_idx
    return cells


def test_stacked_1x2_courses_are_merged_into_longer_bricks():
    before = wall("3004.dat", 8, 5)                     # 4 x 1x2 per course, 5 courses
    assert stability.check(before).running_seam_count == 3
    after, stats = bonding.bond(before)
    assert stats["merged"] > 0 and stats["running_seams_after"] == 0
    assert stability.check(after).running_seam_count == 0
    assert colour_cells(after) == colour_cells(before)
    assert len(after) < len(before)
    assert any(p.part.ldraw == "3010.dat" for p in after)   # Brick 1x4 appears
    assert stability.check(after).ok


def test_split_moves_when_the_combined_length_has_no_part():
    before = [place("3006.dat", x, 0, y, 10, 2, 0)      # 2x10 + 2x10: no 2x20 brick
              for y in range(0, 15, 3) for x in (0, 10)]
    assert stability.check(before).running_seam_count == 1
    after, stats = bonding.bond(before)
    assert stats["shifted"] > 0 and stats["merged"] == 0
    assert stability.check(after).running_seam_count == 0
    assert colour_cells(after) == colour_cells(before)
    rewritten = sorted(p.x_stud + p.w_world for p in after if p.y_plate == 3)
    assert 10 not in rewritten[:-1]                     # the old split at x=10 is bridged


def test_different_colours_are_never_merged():
    before = [place("3004.dat", x, 0, y, 2, 1, 0, color=4 + (x // 2) % 2)
              for y in range(0, 15, 3) for x in range(0, 8, 2)]
    after, stats = bonding.bond(before)
    assert after == before
    assert stats["merged"] == stats["shifted"] == stats["rebuilt"] == 0


def test_tall_pairs_are_rebuilt_as_staggered_courses():
    a = place("30144.dat", 0, 0, 0)                     # Panel 1x4x6, 18 plates
    b = place("30144.dat", 4, 0, 0)
    before = [a, b]
    assert stability.check(before).running_seam_count == 1
    after, stats = bonding.bond(before)
    assert stats["rebuilt"] == 1 and stability.check(after).running_seam_count == 0
    assert colour_cells(after) == colour_cells(before)
    assert all(p.part.shape == "brick" for p in after)
    assert len(after) == 6                              # six 1x8 bricks


def test_assemble_bonds_a_solid_block():
    grid = np.full((27, 12, 8), 4, dtype=np.int16)
    result = pipeline.assemble(pipeline.VoxelGrid(palette_idx=grid, metadata={}))
    bond = result.report["bonding"]
    assert bond["running_seams_after"] < bond["running_seams_before"]
    assert result.report["running_seams"] == bond["running_seams_after"]
    assert result.report["geometrically_connected"]
    assert not stability.check(result.placements).overlaps
    assert int(pipeline.occupancy(result.placements).sum()) == int((result.grid.palette_idx >= 0).sum())
