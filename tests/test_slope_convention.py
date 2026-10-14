"""One slope orientation convention, shared by every writer and reader.

Part 3040 (footprint (2, 1), `slope_dir` "+x") at rotation 0 is taken to
occupy 2 studs along x with its low side toward +x; a quarter turn cycles
the low side +x -> -z -> -x -> +z.  The packer, the LDraw writer, the
instruction renderer, the viewer export and the soft trainer all use
`pack.slope_world_low` / `pack._slope_rotation`, so a change of convention
is made in those two functions and nowhere else.  This test pins the
convention so no writer can drift from it unnoticed.  The LDraw writer
maps it onto each part's own LDraw frame (`Part.yaw`, `Part.origin`,
measured from the part files: data/part_frames.json);
tests/test_ldraw_origin.py checks that against the parts' geometry.
"""

import numpy as np

from everything2lego import io_ldraw, pack, viewer_export
from everything2lego.bricks import catalog
from everything2lego.pack import Placement


def slope():
    return next(p for p in catalog() if p.ldraw == "3040.dat")


def test_rotation_and_low_side_round_trip():
    s = slope()
    for low, rot, (w, d) in (("+x", 0, (2, 1)), ("-z", 90, (1, 2)),
                             ("-x", 180, (2, 1)), ("+z", 270, (1, 2))):
        assert pack._slope_rotation(s, (w, d), low) == rot
        p = Placement(s, 0, 0, 0, 0, w, d, rot)
        assert pack.slope_world_low(p) == low
        assert viewer_export._slope_world_low(p) == low


def test_ldraw_line_rotates_the_part_the_same_way():
    s = slope()
    lines = {rot: io_ldraw.placement_to_line(Placement(s, 0, 0, 0, 0, 2 if rot % 180 == 0 else 1,
                                                          1 if rot % 180 == 0 else 2, rot))
             for rot in (0, 90, 180, 270)}
    matrices = {rot: np.array(l.split()[5:14], dtype=float).reshape(3, 3) for rot, l in lines.items()}
    # rotation 0 is the part's own LDraw frame turned to the convention; the
    # placement's rotation turns it further, the same way for every part
    turn = {rot: m @ matrices[0].T for rot, m in matrices.items()}
    assert np.allclose(matrices[0], np.array(io_ldraw._ROT[s.yaw]).reshape(3, 3))
    assert np.allclose(turn[90] @ turn[90], turn[180])
    assert np.allclose(turn[90] @ turn[270], np.eye(3))
    assert all(l.endswith("3040.dat") for l in lines.values())
