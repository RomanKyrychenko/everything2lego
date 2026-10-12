"""The soft trainer's discretiser must emit real parts that fill the argmax
grid exactly once, with sloped parts as tall as they really are and pointing
the way the packer would point the same part."""

import numpy as np

from everything2lego import instructions
from everything2lego.semantic_v2 import soft_trainer
from everything2lego.semantic_v2.soft_assembly import (
    EMPTY, PLATE, SLOPE_XP, SLOPE_ZP, CHEESE_XM)


def discretise(types, colors=None):
    colors = np.zeros_like(types) if colors is None else colors
    return soft_trainer.SoftTrainer._discretise_consolidated(None, None, types, colors)


def occupancy(placements, shape):
    occ = np.zeros(shape, dtype=int)
    for p in placements:
        occ[p.y_plate:p.y_plate + p.part.height_units,
            p.x_stud:p.x_stud + p.w_world, p.z_stud:p.z_stud + p.d_world] += 1
    return occ


def test_slope_pair_becomes_one_three_plate_slope_pointing_the_packer_way():
    types = np.full((3, 2, 1), SLOPE_XP)
    out = discretise(types)
    assert [p.part.ldraw for p in out] == ["3040.dat"]
    p = out[0]
    assert (p.w_world, p.d_world, p.y_plate) == (2, 1, 0)
    assert instructions._slope_world_low(p) == "+x"
    assert (occupancy(out, types.shape) == 1).all()

    types = np.full((3, 1, 2), SLOPE_ZP)
    out = discretise(types)
    assert [p.part.ldraw for p in out] == ["3040.dat"]
    assert (out[0].w_world, out[0].d_world) == (1, 2)
    assert instructions._slope_world_low(out[0]) == "+z"


def test_short_slope_columns_degrade_to_parts_of_their_own_height():
    lone = np.full((3, 1, 1), SLOPE_XP)
    assert [p.part.ldraw for p in discretise(lone)] == ["3005.dat"]
    two = np.full((2, 1, 1), SLOPE_XP)
    (p,) = discretise(two)
    assert p.part.ldraw == "54200.dat" and p.part.height_units == 2
    one = np.full((1, 1, 1), SLOPE_XP)
    assert [p.part.ldraw for p in discretise(one)] == ["3024.dat"]
    cheese = np.full((2, 1, 1), CHEESE_XM)
    (p,) = discretise(cheese)
    assert p.part.ldraw == "54200.dat" and instructions._slope_world_low(p) == "-x"


def test_roof_over_walls_is_filled_exactly_once():
    rng = np.random.default_rng(3)
    types = np.full((9, 6, 5), EMPTY)
    types[:6] = PLATE
    types[6:9, :3] = SLOPE_XP           # a pitched roof half, 3 plates tall
    types[6:8, 3:5] = CHEESE_XM          # a lower cheese course
    types[6, 5] = SLOPE_XP               # a slope cell too short for a slope
    colors = rng.integers(0, 3, size=types.shape)
    colors[6:] = 1
    out = discretise(types, colors)
    occ = occupancy(out, types.shape)
    assert (occ[types != EMPTY] == 1).all(), "every cell exactly once"
    assert (occ[types == EMPTY] == 0).all()
    assert any(p.part.ldraw == "3040.dat" for p in out)
    assert any(p.part.ldraw == "54200.dat" for p in out)
