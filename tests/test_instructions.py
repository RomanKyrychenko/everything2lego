"""Regression tests for the draw order of the isometric instruction renderer."""

import itertools
import random

import numpy as np

from everything2lego import instructions
from everything2lego.bricks import catalog
from everything2lego.pack import Placement


def brick(shape, footprint):
    return next(p for p in catalog() if p.shape == shape and p.footprint == footprint)


def place(part, x, z, y, w=None, d=None):
    w = part.footprint[0] if w is None else w
    d = part.footprint[1] if d is None else d
    return Placement(part, 1, x, z, y, w, d, 0 if (w, d) == part.footprint else 90)


def drawn_before(order, a, b):
    ids = [id(p) for p, _ in order]
    return ids.index(id(a)) < ids.index(id(b))


def test_tall_brick_in_front_is_drawn_after_higher_plate_behind_it():
    tall = place(brick("brick", (1, 1)), 0, 1, 0)
    plate = place(brick("plate", (1, 1)), 0, 0, 1)
    for placements in ([tall, plate], [plate, tall]):
        assert drawn_before(instructions._painter_order(placements), plate, tall)


def test_long_brick_in_front_is_drawn_after_short_brick_behind_it():
    long_ = place(brick("brick", (8, 1)), 0, 1, 0)
    short = place(brick("brick", (1, 1)), 5, 0, 0)
    for placements in ([long_, short], [short, long_]):
        assert drawn_before(instructions._painter_order(placements), short, long_)


def test_part_sitting_on_studs_is_drawn_after_the_part_below():
    below = place(brick("brick", (2, 2)), 0, 0, 0)
    above = place(brick("plate", (2, 2)), 0, 0, 3)
    for placements in ([below, above], [above, below]):
        assert drawn_before(instructions._painter_order(placements), below, above)


def test_order_respects_every_overlapping_pair_in_a_random_build():
    random.seed(7)
    parts = [brick("brick", (1, 1)), brick("brick", (2, 1)), brick("brick", (4, 1)),
             brick("plate", (2, 2)), brick("plate", (1, 1))]
    size, height = 10, 24
    filled = np.zeros((size, height, size), dtype=bool)
    placements = []
    for _ in range(2000):
        part = random.choice(parts)
        w, d = part.footprint if random.random() < 0.5 else part.footprint[::-1]
        x, z = random.randrange(size - w + 1), random.randrange(size - d + 1)
        y = random.randrange(height - part.height_units + 1)
        region = filled[x:x + w, y:y + part.height_units, z:z + d]
        if region.any():
            continue
        region[:] = True
        placements.append(place(part, x, z, y, w, d))
    assert len(placements) > 150

    order = instructions._painter_order(placements)
    assert {id(p) for p, _ in order} == {id(p) for p in placements}
    assert any(cell is not None for _, cell in order), "expected a cyclic overlap to split"

    body, studs, has_studs = instructions._boxes(order)
    i, j = (np.array(v) for v in zip(*itertools.combinations(range(len(order)), 2)))
    earlier_first, later_first = instructions._pair_order(body, studs, has_studs, i, j)
    assert earlier_first.sum() > len(order)
    assert not later_first.any(), [(order[a], order[b]) for a, b in
                                    zip(i[later_first], j[later_first])][:3]


def test_cyclic_overlap_is_split_into_cells():
    # A plate on a 1x4 brick, and a brick behind the 1x4 but in front of
    # the plate: each hides part of the next, so no order of whole parts
    # is right.  The cycle is split into cells and drawn exactly.
    plate = place(brick("plate", (2, 2)), 0, 0, 3)
    long_ = place(brick("brick", (4, 1)), 1, 1, 0, 1, 4)
    side = place(brick("brick", (1, 1)), 0, 2, 2)
    order = instructions._painter_order([long_, plate, side])
    assert {id(p) for p, _ in order} == {id(long_), id(plate), id(side)}
    assert any(cell is not None for _, cell in order)
    body, studs, has_studs = instructions._boxes(order)
    i, j = (np.array(v) for v in zip(*itertools.combinations(range(len(order)), 2)))
    assert not instructions._pair_order(body, studs, has_studs, i, j)[1].any()


def test_render_step_draws_shadow_and_parts():
    placements = [place(brick("brick", (2, 2)), 0, 0, 0)]
    img = instructions._render_step(placements, set(), (400, 300),
                                    instructions._bounds(placements))
    background = img.getpixel((5, 5))
    pixels = set(map(tuple, np.asarray(img.convert("RGB")).reshape(-1, 3).tolist()))
    assert background == (240, 240, 240)
    assert any(p != background and max(p) - min(p) < 6 for p in pixels), "no grey shadow"
