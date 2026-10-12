"""Post-pack colour swaps and the report that describes their result."""

import numpy as np

from everything2lego import pack, pipeline, stability
from everything2lego.bricks import catalog, full_catalog
from everything2lego.colors import palette
from everything2lego.pack import Placement


def code_idx(code):
    return next(i for i, c in enumerate(palette()) if c.code == code)


def part(ldraw):
    return next(p for p in full_catalog() if p.ldraw == ldraw)


def brick(ldraw, x, z, y, color_code, w=None, d=None):
    p = part(ldraw)
    w = p.footprint[0] if w is None else w
    d = p.footprint[1] if d is None else d
    return Placement(p, code_idx(color_code), x, z, y, w, d, 0 if (w, d) == p.footprint else 90)


def test_window_replaces_two_stacked_bricks_of_its_colour():
    # Two 1x2 bricks (6 plates) in colour 71 under a plain brick: the stack
    # becomes one 1x2x2 window frame; the brick above is untouched.
    stack = [brick("3004.dat", 0, 0, 0, 71), brick("3004.dat", 0, 0, 3, 71),
             brick("3004.dat", 0, 0, 6, 4), brick("3004.dat", 2, 0, 0, 71)]
    out = pack.substitute_by_color(stack, {71: "60592.dat"})
    windows = [p for p in out if p.part.ldraw == "60592.dat"]
    assert len(windows) == 1 and windows[0].y_plate == 0
    assert len(out) == 3                      # lone 1x2 at x=2 has nothing above: unchanged
    assert not stability.check(out).overlaps


def test_door_replaces_five_stacked_1x4_bricks():
    stack = [brick("3010.dat", 0, 0, 3 * i, 70) for i in range(5)]
    out = pack.substitute_by_color(stack, {70: "60623.dat"})
    assert [p.part.ldraw for p in out] == ["60623.dat"]
    assert out[0].part.height_units == 15 and not stability.check(out).overlaps


def test_mixed_colour_or_short_stacks_are_left_alone():
    stack = [brick("3004.dat", 0, 0, 0, 71), brick("3004.dat", 0, 0, 3, 4)]
    out = pack.substitute_by_color(stack, {71: "60592.dat"})
    assert [p.part.ldraw for p in out] == ["3004.dat", "3004.dat"]
    rotated = [brick("3004.dat", 0, 0, 0, 71, 1, 2), brick("3004.dat", 0, 0, 3, 71, 1, 2)]
    out = pack.substitute_by_color(rotated, {71: "60592.dat"})
    assert [p.part.ldraw for p in out] == ["60592.dat"] and out[0].rotation_deg == 90


def test_structure_report_reflects_current_placements():
    a = brick("3004.dat", 0, 0, 0, 4)
    floating = brick("3004.dat", 0, 0, 9, 4)
    assert pipeline.structure_report([a])["geometrically_connected"]
    after = pipeline.structure_report([a, floating])
    assert after["parts"] == 2 and after["unsupported_parts"] == 1
    assert not after["geometrically_connected"]
