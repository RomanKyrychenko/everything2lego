"""Colours and parts a buyer can get: the palette holds only colours LEGO
makes the basic bricks and plates in, and recipes build each run from
parts made in its colour (data/part_colours.json)."""

import numpy as np

from everything2lego.bricks import made_in
from everything2lego.colors import by_code_index, dominant_colour, palette
from everything2lego.semantic_v2.openings import WallFrame
from everything2lego.semantic_v2.setdesign import (COURSE, PitchSpec, _settle_colours, build_gable,
                                                   build_wall, pair_pitches, SetPlan, WallSpec)


def test_palette_has_current_colours_only():
    codes = {c.code for c in palette()}
    assert {28, 70, 72, 379, 15, 0} <= codes          # Dark Tan, Reddish Brown, DBG, Sand Blue...
    assert not codes & {6, 7, 86, 151, 216}            # old Brown, old Light Gray, Light Brown, ...


def test_made_in_knows_what_lego_makes():
    assert made_in("3004.dat", 28)                     # brick 1x2 in Dark Tan: yes
    assert not made_in("3622.dat", 28)                 # brick 1x3 in Dark Tan: no
    assert not made_in("3037.dat", 379)                # slope 45 2x4 in Sand Blue: no
    assert made_in("3037.dat", 72)


def test_a_dark_tan_wall_is_built_without_dark_tan_1x3_bricks():
    dt = by_code_index(28)
    spec = WallSpec(frame=WallFrame(True, 0, 0, 0, 23, 4 * COURSE), colour=dt)
    ps = build_wall(spec)
    covered = {(x, p.y_plate) for p in ps for x in range(p.x_stud, p.x_stud + p.w_world)}
    assert len(covered) == 23 * 4                      # the whole wall is still built
    assert all(made_in(p.part, 28) for p in ps)
    assert not any(p.part.ldraw == "3622.dat" for p in ps)


def test_a_roof_colour_its_slopes_dont_come_in_moves_to_one_they_do():
    sb = by_code_index(379)                            # Sand Blue: 45° slopes only as 2x1
    a = PitchSpec(region=None, axis="x", sign=+1, eave=0.0, ridge=4.0, span=(0, 8),
                  y_bottom=0, y_top=2 * COURSE, colour=sb)
    b = PitchSpec(region=None, axis="x", sign=-1, eave=8.0, ridge=4.0, span=(0, 8),
                  y_bottom=0, y_top=2 * COURSE, colour=sb)
    plan = SetPlan(walls=[], gables=pair_pitches([a, b], wall_colour=by_code_index(28)), consumed=set())
    g = _settle_colours(plan).gables[0]
    code = palette()[g.colour].code
    assert code != 379 and made_in("3037.dat", code) and made_in("3039.dat", code)
    ps = build_gable(g, claimed=np.zeros((20, 12, 12), bool), grid_shape=(20, 12, 12))
    assert all(made_in(p.part, palette()[p.color_idx].code) for p in ps)
    slopes = [p for p in ps if p.part.shape == "slope"]
    assert slopes and max(p.d_world for p in slopes) > 1     # runs of wide parts, not only 1-wide


def test_a_slate_roof_that_cant_be_sand_blue_reads_as_dark_bluish_grey():
    """Photographed blue-grey slate snaps to Sand Blue; a roof can't be
    built in it, and the substitute keeps the slate's tone."""
    rng = np.random.default_rng(0)
    slate = np.clip(np.array([96, 102, 128]) + rng.normal(0, 8, (300, 3)), 0, 255).astype(np.uint8)
    assert palette()[dominant_colour(slate)].code == 379
    roof = dominant_colour(slate, ok=lambda code: made_in("3037.dat", code) and made_in("4286.dat", code))
    assert palette()[roof].code == 72
