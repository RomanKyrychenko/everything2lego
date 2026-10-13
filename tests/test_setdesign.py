"""setdesign recipes: bonded walls around part openings, slope roofs with
interior, gable ends and ridge."""

import numpy as np

from everything2lego.semantic_v2.openings import WallFrame
from everything2lego.semantic_v2.setdesign import (COURSE, GableSpec, PitchSpec, WallSpec,
                                                   bond_course, build_gable, build_wall,
                                                   pair_pitches)
from everything2lego.stability import check
from everything2lego.templates.base import part_by_ldraw


def _seams(course):
    return sorted(u + w for u, w in course[:-1])


def test_bond_course_offsets_seams_between_courses():
    free = np.ones(16, bool)
    even, odd = bond_course(16, 0, free), bond_course(16, 1, free)
    assert sum(w for _, w in even) == 16 and sum(w for _, w in odd) == 16
    assert not set(_seams(even)) & set(_seams(odd))          # running bond


def test_bond_course_skips_reserved_cells():
    free = np.ones(12, bool)
    free[4:8] = False                                        # a window
    course = bond_course(12, 0, free)
    covered = sorted(u + k for u, w in course for k in range(w))
    assert covered == [0, 1, 2, 3, 8, 9, 10, 11]


def test_build_wall_is_one_bonded_assembly_around_a_window():
    frame = WallFrame(axis_x=True, origin=0, other=0, y0=0, length=12, height=3 * COURSE)
    window = part_by_ldraw("60592.dat")                       # 1x2x2 (6 plates)
    win = frame.placement(window, 5, u=4, y=COURSE, width=2)
    reserved = np.zeros((12, 3 * COURSE), bool)
    reserved[4:6, COURSE:COURSE + 6] = True
    spec = WallSpec(frame=frame, colour=4, openings=[win], reserved=reserved)
    ps = build_wall(spec)
    rep = check(ps)
    assert not rep.overlaps
    assert rep.components == 1
    # every non-window cell of the 12x9 face is covered exactly once
    cells = np.zeros((12, 3 * COURSE), int)
    for p in ps:
        for u in range(p.x_stud, p.x_stud + p.w_world):
            for y in range(p.y_plate, p.y_plate + p.part.height_units):
                cells[u, y] += 1
    assert (cells == 1).all()


def test_wall_respects_claimed_cells():
    frame = WallFrame(axis_x=False, origin=0, other=3, y0=0, length=8, height=COURSE)
    claimed = np.zeros((6, 6, 10), bool)
    claimed[0:COURSE, 3, 0] = True                           # corner stud owned by another wall
    ps = build_wall(WallSpec(frame=frame, colour=4), claimed)
    assert all(not (p.x_stud == 3 and p.z_stud == 0) for p in ps)
    assert sum(p.d_world for p in ps) == 7


def test_interlock_corners_bonds_two_walls_through_the_corner_column():
    from everything2lego.semantic_v2.setdesign import interlock_corners
    n = 4
    a = WallSpec(frame=WallFrame(axis_x=True, origin=0, other=0, y0=0, length=8, height=n * COURSE),
                 colour=4, mask=np.ones((8, n), bool))
    b = WallSpec(frame=WallFrame(axis_x=False, origin=0, other=0, y0=0, length=6, height=n * COURSE),
                 colour=4, mask=np.ones((6, n), bool))
    claimed = np.zeros((n * COURSE, 10, 10), bool)
    ps = build_wall(a, claimed)
    for p in ps:
        claimed[p.y_plate:p.y_plate + 3, p.x_stud:p.x_stud + p.w_world, p.z_stud:p.z_stud + p.d_world] = True
    ps += build_wall(b, claimed)
    assert check(ps).components == 2                              # butt joint: no shared stud

    a.mask[:] = True
    b.mask[:] = True
    assert interlock_corners([a, b]) == 1
    claimed[:] = False
    ps = build_wall(a, claimed)
    for p in ps:
        claimed[p.y_plate:p.y_plate + 3, p.x_stud:p.x_stud + p.w_world, p.z_stud:p.z_stud + p.d_world] = True
    ps += build_wall(b, claimed)
    rep = check(ps)
    assert rep.components == 1 and not rep.overlaps
    corner = sorted(p.rotation_deg for p in ps if p.x_stud == 0 and p.z_stud == 0)
    assert corner == [0, 0, 90, 90]                                # the column alternates owners


def _pitch(sign, eave, ridge, span=(0, 8), y_bottom=0, y_top=2 * COURSE):
    return PitchSpec(region=None, axis="x", sign=sign, eave=eave, ridge=ridge, span=span,
                     y_bottom=y_bottom, y_top=y_top, colour=7)


def test_pair_pitches_makes_a_gable_and_a_lean_to():
    a, b = _pitch(+1, eave=0.0, ridge=4.0), _pitch(-1, eave=8.0, ridge=4.0)
    lone = _pitch(+1, eave=20.0, ridge=24.0, span=(20, 28))
    gables = pair_pitches([a, lone, b], wall_colour=28)
    kinds = sorted("gable" if g.b is not None else "lean-to" for g in gables)
    assert kinds == ["gable", "lean-to"]
    g = next(g for g in gables if g.b is not None)
    assert g.n_rows == 2 and g.gable_colour == 28


def test_merge_pitches_joins_slices_of_one_face():
    from everything2lego.semantic_v2.setdesign import merge_pitches
    slices = [_pitch(+1, eave=0.0, ridge=4.0, span=(0, 5)),
              _pitch(+1, eave=0.4, ridge=4.3, span=(5, 11)),       # touches, same plane
              _pitch(+1, eave=0.0, ridge=4.0, span=(17, 23)),      # gap of 6: another face
              _pitch(-1, eave=8.0, ridge=4.0, span=(0, 11))]       # the other side
    merged = merge_pitches(slices)
    spans = sorted((p.sign, p.span) for p in merged)
    assert spans == [(-1, (0, 11)), (1, (0, 11)), (1, (17, 23))]
    assert len(pair_pitches(merged)) == 2                       # one gable + one lean-to


def test_build_gable_has_slopes_both_faces_interior_and_ridge():
    # run 6 per face over 3 bricks → 2-stud step → 33° rows (3 deep); the
    # faces meet after two rows and the ridge closes over their top studs.
    a = _pitch(+1, eave=0.0, ridge=6.0, y_top=3 * COURSE)
    b = _pitch(-1, eave=12.0, ridge=6.0, y_top=3 * COURSE)
    g = pair_pitches([a, b], wall_colour=28)[0]
    ps = build_gable(g, claimed=np.zeros((20, 16, 12), bool), grid_shape=(20, 16, 12))
    rep = check(ps)
    assert not rep.overlaps
    slopes = [p for p in ps if p.part.shape == "slope"]
    assert sum(p.w_world * p.d_world for p in slopes) == 2 * 2 * 3 * 8   # 2 faces × 2 rows × 3×8
    assert {p.rotation_deg for p in slopes} == {0, 180}      # low sides face outward on both faces
    assert all(p.part.name.startswith("Slope 33") for p in slopes)   # 2-stud step → 33° family
    assert any(p.part.shape == "brick" and p.color_idx == 28 for p in ps)   # gable-end bricks
    ridge = [p for p in ps if p.part.shape == "tile" and p.y_plate == g.y_bottom + 2 * COURSE]
    # top studs at x=4 (face a) and x=8 (face b), tiles from one to the other
    assert ridge and {x for p in ridge for x in range(p.x_stud, p.x_stud + p.w_world)} == {4, 5, 6, 7, 8}
    # nothing sits where the faces would have collided
    assert not any(p.part.shape == "slope" and p.y_plate >= g.y_bottom + 2 * COURSE for p in ps)


def test_hipped_gable_slopes_on_all_four_sides_and_steps_in():
    a = _pitch(+1, eave=0.0, ridge=6.0, span=(0, 16), y_top=3 * COURSE)
    b = _pitch(-1, eave=12.0, ridge=6.0, span=(0, 16), y_top=3 * COURSE)
    g = pair_pitches([a, b], wall_colour=28)[0]
    g.hip_lo = g.hip_hi = True
    ps = build_gable(g, claimed=np.zeros((20, 16, 20), bool), grid_shape=(20, 16, 20))
    rep = check(ps)
    assert not rep.overlaps
    slopes = [p for p in ps if p.part.shape == "slope"]
    rots = {p.rotation_deg for p in slopes}
    assert rots == {0, 90, 180, 270}                         # faces out on all four sides
    # the hipped ends step in: row 1's long-face slopes start inside row 0's
    z_row0 = min(p.z_stud for p in slopes if p.y_plate == 0 and p.rotation_deg in (0, 180))
    z_row1 = min(p.z_stud for p in slopes if p.y_plate == COURSE and p.rotation_deg in (0, 180))
    assert z_row1 > z_row0
    assert not any(p.part.shape == "brick" and p.color_idx == 28 and p.z_stud in (0, 15) for p in ps)


def test_absorb_hips_turns_crossing_roofs_into_one_hipped_roof():
    from everything2lego.semantic_v2.setdesign import absorb_hips
    gx = pair_pitches([_pitch(+1, eave=0.0, ridge=6.0, span=(0, 20), y_top=3 * COURSE),
                       _pitch(-1, eave=12.0, ridge=6.0, span=(0, 20), y_top=3 * COURSE)])[0]
    za = PitchSpec(region=None, axis="z", sign=+1, eave=0.0, ridge=6.0, span=(0, 12),
                   y_bottom=0, y_top=3 * COURSE, colour=7)
    zb = PitchSpec(region=None, axis="z", sign=-1, eave=20.0, ridge=14.0, span=(0, 12),
                   y_bottom=0, y_top=3 * COURSE, colour=7)
    ends = pair_pitches([za, zb])                         # two lean-tos: hip ends never pair
    assert len(ends) == 2
    out = absorb_hips([gx] + ends)
    assert len(out) == 1 and out[0] is gx and gx.hip_lo and gx.hip_hi


def test_plan_owned_covers_roof_prism_to_grid_top():
    from everything2lego.semantic_v2.setdesign import SetPlan
    a = _pitch(+1, eave=0.0, ridge=6.0, y_top=3 * COURSE)
    b = _pitch(-1, eave=12.0, ridge=6.0, y_top=3 * COURSE)
    plan = SetPlan(walls=[], gables=pair_pitches([a, b]), consumed=set())
    m = plan.owned((20, 16, 12))
    assert m[0:, 0:13, 0:8].all() and not m[:, 13:, :].any() and not m[:, :, 8:].any()
