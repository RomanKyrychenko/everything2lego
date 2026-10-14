"""Set mode end to end on a synthetic gable house: base, bonded walls, a
slope roof that seats on the eave walls and closes at a ridge, gable ends
in brick — one supported assembly with nothing poking through the roof."""

import numpy as np
import pytest
import trimesh

from everything2lego.semantic_v2.pipeline import PipelineConfig, SemanticLegoPipeline
from everything2lego.semantic_v2.setdesign import COURSE
from everything2lego.stability import check

DARK = np.array([108, 110, 104, 255], np.uint8)     # LDraw 72
WHITE = np.array([255, 255, 255, 255], np.uint8)    # LDraw 15
RED = np.array([201, 26, 9, 255], np.uint8)         # LDraw 4


def _box(cx, cy, cz, w, h, d):
    v = np.array([[cx - w/2, cy - h/2, cz - d/2], [cx + w/2, cy - h/2, cz - d/2],
                  [cx + w/2, cy + h/2, cz - d/2], [cx - w/2, cy + h/2, cz - d/2],
                  [cx - w/2, cy - h/2, cz + d/2], [cx + w/2, cy - h/2, cz + d/2],
                  [cx + w/2, cy + h/2, cz + d/2], [cx - w/2, cy + h/2, cz + d/2]])
    f = np.array([[0, 3, 2], [0, 2, 1], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                  [3, 7, 6], [3, 6, 2], [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5]])
    return v, f


def gable_house(path, W=200.0, D=120.0, H=90.0, RISE=70.0, slab=10.0):
    """Dark slab, white box, red gable roof whose ridge runs along z."""
    V, F, C = [], [], []

    def add(v, f, colour):
        off = sum(len(x) for x in V)
        V.append(v); F.append(f + off); C.append(np.tile(colour, (len(f), 1)))

    add(*_box(0, slab / 2, 0, W + 20, slab, D + 20), DARK)
    add(*_box(0, slab + H / 2, 0, W, H, D), WHITE)
    base_y = slab + H - 4.0
    rv = np.array([[-W/2, base_y, -D/2], [W/2, base_y, -D/2], [0, base_y + RISE, -D/2],
                   [-W/2, base_y, D/2], [W/2, base_y, D/2], [0, base_y + RISE, D/2]])
    rf = np.array([[0, 2, 1], [3, 4, 5], [1, 2, 5], [1, 5, 4], [0, 3, 5], [0, 5, 2]])
    add(rv, rf, RED)
    trimesh.Trimesh(np.vstack(V), np.vstack(F), face_colors=np.vstack(C), process=False).export(path)
    return path


def hip_house(path, W=200.0, D=120.0, H=90.0, RISE=60.0, slab=10.0):
    """Like gable_house, but the roof is hipped at both ends."""
    V, F, C = [], [], []

    def add(v, f, colour):
        off = sum(len(x) for x in V)
        V.append(v); F.append(f + off); C.append(np.tile(colour, (len(f), 1)))

    add(*_box(0, slab / 2, 0, W + 20, slab, D + 20), DARK)
    add(*_box(0, slab + H / 2, 0, W, H, D), WHITE)
    base_y = slab + H - 4.0
    top = base_y + RISE
    half = D / 2
    rv = np.array([[-W/2, base_y, -D/2], [W/2, base_y, -D/2], [W/2, base_y, D/2], [-W/2, base_y, D/2],
                   [-W/2 + half, top, 0], [W/2 - half, top, 0]])
    rf = np.array([[0, 1, 2], [0, 2, 3], [0, 4, 1], [1, 4, 5], [3, 2, 5], [3, 5, 4], [0, 3, 4], [1, 5, 2]])
    add(rv, rf, RED)
    trimesh.Trimesh(np.vstack(V), np.vstack(F), face_colors=np.vstack(C), process=False).export(path)
    return path


def test_set_mode_hip_roof_slopes_on_all_four_sides(tmp_path):
    path = hip_house(str(tmp_path / "hip.glb"))
    res = SemanticLegoPipeline(PipelineConfig(max_studs=24, set_mode=True)).run(path)
    rep = check(res.placements)
    assert rep.components == 1 and not rep.floating and not rep.overlaps
    slopes = [p for p in res.placements if p.part.shape == "slope"]
    assert {p.rotation_deg for p in slopes} == {0, 90, 180, 270}
    assert res.per_region_counts["set:gables"] == 1 and res.per_region_counts["set:lean_tos"] == 0


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    path = gable_house(str(tmp_path_factory.mktemp("set") / "gable.glb"))
    return SemanticLegoPipeline(PipelineConfig(max_studs=24, set_mode=True)).run(path)


def test_set_mode_is_one_supported_assembly(result):
    rep = check(result.placements)
    assert rep.components == 1 and not rep.floating and not rep.overlaps
    c = result.per_region_counts
    assert c["set:gables"] == 1 and c["set:walls"] >= 4
    assert c["set:floors"] >= 2                                   # the base plate and the ceiling


def test_set_mode_roof_is_slope_rows_on_both_faces_with_a_ridge(result):
    slopes = [p for p in result.placements if p.part.shape == "slope"]
    assert len(slopes) >= 10
    assert {p.rotation_deg for p in slopes} == {0, 180}          # faces toward -x and +x
    assert all(p.color_idx == slopes[0].color_idx for p in slopes)  # one roof colour
    top = max(p.y_plate for p in slopes)
    rows = sorted({p.y_plate for p in slopes})
    assert all(b - a == COURSE for a, b in zip(rows, rows[1:]))   # one row per brick course
    # nothing but the ridge tiles above the last slope row, and nothing
    # studded on top: the mesh ridge was higher than the rows reach, and
    # that gap belongs to the roof, not to a plate deck.
    above = [p for p in result.placements if p.y_plate >= top + COURSE]
    assert above and all(p.part.shape == "tile" for p in above)


def test_set_mode_walls_are_bonded_brick_courses(result):
    walls = [p for p in result.surface_placements
             if p.part.shape == "brick" and p.color_idx == slopes_colour_other(result)]
    assert len(walls) >= 40
    assert all(p.y_plate % COURSE == 0 for p in walls)           # on course lines
    # mostly long bricks: 1x1s only close odd runs and the gable steps
    one_by_ones = sum(1 for p in walls if p.w_world * p.d_world == 1)
    assert one_by_ones <= 0.3 * len(walls)
    assert sum(1 for p in walls if max(p.w_world, p.d_world) >= 8) >= 20


def slopes_colour_other(result):
    """The wall colour: the most common brick colour that is not the roof's."""
    from collections import Counter
    roof = next(p.color_idx for p in result.placements if p.part.shape == "slope")
    counts = Counter(p.color_idx for p in result.surface_placements
                     if p.part.shape == "brick" and p.color_idx != roof)
    return counts.most_common(1)[0][0]


def test_set_mode_uses_only_parts_lego_makes_in_their_colours(result):
    """A store set can be bought: every part comes in the colour it has."""
    from everything2lego.bricks import made_in
    from everything2lego.colors import palette
    pal = palette()
    bad = {(p.part.ldraw, pal[p.color_idx].name) for p in result.placements
           if not made_in(p.part, pal[p.color_idx].code)}
    assert not bad, bad


def test_a_house_and_a_shed_beside_it_stand_on_one_base_plate(tmp_path):
    """Two blocks the ground alone joins come out as one assembly: the
    base plate under both is built first and everything stands on it."""
    V, F, C = [], [], []

    def add(v, f, colour):
        off = sum(len(x) for x in V)
        V.append(v); F.append(f + off); C.append(np.tile(colour, (len(f), 1)))

    add(*_box(0, 45, 0, 200, 90, 120), WHITE)                # the house, on the ground
    add(*_box(230, 20, 0, 40, 40, 40), WHITE)                # a shed 110 mm away
    path = str(tmp_path / "two.glb")
    trimesh.Trimesh(np.vstack(V), np.vstack(F), face_colors=np.vstack(C), process=False).export(path)
    res = SemanticLegoPipeline(PipelineConfig(max_studs=24, set_mode=True)).run(path)
    rep = check(res.placements)
    assert rep.components == 1 and not rep.floating
    c = res.per_region_counts
    # the shed is kept, not dropped as an island or tied on with bridges
    assert c.get("structure:dropped_islands", 0) == 0 and c.get("structure:bridges", 0) == 0
    assert max(p.x_stud + p.w_world for p in res.placements) > 20


def test_no_storey_floor_runs_through_a_roof(result):
    """The gable's stepped ends are walls starting inside the roof; they
    must not earn a floor sheet through the roof's rows."""
    plates = [p for p in result.placements if p.part.shape == "plate"]
    slopes = [p for p in result.placements if p.part.shape == "slope"]
    eave = min(p.y_plate for p in slopes)
    assert not any(eave <= p.y_plate < eave + COURSE and
                   any(s.x_stud <= p.x_stud < s.x_stud + s.w_world and s.z_stud <= p.z_stud < s.z_stud + s.d_world
                       for s in slopes if s.y_plate == eave)
                   for p in plates)
