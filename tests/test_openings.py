"""Windows and doors are found on the mesh's wall colours, snapped to real
frames and doors, made identical where they repeat, and the walls are
packed around them."""

import numpy as np

from everything2lego import stability
from everything2lego.semantic_v2 import SemanticLegoPipeline, PipelineConfig, openings
from everything2lego.semantic_v2.detector import RegionDetector
from everything2lego.semantic_v2.mesh_target import MeshTarget
from everything2lego.semantic_v2.regions import PlanarRegion
from meshes import house


def target(mesh, studs=30):
    return MeshTarget.from_mesh(mesh, max_studs=studs, n_samples=60000, color_samples=60000, seed=7)


def front_wall(t):
    walls = [r for r in RegionDetector().detect(t)
             if isinstance(r, PlanarRegion) and r.is_vertical]
    return next(r for r in walls if openings.wall_frame(r, t).other == 20)


def _window_rows(frames):
    """Frames side by side on one course are one window: (x0, x1, y, parts)."""
    rows = []
    for p in sorted(frames, key=lambda q: (q.y_plate, q.x_stud)):
        if rows and rows[-1][2] == p.y_plate and rows[-1][1] == p.x_stud:
            x0, _, y, parts = rows[-1]
            rows[-1] = (x0, p.x_stud + p.w_world, y, parts + [p.part.ldraw])
        else:
            rows.append((p.x_stud, p.x_stud + p.w_world, p.y_plate, [p.part.ldraw]))
    return rows


def test_two_windows_and_a_door_are_found_on_the_front_wall():
    t = target(house())
    found = openings.find_openings(front_wall(t), t)
    doors = [p for p in found if p.part.shape == "door"]
    assert len(doors) == 1 and doors[0].y_plate == 0 and doors[0].part.ldraw == "60623.dat"
    windows = _window_rows([p for p in found if p.part.shape == "window"])
    assert len(windows) == 2
    # each 6-stud window is a row of identical frames as wide as it, the twins alike
    (a0, a1, ay, ap), (b0, b1, by, bp) = windows
    assert a1 - a0 == b1 - b0 == 6 and ay == by == 15 and ap == bp and len(set(ap)) == 1
    assert all(p.w_world == p.part.footprint[0] and p.d_world == 1 for p in found)
    # with clear glass in every frame
    assert all(p.glass == 47 for p in found if p.part.shape == "window")


def test_plain_wall_has_no_openings():
    t = target(house(windows=(), door=(0, 0, 0, 0)))
    assert openings.find_openings(front_wall(t), t) == []


def test_blob_is_peeled_into_its_rectangles():
    cells = np.zeros((20, 30), dtype=bool)
    cells[2:8, 10:25] = True          # a window
    cells[8, 14:18] = True            # colour bleeding into the gap
    cells[9:13, 0:19] = True          # a door
    boxes = openings.rectangles(cells)
    assert sorted(boxes) == [(2, 8, 10, 25), (9, 13, 0, 19)]


def test_walls_are_packed_around_the_openings():
    t = target(house())
    result = SemanticLegoPipeline(PipelineConfig(max_studs=30)).run_on_target(t)
    report = stability.check(result.placements)
    assert report.ok
    kept = [p for p in result.placements if p.part.shape in ("window", "door")]
    assert sum(p.part.shape == "door" for p in kept) == 1
    assert len(_window_rows([p for p in kept if p.part.shape == "window"])) == 2
    bricks = [p for p in result.placements if p.part.shape == "brick" and p.z_stud == 20]
    for o in kept:
        for b in bricks:
            x_overlap = b.x_stud < o.x_stud + o.w_world and o.x_stud < b.x_stud + b.w_world
            y_overlap = (b.y_plate < o.y_plate + o.part.height_units
                         and o.y_plate < b.y_plate + b.part.height_units)
            assert not (x_overlap and y_overlap), (o, b)


def test_openings_can_be_turned_off():
    from everything2lego.semantic_v2.builders import BuilderConfig
    cfg = PipelineConfig(max_studs=30, builder=BuilderConfig(detect_openings=False))
    result = SemanticLegoPipeline(cfg).run_on_target(target(house()))
    assert not any(p.part.shape in ("window", "door") for p in result.placements)
