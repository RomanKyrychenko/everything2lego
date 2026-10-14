"""Windows as sets build them: a wide opening is a row of identical
frames, each with its glass clipped in; an opening has wall above and
below it; a band of another material along a wall is not a window."""

import numpy as np

from everything2lego.colors import _rgb_to_lab
from everything2lego.io_bricklink import write_wanted_list
from everything2lego.io_ldraw import placement_to_line
from everything2lego.semantic_v2.openings import COURSE, WallFrame, openings_from_grid


def _grid(length=24, height=12 * COURSE, wall=(30, 30, 32)):
    mean = np.tile(_rgb_to_lab(np.array(wall, np.float32) / 255.0), (length, height, 1))
    return WallFrame(True, 0, 5, 0, length, height), mean, np.ones((length, height), bool)


def _paint(mean, u0, u1, y0, y1, rgb):
    mean[u0:u1, y0:y1] = _rgb_to_lab(np.array(rgb, np.float32) / 255.0)


def test_a_wide_window_is_a_row_of_frames_with_glass():
    frame, mean, has = _grid()
    _paint(mean, 6, 14, 12, 21, (200, 200, 205))          # 8 studs by 3 bricks, daylit
    out = openings_from_grid(frame, mean, has)
    assert [p.part.ldraw for p in out] == ["60594.dat", "60594.dat"]     # two 1x4x3, side by side
    assert out[0].x_stud + 4 == out[1].x_stud and out[0].y_plate == out[1].y_plate
    assert all(p.glass == 47 for p in out)                # clear glass


def test_no_window_without_wall_above_or_below_and_no_bands():
    frame, mean, has = _grid()
    _paint(mean, 4, 10, 30, 36, (220, 220, 220))          # runs into the wall top: an eave
    _paint(mean, 0, 24, 0, 6, (120, 120, 120))            # a stone base the whole length
    _paint(mean, 14, 20, 0, 6, (220, 220, 220))           # low, short: not a door
    assert openings_from_grid(frame, mean, has) == []


def test_subtle_openings_are_found_by_the_walls_own_threshold():
    frame, mean, has = _grid(wall=(170, 120, 80))         # warm brick
    _paint(mean, 6, 10, 12, 21, (190, 150, 120))          # a window only ~14 dE off the brick
    assert openings_from_grid(frame, mean, has) == []
    assert len(openings_from_grid(frame, mean, has, adaptive=True)) >= 1


def test_the_glass_is_exported_with_its_frame(tmp_path):
    frame, mean, has = _grid()
    _paint(mean, 8, 12, 12, 21, (200, 200, 205))
    (p,) = openings_from_grid(frame, mean, has)
    lines = placement_to_line(p).splitlines()
    assert lines[0].endswith(p.part.ldraw) and lines[1].endswith("60603.dat")
    assert lines[1].split()[1] == "47"
    xml = write_wanted_list([p], tmp_path / "w.xml").read_text()
    assert "<ITEMID>60603</ITEMID>" in xml and "<COLOR>12</COLOR>" in xml   # BrickLink Trans-Clear
