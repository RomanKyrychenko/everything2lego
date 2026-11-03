"""feature_box: a window region's cell box grown one stud into the wall and to the smallest window part."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from semantic_labels import EMBEDDED, feature_box, instance  # noqa: E402


def test_window_on_minus_z_wall_grows_inward_and_to_1x2x2():
    # surface cells y 10-12, x 5, z 3 (one layer), facing -z: inward is +z
    lo, hi = feature_box([10, 5, 3], [13, 6, 4], [0, 0, -1], EMBEDDED["window"], (40, 20, 20))
    assert hi[2] - lo[2] == 2 and lo[2] == 3                   # surface + one stud behind it
    assert hi[0] - lo[0] == 6 and hi[1] - lo[1] == 2           # 6 plates high, 2 studs wide


def test_window_on_plus_x_wall_clipped_to_grid():
    lo, hi = feature_box([0, 9, 0], [8, 10, 3], [1, 0, 0], EMBEDDED["window"], (40, 10, 20))
    assert (lo[1], hi[1]) == (8, 10)                           # inward is -x
    assert hi[0] - lo[0] == 8 and hi[2] - lo[2] == 3           # already big enough: kept
    assert np.all(lo >= 0)


def test_nested_regions_one_window_neighbours_two():
    area = np.full((1, 1, 10), -1, np.int32); cat = np.full((1, 1, 10), -1, np.int8)
    pane = np.zeros((1, 1, 10), bool); pane[..., 1:3] = True
    k = instance(area, cat, pane, 0); area[pane], cat[pane] = k, 0
    frame = np.zeros_like(pane); frame[..., 0:4] = True                # the pane with its frame: same window
    assert instance(area, cat, frame, 0) == k
    area[frame & (cat < 0)], cat[frame] = k, 0
    nxt = np.zeros_like(pane); nxt[..., 4:8] = True                     # the next window, touching: a new one
    assert instance(area, cat, nxt, 0) == k + 1
    assert instance(area, cat, frame, 1) == k + 1                       # another category never joins
