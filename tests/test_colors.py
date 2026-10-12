"""House palette, palette reduction and per-region speckle smoothing."""

import numpy as np
import pytest

from everything2lego import colors, voxelize
from everything2lego.colors import palette


def code_of(index):
    return palette()[int(index)].code


def index_of(code):
    return colors.by_code_index(code)


@pytest.mark.parametrize("rgb, expected_code", [
    ((250, 250, 250), 15),    # off-white wall -> White
    ((235, 210, 160), 19),    # beige render -> Tan
    ((150, 155, 160), 71),    # mid grey -> Light Bluish Gray
    ((95, 100, 95), 72),      # slate -> Dark Bluish Gray
    ((95, 50, 25), 70),       # timber -> Reddish Brown
    ((210, 40, 20), 4),       # brick red -> Red
    ((120, 20, 20), 320),     # roof tile -> Dark Red
    ((15, 20, 25), 0),        # shadowed trim -> Black
    ((185, 220, 240), 71),    # glass -> trans light blue is not shipped, so 71
])
def test_house_palette_snaps_rgb_to_building_colors(rgb, expected_code):
    idx = colors.snap_rgb_to_house(np.array([rgb], dtype=np.uint8))
    assert code_of(idx[0]) == expected_code


def test_house_palette_entries_exist_and_glass_falls_back():
    codes = {code_of(i) for _, i in colors.house_palette()}
    assert codes == {15, 19, 71, 72, 70, 4, 320, 0}
    assert 43 not in {c.code for c in palette()}


def test_remap_to_house_keeps_empty_cells_and_shapes():
    grid = np.full((2, 3, 3), -1, dtype=np.int16)
    grid[0] = index_of(1)       # Blue is not a house colour
    grid[1, 1, 1] = index_of(19)
    out = colors.remap_to_house(grid)
    assert out.shape == grid.shape and out.dtype == grid.dtype
    assert (out[grid < 0] == -1).all()
    assert code_of(out[1, 1, 1]) == 19
    assert code_of(out[0, 0, 0]) in {15, 19, 71, 72, 70, 4, 320, 0}


def test_reduce_palette_merges_to_k_dominant_colors_deterministically():
    grid = np.full((1, 10, 10), index_of(15), dtype=np.int16)   # mostly white
    grid[0, :, :4] = index_of(4)                                 # red band
    grid[0, 0, 0] = index_of(151)                                # a very light grey speck
    grid[0, 9, 9] = index_of(320)                                # a dark red speck
    grid[0, 5, 5] = -1
    out = colors.reduce_palette(grid, 2)
    used = set(np.unique(out[out >= 0]).tolist())
    assert {code_of(i) for i in used} == {15, 4}
    assert code_of(out[0, 0, 0]) == 15 and code_of(out[0, 9, 9]) == 4
    assert out[0, 5, 5] == -1
    assert np.array_equal(out, colors.reduce_palette(grid, 2))
    assert np.array_equal(colors.reduce_palette(grid, 10), grid)
    with pytest.raises(ValueError):
        colors.reduce_palette(grid, 0)


def test_speckle_smoothing_removes_small_regions_and_keeps_large_ones():
    wall, speck = index_of(15), index_of(4)
    grid = np.full((4, 8, 8), wall, dtype=np.int16)
    grid[1, 2, 2] = speck                 # isolated voxel
    grid[2, 4, 4:7] = speck               # 3-voxel run (< 4)
    grid[0:2, 6:8, 0:2] = speck           # 8-voxel block (>= 4)
    grid[3, :, 0] = -1                    # empty cells are left alone
    out = voxelize.smooth_color_regions(grid, min_region=4)
    assert out[1, 2, 2] == wall
    assert (out[2, 4, 4:7] == wall).all()
    assert (out[0:2, 6:8, 0:2] == speck).all()
    assert (out[3, :, 0] == -1).all()
    assert (out[grid >= 0] >= 0).all()
    # Exactly N voxels survive; N-1 do not.
    grid = np.full((1, 6, 6), wall, dtype=np.int16)
    grid[0, 0, :4] = speck
    assert (voxelize.smooth_color_regions(grid, 4)[0, 0, :4] == speck).all()
    assert (voxelize.smooth_color_regions(grid, 5)[0, 0, :4] == wall).all()
    # 1 or 0 disables; a region with no differently coloured neighbour stays.
    assert np.array_equal(voxelize.smooth_color_regions(grid, 1), grid)
    lone = np.full((1, 1, 1), speck, dtype=np.int16)
    assert voxelize.smooth_color_regions(lone, 4)[0, 0, 0] == speck


def test_hollow_and_resample_grid_helpers():
    grid = voxelize.dummy_grid_from_primitive("sphere", radius_studs=4)
    shell = voxelize.hollow_grid(grid, wall_thickness=1)
    assert 0 < shell.occupied().sum() < grid.occupied().sum()
    assert (shell.palette_idx[shell.occupied()] == grid.palette_idx[shell.occupied()]).all()
    assert shell.occupied()[0].sum() == grid.occupied()[0].sum()   # floor kept
    small = voxelize.resample_grid(grid, 5)
    assert max(small.shape_yxz[1], small.shape_yxz[2]) == 5
    assert small.metadata["max_studs"] == 5
    assert set(np.unique(small.palette_idx)) <= set(np.unique(grid.palette_idx))
