"""Post-pack styling passes must keep the model's geometry sound: every
cell covered once, nothing floating that was not floating before."""

import numpy as np
import pytest

from everything2lego import pack, stability, voxelize
from everything2lego.bricks import full_catalog
from everything2lego.colors import palette
from everything2lego.pack import Placement


def code_idx(code):
    return next(i for i, c in enumerate(palette()) if c.code == code)


def part(ldraw):
    return next(p for p in full_catalog() if p.ldraw == ldraw)


def brick(ldraw, x, z, y, code=4, w=None, d=None):
    p = part(ldraw)
    w = p.footprint[0] if w is None else w
    d = p.footprint[1] if d is None else d
    return Placement(p, code_idx(code), x, z, y, w, d, 0 if (w, d) == p.footprint else 90)


def cells(placements):
    out = {}
    for p in placements:
        for y in range(p.y_plate, p.y_plate + p.part.height_units):
            for x in range(p.x_stud, p.x_stud + p.w_world):
                for z in range(p.z_stud, p.z_stud + p.d_world):
                    out[(y, x, z)] = out.get((y, x, z), 0) + 1
    return out


@pytest.fixture
def house():
    """A packed 12x8 box, 9 bricks tall, in one wall colour."""
    grid = np.full((27, 12, 8), -1, dtype=np.int16)
    grid[:, :, :] = code_idx(4)
    return pack.pack(voxelize.VoxelGrid(palette_idx=grid, metadata={}))


def same_geometry(before, after):
    return cells(before) == cells(after)


@pytest.mark.parametrize("preset", ["log", "masonry", "curved", "smooth"])
def test_texture_presets_keep_geometry(house, preset):
    out = pack.substitute_texture(house, preset)
    assert len(out) == len(house) and same_geometry(house, out)


def test_variety_and_architectural_style_keep_geometry(house):
    out = pack.apply_variety(house)
    assert same_geometry(house, out)
    styled, stats = pack.apply_architectural_style(house)
    assert same_geometry(house, styled)
    assert stats["roof_line"] is not None


def test_smooth_isolated_colors_recolours_a_lone_outlier():
    row = [brick("3005.dat", x, z, 0, code=4) for x in range(3) for z in range(3)]
    odd = row[4]                                        # the centre, four neighbours
    row[4] = Placement(odd.part, code_idx(1), odd.x_stud, odd.z_stud, odd.y_plate, 1, 1, 0)
    out = pack.smooth_isolated_colors(row)
    assert out[4].color_idx == code_idx(4)
    assert all(p.color_idx == code_idx(4) for p in out)


def test_window_detection_swaps_an_exact_cluster_and_nothing_else():
    wall = [brick("3010.dat", x, 0, y, code=4) for x in range(0, 12, 4) for y in range(0, 15, 3)]
    # A 2-wide, 2-brick (6-plate) hole in colour 71 at x=4..5, y=3..8.
    wall = [p for p in wall if not (p.x_stud == 4 and p.y_plate in (3, 6))]
    wall += [brick("3004.dat", 4, 0, y, code=71) for y in (3, 6)]
    wall += [brick("3004.dat", 6, 0, y, code=4) for y in (3, 6)]
    wall += [brick("3010.dat", x, 0, y, code=1) for x in range(0, 12, 4) for y in (15, 18)]  # roof
    out, stats = pack.detect_windows_and_doors(wall)
    assert stats["windows_placed"] == 1
    window = next(p for p in out if p.part.ldraw == "60592.dat")
    assert (window.x_stud, window.y_plate, window.part.height_units) == (4, 3, 6)
    assert max(cells(out).values()) == 1 and cells(out).keys() == cells(wall).keys()
    # A 3-brick-tall cluster of the same width is not a 2-brick window.
    taller = [p for p in wall if not (p.x_stud == 4 and p.y_plate == 9)]
    taller += [brick("3004.dat", 4, 0, 9, code=71), brick("3004.dat", 6, 0, 9, code=4)]
    out, stats = pack.detect_windows_and_doors(taller)
    assert stats["windows_placed"] == 1
    assert next(p for p in out if p.part.shape == "window").part.ldraw == "60800.dat"
    assert max(cells(out).values()) == 1


def test_window_detection_leaves_irregular_clusters_alone():
    wall = [brick("3010.dat", x, 0, y, code=4) for x in range(0, 12, 4) for y in range(0, 15, 3)]
    wall = [p for p in wall if not (p.x_stud == 4 and p.y_plate in (3, 6))]
    wall += [brick("3004.dat", 4, 0, 3, code=71), brick("3005.dat", 4, 0, 6, code=71),
             brick("3005.dat", 5, 0, 6, code=4), brick("3004.dat", 6, 0, 3, code=4),
             brick("3004.dat", 6, 0, 6, code=4)]
    wall += [brick("3010.dat", x, 0, y, code=1) for x in range(0, 12, 4) for y in (15, 18)]
    out, stats = pack.detect_windows_and_doors(wall)
    assert stats["windows_placed"] == 0 and max(cells(out).values()) == 1
