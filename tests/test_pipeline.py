"""End-to-end smoke tests for the Phase 1 pipeline."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from everything2lego import backends, carve, io_bricklink, io_ldraw, pack, stability, viewer_export, voxelize
from everything2lego.bricks import catalog
from everything2lego.colors import palette, snap_rgb_to_palette


def test_palette_loads():
    pal = palette()
    assert len(pal) > 20
    assert any(c.name == "Red" for c in pal)
    assert any(c.name == "White" for c in pal)


def test_catalog_loads_and_has_1x1_plate():
    cat = catalog()
    assert any(p.ldraw == "3024.dat" for p in cat), "1x1 plate is required for leftover fill"
    assert any(p.shape == "brick" for p in cat)
    assert any(p.shape == "tile" for p in cat)
    assert any(p.shape == "slope" for p in cat), "slopes are required for smoothing step edges"


def test_slope_refinement_swaps_top_brick_at_step_edge():
    """Two-tier step: a 2-stud-wide brick sitting on top of a 4-stud-wide
    brick, with the shorter one flush against one side. The top brick's +x
    neighbour drops down one brick height; slope 3040 should replace it,
    pointing low toward +x (rotation 0)."""
    grid_arr = np.full((6, 4, 1), -1, dtype=np.int16)
    color = 0  # any valid palette index
    grid_arr[0:3, 0:4, 0] = color
    grid_arr[3:6, 0:2, 0] = color
    grid = voxelize.VoxelGrid(palette_idx=grid_arr, metadata={})

    placements = pack.pack(grid, use_tiles=False, use_slopes=True)
    slopes = [p for p in placements if p.part.shape == "slope"]
    assert len(slopes) == 1, "expected exactly one slope on the exposed top brick"
    s = slopes[0]
    assert s.part.ldraw == "3040.dat"
    assert (s.w_world, s.d_world) == (2, 1)
    assert s.y_plate == 3
    assert s.rotation_deg == 0, "low side should point toward +x"


def test_slope_refinement_preserves_voxel_coverage():
    grid_arr = np.full((6, 4, 1), -1, dtype=np.int16)
    grid_arr[0:3, 0:4, 0] = 0
    grid_arr[3:6, 0:2, 0] = 0
    grid = voxelize.VoxelGrid(palette_idx=grid_arr, metadata={})

    with_slopes = pack.pack(grid, use_tiles=False, use_slopes=True)
    without_slopes = pack.pack(grid, use_tiles=False, use_slopes=False)

    def _volume(placements):
        return sum(p.w_world * p.d_world * p.part.height_units for p in placements)

    assert _volume(with_slopes) == _volume(without_slopes)


def test_stepped_pyramid_gets_edge_slopes_and_corner_slopes():
    """The pyramid primitive with 2-stud step-backs should exercise both the
    edge-slope (single-direction) and corner-slope (two-adjacent-direction)
    branches of the refinement pass."""
    grid = voxelize.dummy_grid_from_primitive("pyramid", radius_studs=3)
    placements = pack.pack(grid, use_tiles=False, use_slopes=True)
    edge_slopes = [p for p in placements if p.part.shape == "slope"]
    corner_slopes = [p for p in placements if p.part.shape == "slope_corner"]
    assert len(edge_slopes) >= 4, f"expected edge slopes along the exposed rings, got {len(edge_slopes)}"
    assert len(corner_slopes) >= 4, f"expected corner slopes at outside corners, got {len(corner_slopes)}"
    # Every corner slope must use one of the four canonical rotations.
    for c in corner_slopes:
        assert c.rotation_deg in (0, 90, 180, 270)


def test_corner_slope_placed_on_outside_corner():
    """6x6 base with a 2x2 tower on the (0, 0) corner: the tower's top brick
    has step-downs on both +x and +z, which matches the corner-slope 3045."""
    grid_arr = np.full((6, 6, 6), -1, dtype=np.int16)
    grid_arr[0:3, 0:6, 0:6] = 4
    grid_arr[3:6, 0:2, 0:2] = 4
    grid = voxelize.VoxelGrid(palette_idx=grid_arr, metadata={})

    placements = pack.pack(grid, use_tiles=False, use_slopes=True)
    corners = [p for p in placements if p.part.shape == "slope_corner"]
    assert len(corners) == 1
    c = corners[0]
    # 3045 is the canonical outside corner; 3676 is the inverted alternative.
    # Position-hash variety may pick either. Both are valid outside corners.
    assert c.part.ldraw in ("3045.dat", "3676.dat")
    assert (c.w_world, c.d_world) == (2, 2)
    assert c.rotation_deg == 0, "low corner should point toward (+x, +z)"


@pytest.mark.parametrize("origin,expected_rot", [
    ((0, 0), 0),    # +x, +z step-down -> rot 0
    ((4, 0), 270),  # -x, +z step-down -> rot 270
    ((0, 4), 90),   # +x, -z step-down -> rot 90
    ((4, 4), 180),  # -x, -z step-down -> rot 180
])
def test_corner_slope_rotation_matches_step_direction(origin, expected_rot):
    """Same 2x2 tower placed on each of the four corners of a 6x6 base.
    The corner-slope rotation must follow the low-corner direction."""
    grid_arr = np.full((6, 6, 6), -1, dtype=np.int16)
    grid_arr[0:3, 0:6, 0:6] = 4
    ox, oz = origin
    grid_arr[3:6, ox:ox + 2, oz:oz + 2] = 4
    grid = voxelize.VoxelGrid(palette_idx=grid_arr, metadata={})

    placements = pack.pack(grid, use_tiles=False, use_slopes=True)
    corners = [p for p in placements if p.part.shape == "slope_corner"]
    assert len(corners) == 1, f"corner {origin}: expected 1 corner slope, got {len(corners)}"
    assert corners[0].rotation_deg == expected_rot, \
        f"corner {origin}: expected rotation {expected_rot}, got {corners[0].rotation_deg}"


def test_slope_pre_bias_produces_slopes_on_a_wide_step_ramp():
    """Three-tier ramp with 2-stud-deep step-backs. With the level-consistency
    bias in pack(), the packer keeps 2x2 candidates at each step edge so the
    refinement pass can swap them for slope 3039."""
    grid_arr = np.full((9, 6, 2), -1, dtype=np.int16)
    grid_arr[0:3, 0:6, 0:2] = 4
    grid_arr[3:6, 0:4, 0:2] = 4
    grid_arr[6:9, 0:2, 0:2] = 4
    grid = voxelize.VoxelGrid(palette_idx=grid_arr, metadata={})

    placements = pack.pack(grid, use_tiles=False, use_slopes=True)
    slopes = [p for p in placements if p.part.shape == "slope"]
    assert len(slopes) >= 2, f"expected >=2 slopes at the two step edges, got {len(slopes)}"
    # Position-hash variety may swap the 2x2 slope 3039 for another 2x2 slope
    # (curved 15068 etc). Just check they're 2x2 slopes pointing +x.
    for s in slopes:
        assert (s.w_world, s.d_world) == (2, 2)
        assert s.rotation_deg == 0, "low side should point toward +x on this ramp"


def test_slope_rotation_covers_all_four_directions():
    """Sanity-check the low-direction / rotation cycle by rebuilding it from
    the io_ldraw rotation matrices. If someone changes either table in
    isolation, this test should catch the drift."""
    from everything2lego.io_ldraw import _ROT
    from everything2lego.pack import _LOW_DIR_CYCLE

    dir_to_vec = {"+x": (1, 0), "-x": (-1, 0), "+z": (0, 1), "-z": (0, -1)}
    for i, world_dir in enumerate(_LOW_DIR_CYCLE):
        rot = i * 90
        m = _ROT[rot]
        # Apply rotation to the default +x direction and read the world (x, z).
        local = (1, 0, 0)
        world_x = m[0] * local[0] + m[1] * local[1] + m[2] * local[2]
        world_z = m[6] * local[0] + m[7] * local[1] + m[8] * local[2]
        expected_x, expected_z = dir_to_vec[world_dir]
        assert (world_x, world_z) == (expected_x, expected_z), \
            f"rotation {rot} sends +x to ({world_x}, {world_z}), not {world_dir}"


def test_snap_rgb_maps_white_and_red():
    idx = snap_rgb_to_palette(np.array([[255, 255, 255], [200, 20, 10]], dtype=np.uint8))
    pal = palette()
    assert pal[idx[0]].name == "White"
    assert pal[idx[1]].name == "Red"


def test_end_to_end_demo(tmp_path: Path):
    grid = voxelize.dummy_grid_from_primitive("sphere", radius_studs=3)
    placements = pack.pack(grid, use_tiles=True)
    assert len(placements) > 0

    ldr = io_ldraw.write_ldr(placements, tmp_path / "model.ldr")
    xml_path = io_bricklink.write_wanted_list(placements, tmp_path / "wanted.xml")
    viewer_index = viewer_export.write_viewer(placements, tmp_path / "viewer")

    assert ldr.exists() and ldr.stat().st_size > 0
    # Every non-comment line in the .ldr should be a type-1 line with 15 tokens.
    for line in ldr.read_text().splitlines():
        if line.startswith("1 "):
            tokens = line.split()
            assert len(tokens) == 15, line

    # XML should parse and contain <ITEM> elements.
    tree = ET.parse(xml_path)
    items = tree.getroot().findall("ITEM")
    assert len(items) > 0
    for item in items:
        assert item.find("ITEMTYPE").text == "P"
        assert int(item.find("MINQTY").text) > 0

    # Viewer bundle has index.html, main.js, scene.json.
    v = viewer_index.parent
    assert (v / "index.html").exists()
    assert (v / "main.js").exists()
    scene = json.loads((v / "scene.json").read_text())
    assert len(scene["bricks"]) == len(placements)


def test_stability_check_runs():
    grid = voxelize.dummy_grid_from_primitive("sphere", radius_studs=3)
    placements = pack.pack(grid, use_tiles=False)
    report = stability.check(placements)
    assert report.total == len(placements)
    assert report.supported + len(report.floating) == report.total


def _synthetic_ball_views(tmp_path: Path, n_views: int = 6, size: int = 240):
    """Create N identical-silhouette images of a red ball on a white background.
    Because a sphere's silhouette is rotationally invariant, this is enough to
    exercise the space carver end-to-end deterministically."""
    from PIL import Image, ImageDraw
    paths = []
    r = size // 4
    for i in range(n_views):
        img = Image.new("RGB", (size, size), (255, 255, 255))
        d = ImageDraw.Draw(img)
        cx, cy = size // 2, size // 2
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(200, 40, 40))
        p = tmp_path / f"view_{i:02d}.png"
        img.save(p)
        paths.append(p)
    return paths


def test_carve_produces_roughly_spherical_grid(tmp_path: Path):
    paths = _synthetic_ball_views(tmp_path, n_views=6)
    grid = carve.carve(paths, max_studs=16, tolerance=40)
    occ = grid.occupied()
    assert occ.any()
    # The grid should be roughly cubic in horizontal extent (visual hull of a sphere).
    H, W, D = occ.shape
    assert W == 16
    assert D == 16
    # Occupancy should be dense near the middle plate row.
    mid_slice = occ[H // 2]
    assert mid_slice.sum() > (W * D) * 0.3


def test_carve_backend_dispatch(tmp_path: Path):
    paths = _synthetic_ball_views(tmp_path, n_views=4)
    grid = backends.build("carve", paths, max_studs=12, tolerance=40)
    assert grid.occupied().any()


def test_project_photo_colors_paints_vertex_colors(tmp_path: Path):
    """Project a two-tone image onto a small cube mesh and check the top half
    of the cube ends up red-ish, the bottom half blue-ish."""
    import trimesh as _tm
    from PIL import Image as _Image
    from everything2lego.voxelize import project_photo_colors

    # 200x200 image: top half red, bottom half blue.
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[:100] = (220, 40, 40)
    img[100:] = (30, 30, 200)
    img_path = tmp_path / "twotone.png"
    _Image.fromarray(img).save(img_path)

    cube = _tm.creation.box(extents=(1.0, 1.0, 1.0))
    coloured = project_photo_colors(cube, img_path, forward_axis="-z",
                                     crop_to_subject=False, flip_v=False)

    verts = np.asarray(coloured.vertices)
    rgb = np.asarray(coloured.visual.vertex_colors)[:, :3]
    top = rgb[verts[:, 1] > 0]
    bottom = rgb[verts[:, 1] < 0]
    # Top vertices should mostly be red-dominant; bottom mostly blue-dominant.
    assert (top[:, 0] > top[:, 2]).mean() > 0.5
    assert (bottom[:, 2] > bottom[:, 0]).mean() > 0.5


def test_hunyuan3d_backend_registered():
    # Fully lazy — registered so the CLI can surface a good error even
    # without the vendor checkout / weights present.
    assert "hunyuan3d" in backends.BACKENDS


def test_neural_backends_registered():
    """`triposr`, `hybrid` and `dust3r` are lazy-imported: registered even
    without the torch model weights so the CLI can surface a good error."""
    assert "triposr" in backends.BACKENDS
    assert "hybrid" in backends.BACKENDS
    assert "dust3r" in backends.BACKENDS


def test_dust3r_background_filter_removes_matching_colors():
    pytest.importorskip("torch", reason="DUSt3R backend needs torch (pip install -e '.[neural]')")
    from everything2lego.backends import dust3r as _dust3r
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=np.float32)
    rgb = np.array([[100, 100, 100], [200, 40, 40], [98, 102, 101]], dtype=np.uint8)
    backgrounds = np.array([[100, 100, 100]], dtype=np.int16)
    kept_pts, kept_rgb = _dust3r._drop_background(pts, rgb, backgrounds, tolerance=30)
    assert len(kept_pts) == 1
    assert (kept_rgb[0] == [200, 40, 40]).all()


def test_dust3r_outlier_filter_drops_far_points():
    pytest.importorskip("torch", reason="DUSt3R backend needs torch (pip install -e '.[neural]')")
    from everything2lego.backends import dust3r as _dust3r
    rng = np.random.default_rng(0)
    pts = rng.normal(0, 1, size=(1000, 3)).astype(np.float32)
    pts = np.concatenate([pts, np.array([[100.0, 100.0, 100.0]], dtype=np.float32)])
    rgb = np.zeros((len(pts), 3), dtype=np.uint8)
    kept_pts, kept_rgb = _dust3r._drop_outliers(pts, rgb, k=3.0)
    assert len(kept_pts) < len(pts)
    assert not np.any(np.all(kept_pts == 100.0, axis=1))


def test_dust3r_ground_snap_zeroes_bottom_and_drops_underground():
    pytest.importorskip("torch", reason="DUSt3R backend needs torch (pip install -e '.[neural]')")
    from everything2lego.backends import dust3r as _dust3r
    rng = np.random.default_rng(1)
    pts = rng.uniform(0, 5, size=(500, 3)).astype(np.float32)
    pts[:, 1] += 10  # lift so ground snap has work to do
    pts = np.concatenate([pts, np.array([[0.0, -5.0, 0.0]], dtype=np.float32)])
    rgb = np.zeros((len(pts), 3), dtype=np.uint8)
    kept_pts, _ = _dust3r._snap_ground_plane(pts, rgb, percentile=2.0)
    assert kept_pts[:, 1].min() >= -1e-6, "no points should sit below y=0"
    # Underground artefact (y=-5 initially → -15 after shift) must be dropped.
    assert kept_pts[:, 1].max() < 10, "under-ground artefact should be gone"


def test_voxelize_pointcloud_from_synthetic_cube():
    """A cube-shaped cloud voxelizes back into a solid cube region at the
    requested scale."""
    rng = np.random.default_rng(0)
    side_mm = 200.0
    pts = rng.uniform(-side_mm / 2, side_mm / 2, size=(50_000, 3)).astype(np.float32)
    rgb = np.tile(np.array([200, 40, 40], dtype=np.uint8), (len(pts), 1))
    grid = voxelize.voxelize_pointcloud(pts, rgb, max_studs=16, min_points_per_voxel=1,
                                         close_iterations=0)
    assert grid.occupied().sum() > 0
    # Horizontal extent should be close to max_studs (16). Allow rounding.
    _, W, D = grid.shape_yxz
    assert 15 <= max(W, D) <= 17
    # All colors should snap to Red (LDraw code 4).
    idxs = grid.palette_idx[grid.palette_idx >= 0]
    from everything2lego.colors import palette
    pal = palette()
    assert all(pal[int(i)].name == "Red" for i in np.unique(idxs))


def test_pack_leaves_no_leftover_voxels():
    grid = voxelize.dummy_grid_from_primitive("sphere", radius_studs=3)
    placements = pack.pack(grid, use_tiles=False)

    accounted = np.zeros_like(grid.palette_idx, dtype=bool)
    for p in placements:
        accounted[
            p.y_plate:p.y_plate + p.part.height_units,
            p.x_stud:p.x_stud + p.w_world,
            p.z_stud:p.z_stud + p.d_world,
        ] = True
    occupied = grid.palette_idx >= 0
    assert np.array_equal(accounted, occupied), "packer must claim exactly the occupied voxels"


def test_catalog_keeps_only_plain_geometry_from_imports():
    """Imported Rebrickable parts share shape labels with arches, round bricks,
    brackets, wedge plates and window glass; none of those may reach the packer."""
    cat = catalog()
    names = {p.name for p in cat}
    for bad in ("Arch", "Round", "Curved", "Bracket", "Wedge", "Glass", "Clip",
                "Cone", "Dish", "Shutter", "Macaroni", "Special", "Double"):
        assert not any(bad in p.name for p in cat if p.shape not in ("slope", "slope_inv", "slope_corner")), bad
    assert "Brick 1 x 12" in names and "Brick 1 x 16" in names
    for p in cat:
        if p.shape == "brick":
            assert p.height_units == 3, p
        if p.shape == "plate":
            assert p.height_units == 1, p
    tall = {p.name: p.height_units for p in cat if p.shape == "brick_tall"}
    assert tall["Brick 1 x 1 x 3"] == 9 and tall["Brick 1 x 2 x 5"] == 15
    from everything2lego.bricks import full_catalog
    from everything2lego.templates.base import part_by_ldraw
    assert part_by_ldraw("4733.dat") is not None and not part_by_ldraw("4733.dat").plain
    assert len(full_catalog()) > len(cat)
