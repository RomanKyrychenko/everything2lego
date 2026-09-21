"""Regression tests for geometry, color fidelity and usable CLI exports."""

import json

import numpy as np
import pytest
import trimesh
from PIL import Image, ImageDraw
from typer.testing import CliRunner

from everything2lego import carve, io_ldraw, pack, pipeline, stability, voxelize
from everything2lego.bricks import catalog
from everything2lego.cli import app
from everything2lego.colors import palette


def color(name):
    return next(i for i, c in enumerate(palette()) if c.name == name)


def part(shape, y=0, x=0):
    p = next(p for p in catalog() if p.shape == shape and p.footprint == (1, 1))
    return pack.Placement(p, color("Red"), x, 0, y, 1, 1, 0)


def test_tiles_cannot_connect_bricks_above():
    report = stability.check([part("tile"), part("brick", y=1)])
    assert not report.ok
    assert len(report.floating) == 1
    assert report.components == 2


def test_export_color_ids_refer_to_the_same_colors():
    expected = {"Pink": (13, 23), "Medium Azure": (322, 156),
                "Very Light Bluish Gray": (151, 99), "Light Gray": (7, 9)}
    for name, identifiers in expected.items():
        entry = palette()[color(name)]
        assert (entry.code, entry.bl) == identifiers


def test_grounded_separate_parts_are_not_one_assembly():
    report = stability.check([part("brick"), part("brick", x=1)])
    assert not report.ok
    assert report.components == 2


def test_floating_stack_and_overlaps_are_rejected():
    report = stability.check([part("brick", y=4), part("brick", y=7)])
    assert len(report.floating) == 2
    assert not stability.check([part("brick"), part("brick")]).ok
    assert stability.check([part("brick"), part("brick")]).overlaps == [(0, 1)]


def test_supports_preserve_source_and_make_one_assembly():
    original = voxelize.dummy_grid_from_primitive(radius_studs=3)
    result = pipeline.assemble(original)
    assert result.report["geometrically_connected"]
    assert result.report["added_support_voxels"] > 0
    coords = np.argwhere(original.occupied())
    lo, hi = coords.min(axis=0), coords.max(axis=0) + 1
    cropped = original.palette_idx[tuple(slice(a, b) for a, b in zip(lo, hi))]
    restored = result.grid.palette_idx[2:, 1:-1, 1:-1]
    assert np.array_equal(restored[cropped >= 0], cropped[cropped >= 0])
    counts = np.zeros(result.grid.shape_yxz, dtype=int)
    for p in result.placements:
        region = (slice(p.y_plate, p.y_plate + p.part.height_units),
                  slice(p.x_stud, p.x_stud + p.w_world),
                  slice(p.z_stud, p.z_stud + p.d_world))
        counts[region] += 1
        assert np.all(result.grid.palette_idx[region] == p.color_idx)
    assert np.array_equal(counts, result.grid.occupied().astype(int))


@pytest.mark.parametrize("dimensions", [(1, 1), (1, 12), (5, 7), (16, 16)])
def test_foundation_connects_different_footprints(dimensions):
    grid = voxelize.VoxelGrid(np.full((3, *dimensions), color("Red"), dtype=np.int16))
    assert pipeline.assemble(grid).report["geometrically_connected"]


def test_alpha_and_supplied_masks(tmp_path):
    rgba = Image.new("RGBA", (64, 64), (255, 0, 0, 0))
    ImageDraw.Draw(rgba).rectangle((20, 20, 43, 43), fill=(255, 255, 255, 255))
    path = tmp_path / "white.png"
    rgba.save(path)
    views = carve.load_views([path, path])
    assert views[0].mask.sum() == 24 * 24
    mask_path = tmp_path / "mask.png"
    mask = Image.new("L", (64, 64), 0)
    ImageDraw.Draw(mask).rectangle((16, 16, 47, 47), fill=255)
    mask.save(mask_path)
    assert carve.load_views([path, path], masks=[mask_path, mask_path])[0].mask.sum() == 32 * 32


def test_front_and_back_keep_distinct_colors(tmp_path):
    paths = []
    for name in ("Red", "Blue"):
        img = Image.new("RGB", (64, 64), "white")
        ImageDraw.Draw(img).rectangle((16, 16, 47, 47), fill=palette()[color(name)].rgb)
        path = tmp_path / f"{name}.png"
        img.save(path)
        paths.append(path)
    grid = carve.carve(paths, angles=[0, 180], max_studs=8)
    middle = grid.palette_idx.shape[0] // 2
    assert grid.palette_idx[middle, 4, -1] == color("Red")
    assert grid.palette_idx[middle, 4, 0] == color("Blue")


def test_vote_count_does_not_overflow():
    rgb = np.full((24, 24, 3), (200, 20, 10), dtype=np.uint8)
    view = carve.View(rgb, np.ones((24, 24), dtype=bool), 0)
    occupied, _ = carve._carve_grid([view] * 260, 2, 1.0)
    assert occupied.all()


@pytest.mark.parametrize("fraction", [0, -1, 1.1, float("nan")])
def test_invalid_carving_fraction(fraction):
    with pytest.raises(ValueError, match="min_views_fraction"):
        carve._carve_grid([], 8, fraction)


def test_mesh_color_coordinates_and_determinism():
    mesh = trimesh.creation.box(extents=(2, 2, 2))
    mesh.visual.face_colors = np.array([palette()[color("Red" if n[0] < 0 else "Blue")].rgb + (255,)
                                       for n in mesh.face_normals])
    grid = voxelize.voxelize(mesh, max_studs=8, color_samples=10000)
    h = grid.shape_yxz[0] // 2
    assert grid.palette_idx[h, 0, 4] == color("Red")
    assert grid.palette_idx[h, -1, 4] == color("Blue")
    assert np.array_equal(grid.palette_idx, voxelize.voxelize(mesh, 8, 10000).palette_idx)


def test_mesh_samples_texture_within_triangles():
    mesh = trimesh.Trimesh(vertices=[[-1, 0, -1], [1, 0, -1], [1, 0, 1], [-1, 0, 1]],
                           faces=[[0, 1, 2], [0, 2, 3]], process=False)
    texture = Image.new("RGB", (64, 64), palette()[color("Red")].rgb)
    ImageDraw.Draw(texture).rectangle((32, 0, 63, 63), fill=palette()[color("Blue")].rgb)
    mesh.visual = trimesh.visual.TextureVisuals(uv=[[0, 0], [1, 0], [1, 1], [0, 1]], image=texture)
    grid = voxelize.voxelize(mesh, 8, 10000)
    assert grid.palette_idx[0, 1, 4] == color("Red")
    assert grid.palette_idx[0, -2, 4] == color("Blue")


def test_pbr_factors_work_with_and_without_textures():
    from trimesh.visual.material import PBRMaterial
    uv = np.zeros((3, 2))
    plain = PBRMaterial(baseColorFactor=[0.5, 0, 0, 1])
    textured = PBRMaterial(baseColorFactor=[0.5, 0, 0, 1], baseColorTexture=Image.new("RGB", (4, 4), "white"))
    actual = voxelize._material_rgb(plain, uv)
    assert actual.shape == (3, 3)
    assert 187 <= actual[0, 0] <= 189
    assert (actual[:, 1:] == 0).all()
    assert np.array_equal(actual, voxelize._material_rgb(textured, uv))
    mesh = trimesh.creation.box()
    mesh.visual = trimesh.visual.TextureVisuals(material=plain)
    assert voxelize.voxelize(mesh, 4, 1000).occupied().any()


def test_cli_errors_are_actionable(tmp_path):
    runner = CliRunner()
    result = runner.invoke(app, ["carve", str(tmp_path)])
    assert result.exit_code == 2
    assert "At least two views" in result.output
    result = runner.invoke(app, ["demo", "--radius", "0"])
    assert result.exit_code == 2


def test_complete_export(tmp_path):
    result = CliRunner().invoke(app, ["demo", "--radius", "2", "--out", str(tmp_path)])
    assert result.exit_code == 0, result.output
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["geometrically_connected"]
    assert (tmp_path / "preview.png").exists()
    assert (tmp_path / "instructions.pdf").read_bytes().startswith(b"%PDF")
    assert "0 STEP" in (tmp_path / "model.ldr").read_text()
    viewer = (tmp_path / "viewer/index.html").read_text()
    assert '<script id="scene-data"' in viewer
    assert '<script src=' not in viewer
    assert 'unpkg.com' not in viewer
