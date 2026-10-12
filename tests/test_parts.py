"""Part labels as an optional input: a colour-coded mesh (from a segmenter
or painted by hand) or the built-in CPU segmenter gives the regions; the
facet detector stays the default."""

import json

import numpy as np
import trimesh

from everything2lego import stability
from everything2lego.semantic_v2 import SemanticLegoPipeline, PipelineConfig, parts
from everything2lego.semantic_v2.mesh_target import MeshTarget
from everything2lego.semantic_v2.regions import RegionLabel
from meshes import house


def painted_house():
    """The test house with one flat colour per part, as a segmenter would
    hand it over: walls, roof, windows and door each one colour."""
    m = house()
    fc = np.asarray(m.visual.face_colors)[:, :3]
    n = np.asarray(m.face_normals)
    out = np.tile([200, 200, 200], (len(m.faces), 1))          # walls
    out[n[:, 1] > 0.9] = [255, 0, 0]                            # roof
    out[n[:, 1] < -0.9] = [0, 255, 0]                           # underside
    out[(fc == [40, 60, 90]).all(axis=1)] = [0, 0, 255]         # windows
    out[(fc == [70, 40, 20]).all(axis=1)] = [255, 255, 0]       # door
    m.visual.face_colors = np.column_stack([out, np.full(len(out), 255)])
    return m


def test_labels_from_colours_and_sidecar_names(tmp_path):
    m = painted_house()
    path = tmp_path / "labels.glb"
    m.export(path)
    (tmp_path / "labels.json").write_text(json.dumps(
        {"#c8c8c8": "wall", "#ff0000": "roof", "#0000ff": "window", "#ffff00": "door"}))
    labels = parts.load_part_labels(path, house())
    assert len(labels.ids) == 5
    assert sorted(labels.names.values()) == ["door", "roof", "wall", "window"]


def test_labels_follow_a_decimated_mesh():
    m = painted_house()
    coarse = house(subdiv=3)             # same shape, a quarter of the faces
    labels = parts.labels_from_mesh(m, coarse)
    assert len(labels.labels) == len(coarse.faces) and len(labels.ids) == 5


def test_named_parts_become_regions_and_forced_openings(tmp_path):
    m = painted_house()
    path = tmp_path / "labels.glb"; m.export(path)
    (tmp_path / "labels.json").write_text(json.dumps(
        {"#c8c8c8": "wall", "#ff0000": "roof", "#0000ff": "window", "#ffff00": "door"}))
    original = house()
    labels = parts.load_part_labels(path, original)
    t = MeshTarget.from_mesh(original, max_studs=30, n_samples=30000, color_samples=30000, seed=3)
    regions = parts.regions_from_labels(t, labels)
    kinds = sorted(r.label.value for r in regions)
    # One "wall" part wrapping the house becomes a region per wall.
    assert kinds == ["planar_roof"] + ["planar_wall"] * 4, kinds
    forced = t.metadata["forced_openings"]
    assert sorted(b["name"] for b in forced) == ["door", "window", "window"]


def test_pipeline_from_labelled_mesh_builds_openings(tmp_path):
    m = painted_house()
    mesh_path, parts_path = tmp_path / "house.glb", tmp_path / "parts.glb"
    house().export(mesh_path); m.export(parts_path)
    (tmp_path / "parts.json").write_text(json.dumps(
        {"#c8c8c8": "wall", "#ff0000": "roof", "#0000ff": "window", "#ffff00": "door"}))
    cfg = PipelineConfig(max_studs=30, n_samples=30000, color_samples=30000, parts_path=str(parts_path))
    result = SemanticLegoPipeline(cfg).run(mesh_path)
    kinds = sorted(p.part.shape for p in result.placements if p.part.shape in ("window", "door"))
    assert kinds == ["door", "window", "window"]
    assert stability.check(result.placements).ok


def test_builtin_segmenter_separates_walls_roof_and_openings():
    m = house()
    labels = parts.segment_builtin(m)
    assert 5 <= len(labels.ids) <= 12, len(labels.ids)
    t = MeshTarget.from_mesh(m, max_studs=30, n_samples=30000, color_samples=30000, seed=3)
    regions = parts.regions_from_labels(t, labels)
    assert any(r.label == RegionLabel.PLANAR_ROOF for r in regions)
    assert sum(r.label == RegionLabel.PLANAR_WALL for r in regions) >= 4
    assert len(t.metadata["forced_openings"]) == 3
    result = SemanticLegoPipeline(PipelineConfig(max_studs=30, segmenter="builtin")).run_on_target(t)
    assert sorted(p.part.shape for p in result.placements if p.part.shape in ("window", "door")) == ["door", "window", "window"]
    assert stability.check(result.placements).ok


def test_without_labels_the_detector_is_used():
    t = MeshTarget.from_mesh(house(), max_studs=30, n_samples=30000, color_samples=30000, seed=3)
    result = SemanticLegoPipeline(PipelineConfig(max_studs=30)).run_on_target(t)
    assert "forced_openings" not in t.metadata
    assert result.placements
