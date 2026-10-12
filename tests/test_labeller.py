"""The vision part labeller, with the API faked: renders, request shape,
naming into the builder vocabulary, cache, graceful failure."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh
from PIL import Image
from typer.testing import CliRunner

from everything2lego import stability
from everything2lego.cli import app
from everything2lego.semantic_v2 import PipelineConfig, SemanticLegoPipeline, labeller, parts
from everything2lego.semantic_v2.mesh_target import MeshTarget
from everything2lego.semantic_v2.regions import RegionLabel
from meshes import house, noisy_house


@pytest.fixture(autouse=True)
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(labeller, "CACHE_DIR", tmp_path / "cache")
    return tmp_path / "cache"


def house_parts(mesh):
    """What each builtin-segmenter part of the house is, from its geometry,
    standing in for the model's eyes."""
    labels = parts.segment_builtin(mesh)
    n = np.asarray(mesh.face_normals)
    rgb = parts._face_colours(mesh)
    truth = {}
    for k in labels.ids:
        faces = labels.labels == k
        ny = float(n[faces, 1].mean())
        colour = rgb[faces].mean(axis=0)
        if ny > 0.9:
            truth[k] = ("roof", "pitched roof")
        elif ny < -0.9:
            truth[k] = ("foundation", "underside")
        elif colour[2] > colour[0] + 20:
            truth[k] = ("window", "window")
        elif colour[0] > colour[2] + 30 and colour[1] < 100:
            truth[k] = ("door", "front door")
        else:
            truth[k] = ("wall", "wall")
    return labels, truth


class FakeClient:
    """Answers like the SDK, from a {id: (kind, name)} truth table, and
    records every request."""

    def __init__(self, truth, *, fail=None, stop_reason="end_turn", text=None):
        self.truth, self.fail, self.stop_reason, self.text = truth, fail, stop_reason, text
        self.requests = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.requests.append(kwargs)
        if self.fail is not None:
            raise self.fail
        content = kwargs["messages"][0]["content"]
        asked = [int(b["text"].split()[1].rstrip(",")) for b in content
                 if b["type"] == "text" and b["text"].startswith("Part ")]
        body = self.text if self.text is not None else json.dumps({
            "object": "a small house",
            "parts": [{"id": k, "kind": self.truth[k][0], "name": self.truth[k][1], "confidence": 0.9}
                      for k in asked if k in self.truth]})
        return SimpleNamespace(stop_reason=self.stop_reason, model=kwargs["model"],
                               content=[SimpleNamespace(type="text", text=body)],
                               usage=SimpleNamespace(input_tokens=1234, output_tokens=56))


def test_renders_show_each_part_highlighted_from_its_best_view():
    m = noisy_house(rotate_deg=20)
    labels = parts.segment_builtin(m)
    images = labeller.render_parts(m, labels, size=256, overview_size=320)
    assert set(images) == {"overview-front-left", "overview-back-right"} | {f"part-{k}" for k in labels.ids}
    assert images["overview-front-left"].size == (320, 320)
    for k in labels.ids:
        px = np.asarray(images[f"part-{k}"])
        px = px.astype(int)
        magenta = (px[..., 0] > px[..., 1] + 60) & (px[..., 2] > px[..., 1] + 40)
        # The highlight is visible, and is a fraction of the picture, not all of it.
        assert 0.002 < magenta.mean() < 0.6, (k, magenta.mean())
    # The window on the +z face is best seen from the front, the roof from the top.
    roof = next(k for k in labels.ids if np.asarray(m.face_normals)[labels.labels == k, 1].mean() > 0.9)
    assert labeller._visible_area(m, np.flatnonzero(labels.labels == roof), labeller.VIEWS["top"]) > \
        labeller._visible_area(m, np.flatnonzero(labels.labels == roof), labeller.VIEWS["front"])


def test_request_shape_and_structured_output():
    m = house()
    labels, truth = house_parts(m)
    client = FakeClient(truth)
    naming = labeller.name_parts(m, labels, client=client, model="test-model", hint="a cottage")
    assert len(client.requests) == 1
    req = client.requests[0]
    assert req["model"] == "test-model"
    assert req["fallbacks"] == "default" and labeller.FALLBACK_BETA in req["betas"]
    assert req["output_config"]["format"]["type"] == "json_schema"
    assert req["output_config"]["format"]["schema"]["properties"]["parts"]["items"]["properties"]["kind"]["enum"] == list(labeller.KINDS)
    content = req["messages"][0]["content"]
    images = [b for b in content if b["type"] == "image"]
    assert len(images) == 2 + len(labels.ids)
    assert all(b["source"]["media_type"] == "image/png" for b in images)
    assert "a cottage" in content[0]["text"]
    assert "window" in req["system"] and "wall" in req["system"]
    assert naming.object == "a small house" and naming.model == "test-model"
    assert sorted(p.id for p in naming.parts) == labels.ids
    assert naming.input_tokens == 1234 and not naming.cached


def test_naming_feeds_the_builders(tmp_path):
    m = house()
    labels, truth = house_parts(m)
    client = FakeClient(truth)
    t = MeshTarget.from_mesh(m, max_studs=30, n_samples=30000, color_samples=30000, seed=3)
    cfg = PipelineConfig(max_studs=30, label_parts=True, labeller_client=client,
                         labeller_debug_dir=str(tmp_path / "pics"))
    pipe = SemanticLegoPipeline(cfg)
    result = pipe.run_on_target(t)
    assert pipe.last_naming is not None and t.metadata["naming"].object == "a small house"
    assert sorted(b["name"] for b in t.metadata["forced_openings"]) == ["door", "window", "window"]
    assert any(r.label == RegionLabel.PLANAR_ROOF for r in result.regions)
    kinds = sorted(p.part.shape for p in result.placements if p.part.shape in ("window", "door"))
    assert kinds == ["door", "window", "window"]
    assert stability.check(result.placements).ok
    assert (tmp_path / "pics" / "overview-front-left.png").exists()
    assert Image.open(tmp_path / "pics" / "overview-front-left.png").size == (768, 768)


def test_apply_keeps_sidecar_names_and_drops_other():
    labels = parts.PartLabels(np.array([0, 0, 1, 1, 2, 2]), names={0: "roof"})
    naming = labeller.Naming("a car", [
        labeller.PartName(0, "bonnet", "wall", 0.9),
        labeller.PartName(1, "left wheel", "other", 0.8),
        labeller.PartName(2, "windscreen", "window", 0.3),
    ])
    labeller.apply(labels, naming, min_confidence=0.5)
    assert labels.names == {0: "roof"}
    assert labels.descriptions == {0: "bonnet", 1: "left wheel"}
    labeller.apply(labels, naming, keep_existing=False)
    assert labels.names == {0: "wall", 2: "window"}


def test_second_run_is_cached_and_parts_are_chunked(cache_dir, monkeypatch):
    m = house()
    labels, truth = house_parts(m)
    monkeypatch.setattr(labeller, "PARTS_PER_REQUEST", 3)
    client = FakeClient(truth)
    first = labeller.name_parts(m, labels, client=client)
    assert len(client.requests) == -(-len(labels.ids) // 3)
    assert all(len([b for b in r["messages"][0]["content"] if b["type"] == "image"]) <= 5
               for r in client.requests)
    assert len(cache_dir.glob("*.json") and list(cache_dir.glob("*.json"))) == 1
    again = FakeClient(truth)
    second = labeller.name_parts(m, labels, client=again)
    assert again.requests == [] and second.cached
    assert {p.id: (p.kind, p.name) for p in second.parts} == {p.id: (p.kind, p.name) for p in first.parts}
    # Different labels, different cache entry.
    other = parts.PartLabels(np.roll(labels.labels, 1), colours=labels.colours)
    labeller.name_parts(m, other, client=again)
    assert len(again.requests) >= 1


def test_failures_are_a_warning_in_the_pipeline_and_an_error_alone():
    m = house()
    labels, truth = house_parts(m)
    for client, message in ((FakeClient(truth, fail=ConnectionError("no network")), "no network"),
                            (FakeClient(truth, stop_reason="refusal"), "declined"),
                            (FakeClient(truth, text="not json"), "invalid JSON")):
        with pytest.raises(labeller.LabellerError, match=message):
            labeller.name_parts(m, labels, client=client, cache=False)
    t = MeshTarget.from_mesh(m, max_studs=30, n_samples=30000, color_samples=30000, seed=3)
    cfg = PipelineConfig(max_studs=30, label_parts=True,
                         labeller_client=FakeClient(truth, fail=ConnectionError("no network")))
    with pytest.warns(RuntimeWarning, match="part naming skipped"):
        result = SemanticLegoPipeline(cfg).run_on_target(t)
    # Unnamed, the parts still build: the builtin segmenter's own classification stands.
    assert result.placements and "naming" not in t.metadata


def test_missing_sdk_is_a_clear_error(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def no_anthropic(name, *a, **k):
        if name == "anthropic":
            raise ImportError("No module named 'anthropic'")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", no_anthropic)
    with pytest.raises(labeller.LabellerError, match=r"everything2lego\[vlm\]"):
        labeller.make_client()


def test_export_round_trips_through_parts(tmp_path):
    m = house()
    labels, truth = house_parts(m)
    labeller.apply(labels, labeller.name_parts(m, labels, client=FakeClient(truth)))
    out = tmp_path / "house.parts.glb"
    names = labeller.export_labelled(m, labels, out)
    assert set(names.values()) == {"wall", "roof", "window", "door", "foundation"}
    assert json.loads(out.with_suffix(".json").read_text()) == names
    reloaded = parts.load_part_labels(out, m)
    assert len(reloaded.ids) == len(labels.ids)
    assert sorted(set(reloaded.names.values())) == sorted(set(names.values()))
    t = MeshTarget.from_mesh(m, max_studs=30, n_samples=30000, color_samples=30000, seed=3)
    parts.regions_from_labels(t, reloaded)
    assert sorted(b["name"] for b in t.metadata["forced_openings"]) == ["door", "window", "window"]


def test_cli_label_parts_and_semantic_flag(tmp_path, monkeypatch):
    m = house()
    _, truth = house_parts(m)
    mesh_path = tmp_path / "house.glb"
    m.export(mesh_path)
    monkeypatch.setattr(labeller, "make_client", lambda: FakeClient(truth))
    out = tmp_path / "named.parts.glb"
    result = CliRunner().invoke(app, ["label-parts", str(mesh_path), "-o", str(out), "--hint", "a shed"])
    assert result.exit_code == 0, result.output
    assert "a small house" in result.output and "-> window" in result.output
    assert out.exists() and out.with_suffix(".json").exists()

    result = CliRunner().invoke(app, ["semantic", str(mesh_path), "--label-parts", "--max-studs", "30",
                                      "--n-samples", "30000", "--out", str(tmp_path / "out")])
    assert result.exit_code == 0, result.output
    assert "object: a small house" in result.output and "(cached)" not in result.output
    assert (tmp_path / "out" / "parts" / "part-0.png").exists()

    monkeypatch.setattr(labeller, "make_client", lambda: FakeClient(truth, fail=RuntimeError("no key")))
    result = CliRunner().invoke(app, ["label-parts", str(mesh_path), "-o", str(out), "--no-cache"])
    assert result.exit_code == 2 and "no key" in result.output
