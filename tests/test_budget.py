"""The part-budget chooser: monotonic, respects the budget, orders its knobs."""

from pathlib import Path

import pytest
from PIL import Image, ImageDraw
from typer.testing import CliRunner

from everything2lego import backends, budget, pipeline, voxelize
from everything2lego.cli import app
from everything2lego.colors import palette

VIEWS = Path(__file__).resolve().parents[1] / "examples" / "views"
CARVE_16 = budget.Measurement(16, False, 632, 1)
# Measured on examples/views with pipeline.assemble defaults.
CARVE_ACTUAL = {16: 632, 24: 1601, 32: 3550}


def fake_measure(parts=600, colors=6):
    calls = []

    def measure(max_studs, hollow):
        calls.append((max_studs, hollow))
        return budget.Measurement(max_studs, hollow, parts, colors)
    measure.calls = calls
    return measure


def test_prediction_matches_the_carve_example_within_25_percent():
    for studs, actual in CARVE_ACTUAL.items():
        predicted = budget.predict(CARVE_16, studs, False)
        assert abs(predicted - actual) / actual <= 0.25, (studs, predicted, actual)


def test_prediction_is_monotonic_in_every_knob():
    m = budget.Measurement(16, False, 1000, 10)
    solid = [budget.predict(m, s, False) for s in budget.STUD_CANDIDATES]
    assert solid == sorted(solid)
    for s in budget.STUD_CANDIDATES:
        assert budget.predict(m, s, True) <= budget.predict(m, s, False)
        colours = [budget.predict(m, s, True, k) for k in (10, 8, 5, 2)]
        assert colours == sorted(colours, reverse=True)
    # Hollowing saves about 30% at 80 studs, as in the README's watermill.
    assert 0.65 <= budget.predict(m, 80, True) / budget.predict(m, 80, False) <= 0.75


def test_choice_is_monotonic_in_the_budget():
    previous = None
    for max_parts in (100, 200, 500, 1000, 2000, 5000, 10_000, 50_000, 200_000):
        choice = budget.choose(max_parts, fake_measure())
        assert choice.fits
        assert choice.predicted <= max_parts
        if previous is not None:
            assert choice.max_studs >= previous.max_studs
            if choice.max_studs == previous.max_studs:
                # More budget never asks for more aggressive hollowing / fewer colours.
                assert (choice.hollow, -choice.color_clusters) <= (previous.hollow, -previous.color_clusters)
        previous = choice
    assert choice.max_studs == 128


def test_knob_order_is_resolution_then_hollow_then_colours():
    measure = fake_measure()
    m = measure(16, False)
    solid_32 = budget.predict(m, 32, False)
    hollow_32 = budget.predict(m, 32, True)
    assert hollow_32 < solid_32
    choice = budget.choose((solid_32 + hollow_32) // 2, measure, max_studs_cap=32)
    assert (choice.max_studs, choice.hollow, choice.color_clusters) == (32, True, 6)
    # Colours only once hollowing is not enough.
    few_colours = budget.predict(m, 32, True, 2)
    choice = budget.choose((hollow_32 + few_colours) // 2, measure, max_studs_cap=32)
    assert choice.max_studs == 32 and choice.hollow and choice.color_clusters < 6
    # Pinned hollowing is measured hollow and never un-hollowed.
    choice = budget.choose(10_000, fake_measure(), hollow=True)
    assert choice.hollow and choice.measurement.hollow
    choice = budget.choose(10_000, fake_measure(), hollow=False)
    assert not choice.hollow
    choice = budget.choose(10_000, fake_measure(), allow_colors=False)
    assert choice.color_clusters == 6


def test_one_cheap_measurement_and_a_cap():
    measure = fake_measure()
    choice = budget.choose(100_000, measure, max_studs_cap=40)
    assert measure.calls == [(16, False)]
    assert choice.max_studs == 40
    tiny = budget.choose(1, fake_measure())
    assert not tiny.fits and tiny.max_studs == min(budget.STUD_CANDIDATES)
    with pytest.raises(ValueError):
        budget.choose(0, fake_measure())


def test_carve_example_respects_the_budget():
    images = sorted(VIEWS.glob("*.png"))
    max_parts = 1500

    def make_grid(studs, hollow):
        g = backends.build("carve", images, max_studs=studs)
        return voxelize.hollow_grid(g) if hollow else g

    def measure(studs, hollow):
        grid = make_grid(studs, hollow)
        return budget.Measurement(studs, hollow, len(pipeline.assemble(grid).placements), 1)

    choice = budget.choose(max_parts, measure)
    assert choice.fits
    assert abs(choice.measurement.parts - CARVE_ACTUAL[16]) / CARVE_ACTUAL[16] < 0.05
    actual = len(pipeline.assemble(make_grid(choice.max_studs, choice.hollow)).placements)
    assert actual <= max_parts * 1.25
    assert abs(actual - choice.predicted) / actual <= 0.25
    # The next resolution up would not have fitted either.
    bigger = [s for s in budget.STUD_CANDIDATES if s > choice.max_studs][0]
    assert budget.predict(choice.measurement, bigger, True) > max_parts


def test_cli_max_parts_wins_over_max_studs(tmp_path):
    red = next(c for c in palette() if c.name == "Red").rgb
    for i in range(2):
        img = Image.new("RGB", (64, 64), "white")
        ImageDraw.Draw(img).rectangle((16, 8, 47, 55), fill=red)
        img.save(tmp_path / f"view_{i}.png")
    out = tmp_path / "out"
    result = CliRunner().invoke(app, ["carve", str(tmp_path), "--max-studs", "40",
                                      "--max-parts", "150", "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert "--max-parts 150 overrides --max-studs 40" in result.output
    assert "chose" in result.output
    assert (out / "model.ldr").exists()
    parts = sum(1 for line in (out / "model.ldr").read_text().splitlines() if line.startswith("1 "))
    assert parts <= 150 * 1.25
