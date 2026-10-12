"""Build-step grouping shared by the PDF instructions and the LDraw file."""

import base64
import re
import zlib

import pytest

from everything2lego import instructions, io_ldraw, pack
from everything2lego.bricks import catalog
from everything2lego.colors import palette


def color(name):
    return next(i for i, c in enumerate(palette()) if c.name == name)


def plate_1x1():
    return next(p for p in catalog() if p.shape == "plate" and p.footprint == (1, 1))


def placement(x, z, y, color_name="Red"):
    return pack.Placement(plate_1x1(), color(color_name), x, z, y, 1, 1, 0)


def tower(layers, parts_per_layer):
    """`layers` plate layers of a square-ish slab, `parts_per_layer` 1x1 plates each."""
    out = []
    side = int(parts_per_layer ** 0.5) + 1
    for y in range(layers):
        for i in range(parts_per_layer):
            out.append(placement(i % side, i // side, y))
    return out


def count_pages(pdf_bytes: bytes) -> int:
    return len(re.findall(rb"/Type\s*/Page(?![s/])", pdf_bytes))


def pdf_text(pdf_bytes: bytes) -> str:
    """Strings drawn on any page.  reportlab Flate-compresses each content
    stream, optionally after ASCII85-encoding it."""
    out = []
    for chunk in pdf_bytes.split(b"endstream")[:-1]:
        head, sep, raw = chunk.rpartition(b"stream\n")
        if not sep or b"/Image" in head[-400:]:
            continue        # a step render, megabytes of pixels with no text
        raw = raw.strip()
        content = None
        for decode in (zlib.decompress,
                       lambda b: zlib.decompress(base64.a85decode(b, adobe=True))):
            try:
                content = decode(raw)
                break
            except Exception:
                continue
        if content is None:
            content = raw
        out.extend(s.decode("latin1").replace("\\(", "(").replace("\\)", ")")
                   for s in re.findall(rb"\(((?:[^()\\]|\\.)*)\)\s*Tj", content))
    return "\n".join(out)


def test_plan_steps_respects_budget_and_keeps_layers_whole():
    # Layers of varying size: 5, 12, 12, 12, 45, 3, 3, 3 parts.
    sizes = [5, 12, 12, 12, 45, 3, 3, 3]
    placements = []
    for y, n in enumerate(sizes):
        placements.extend(placement(i % 8, i // 8, y) for i in range(n))

    steps = instructions.plan_steps(placements, budget=30)

    # Every placement appears exactly once, in bottom-up order.
    flat = [p for s in steps for p in s]
    assert sorted(map(id, flat)) == sorted(map(id, placements))
    assert [p.y_plate for p in flat] == sorted(p.y_plate for p in flat)

    # Layers are never split across steps.
    for a, b in zip(steps, steps[1:]):
        assert {p.y_plate for p in a}.isdisjoint({p.y_plate for p in b})

    # Budget respected except for a layer bigger than the budget, which
    # stands alone.
    for s in steps:
        assert len(s) <= 30 or len({p.y_plate for p in s}) == 1
    assert [len(s) for s in steps] == [29, 12, 45, 9]


def test_plan_steps_merges_greedily_up_to_budget():
    placements = tower(30, 4)
    steps = instructions.plan_steps(placements, budget=20)
    assert [len(s) for s in steps] == [20] * 6
    assert all(len({p.y_plate for p in s}) == 5 for s in steps)


def test_plan_steps_budget_one_is_one_layer_per_step():
    placements = tower(7, 3)
    steps = instructions.plan_steps(placements, budget=1)
    assert len(steps) == 7
    assert all(len(s) == 3 for s in steps)


def test_pdf_has_far_fewer_pages_than_two_per_layer(tmp_path):
    layers = 30
    placements = tower(layers, 6)   # 180 parts
    out = instructions.write_pdf(placements, tmp_path / "steps.pdf")
    data = out.read_bytes()
    assert data.startswith(b"%PDF")
    pages = count_pages(data)
    steps = instructions.plan_steps(placements)
    assert 1 < len(steps) < layers
    # Cover + per step: one isometric page and the plan/inventory pages (one
    # plan per plate level with new parts; small plans share a page).
    assert pages < 2 * layers
    assert pages <= 1 + sum(1 + len({p.y_plate for p in s}) for s in steps)
    text = pdf_text(data)
    assert f"{len(placements)} parts in {len(steps)} steps ({layers} plate layers)" in text
    assert f"Step 1 of {len(steps)}" in text
    assert "Plate level 0" in text and f"Plate level {layers - 1}" in text


def test_pdf_accepts_precomputed_steps(tmp_path):
    placements = tower(6, 2)
    steps = instructions.plan_steps(placements, budget=4)
    assert len(steps) == 3
    out = instructions.write_pdf(placements, tmp_path / "given.pdf", steps=steps)
    text = pdf_text(out.read_bytes())
    assert "Step 3 of 3" in text
    assert "Step 4 of" not in text


def test_wide_model_plan_is_tiled(tmp_path):
    # 120 studs wide: 14pt minimum cells cannot fit on one letter page.
    placements = [placement(x, 0, 0) for x in range(120)]
    out = instructions.write_pdf(placements, tmp_path / "wide.pdf")
    data = out.read_bytes()
    cell, tiles = instructions._plan_tiles(120, 1, 612 - 110, 600)
    assert cell == instructions.MIN_PLAN_CELL
    assert len(tiles) > 1
    text = pdf_text(data)
    assert "studs x 0-" in text
    assert f"-{119}, z 0-0" in text
    assert count_pages(data) >= 3


def test_plan_cell_never_below_minimum():
    for width, depth in ((5, 5), (40, 40), (200, 30), (30, 300)):
        cell, tiles = instructions._plan_tiles(width, depth, 502, 600)
        assert instructions.MIN_PLAN_CELL <= cell <= instructions.MAX_PLAN_CELL
        for x0, x1, z0, z1 in tiles:
            assert (x1 - x0) * cell <= 502 and (z1 - z0) * cell <= 600
        covered = {(x, z) for x0, x1, z0, z1 in tiles
                   for x in range(x0, x1) for z in range(z0, z1)}
        assert len(covered) == width * depth


def test_ldr_has_one_step_marker_per_step(tmp_path):
    placements = tower(30, 6)
    steps = instructions.plan_steps(placements)
    out = io_ldraw.write_ldr(placements, tmp_path / "model.ldr", steps=steps)
    text = out.read_text()
    assert text.count("0 STEP") == len(steps)
    # All parts present, bottom-up.
    body = [l for l in text.splitlines() if l.startswith("1 ")]
    assert len(body) == len(placements)
    ys = [-int(l.split()[3]) for l in body]
    assert ys == sorted(ys)


def test_ldr_default_grouping_matches_plan_steps(tmp_path):
    placements = tower(30, 6)
    out = io_ldraw.write_ldr(placements, tmp_path / "model.ldr")
    assert out.read_text().count("0 STEP") == len(instructions.plan_steps(placements))


def test_write_pdf_rejects_empty():
    with pytest.raises(ValueError):
        instructions.write_pdf([], "/nonexistent/x.pdf")
