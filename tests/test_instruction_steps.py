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


def varied_tower(layers, parts_per_layer, side=14, seed=3):
    """Like `tower`, but each layer is a random packing of 1x1, 1x2, 2x2 and
    1x4 plates in four colours on a `side` x `side` footprint."""
    import random
    rnd = random.Random(seed)
    parts = [next(p for p in catalog() if p.shape == "plate" and p.footprint == fp)
             for fp in ((1, 1), (2, 1), (2, 2), (4, 1))]
    colors = [color(n) for n in ("Red", "Blue", "Black", "Orange")]
    out = []
    for y in range(layers):
        occupied, n, tries = set(), 0, 0
        while n < parts_per_layer and tries < 2000:
            tries += 1
            part = rnd.choice(parts)
            w, d = part.footprint if rnd.random() < 0.5 else part.footprint[::-1]
            x, z = rnd.randrange(side - w + 1), rnd.randrange(side - d + 1)
            cells = {(x + i, z + j) for i in range(w) for j in range(d)}
            if cells & occupied:
                continue
            occupied |= cells
            out.append(pack.Placement(part, rnd.choice(colors), x, z, y, w, d,
                                      0 if (w, d) == part.footprint else 90))
            n += 1
    return out


def rests_on_earlier_steps(steps):
    """Every part of a step stands on the ground or on parts of earlier steps
    (true for solid slabs, where the cell below is always filled)."""
    placed = set()
    for step in steps:
        for p in step:
            if p.y_plate > 0:
                below = {(p.x_stud + i, p.z_stud + j, p.y_plate - 1)
                         for i in range(p.w_world) for j in range(p.d_world)}
                if not below <= placed:
                    return False
        for p in step:
            placed.update((p.x_stud + i, p.z_stud + j, p.y_plate + k)
                          for i in range(p.w_world) for j in range(p.d_world)
                          for k in range(p.part.height_units))
    return True


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

    # Layers are never merged partially: a step either holds whole layers
    # or is one half of a single layer bigger than the budget.
    for a, b in zip(steps, steps[1:]):
        shared = {p.y_plate for p in a} & {p.y_plate for p in b}
        assert not shared or ({p.y_plate for p in a} == {p.y_plate for p in b} == shared)

    # Budget respected: the 45-part layer is bigger than the budget, so it
    # is cut into two spatial halves (left/right by stud x) that stand alone.
    assert [len(s) for s in steps] == [29, 12, 22, 23, 9]
    left, right = steps[2], steps[3]
    assert {p.y_plate for p in left} == {p.y_plate for p in right} == {4}
    assert max(p.x_stud for p in left) <= min(p.x_stud for p in right)


def test_plan_steps_default_budget_scales_with_part_count():
    assert instructions.default_budget(100) == instructions.DEFAULT_STEP_BUDGET
    assert instructions.default_budget(1890) == 63
    placements = tower(54, 20)   # 1080 parts
    steps = instructions.plan_steps(placements)
    assert len(steps) <= instructions.TARGET_STEPS
    assert all(len(s) <= instructions.default_budget(len(placements)) for s in steps)
    assert len(steps) == len(instructions.plan_steps(placements, instructions.default_budget(1080)))


def test_plan_steps_splits_oversized_layers_and_stays_bottom_up():
    placements = tower(3, 100)
    steps = instructions.plan_steps(placements, budget=60)
    assert [len(s) for s in steps] == [50, 50] * 3
    flat = [p for s in steps for p in s]
    assert [p.y_plate for p in flat] == sorted(p.y_plate for p in flat)
    for a, b in zip(steps[::2], steps[1::2]):
        assert {p.y_plate for p in a} == {p.y_plate for p in b}
        # Front/back halves: the 11-wide x 10-deep slab is cut along stud x.
        assert max(p.x_stud for p in a) <= min(p.x_stud for p in b)
    assert rests_on_earlier_steps(steps)
    # A layer far over the budget keeps halving until every piece fits.
    quarters = instructions.plan_steps(tower(1, 100), budget=30)
    assert sorted(len(s) for s in quarters) == [25, 25, 25, 25]


def test_plan_steps_merges_greedily_up_to_budget():
    placements = tower(30, 4)
    steps = instructions.plan_steps(placements, budget=20)
    assert [len(s) for s in steps] == [20] * 6
    assert all(len({p.y_plate for p in s}) == 5 for s in steps)


def test_plan_steps_budget_one_is_one_part_per_step():
    placements = tower(7, 3)
    steps = instructions.plan_steps(placements, budget=1)
    assert len(steps) == 21
    assert all(len(s) == 1 for s in steps)
    assert rests_on_earlier_steps(steps)


def test_pdf_has_far_fewer_pages_than_two_per_layer(tmp_path):
    layers = 30
    placements = tower(layers, 6)   # 180 parts
    out = instructions.write_pdf(placements, tmp_path / "steps.pdf")
    data = out.read_bytes()
    assert data.startswith(b"%PDF")
    pages = count_pages(data)
    steps = instructions.plan_steps(placements)
    assert 1 < len(steps) < layers
    # Cover + per step: a render page with the parts callout and one
    # combined plan, which fits beside the callout for a model this small.
    assert pages < 2 * layers
    assert pages == 1 + len(steps)
    text = pdf_text(data)
    assert f"{len(placements)} parts in {len(steps)} steps ({layers} plate layers)" in text
    assert f"Step 1 of {len(steps)}" in text
    assert "Placement plan, plate levels 0-5" in text
    assert f"Placement plan, plate levels {layers - 6}-{layers - 1}" in text
    assert "Step 1: parts" in text and f"Step {len(steps)}: parts" in text


def test_pdf_pages_per_step_under_two_for_a_tall_varied_model(tmp_path):
    # Layers of ~35 parts of mixed footprints (a 54-layer model of this kind
    # has ~1,900 parts): one page per step, as the plan fits beside the
    # callout, instead of a render page plus a plan page per level.
    placements = varied_tower(30, 35)
    steps = instructions.plan_steps(placements)
    out = instructions.write_pdf(placements, tmp_path / "tall.pdf", steps=steps)
    data = out.read_bytes()
    pages = count_pages(data)
    assert pages < 2 * len(steps)
    assert pages == 1 + len(steps)
    text = pdf_text(data)
    assert all(f"Step {k}: parts" in text for k in range(1, len(steps) + 1))
    assert "Step 1: placement plan" not in text    # no separate plan pages


def test_multi_level_step_lists_levels_and_draws_legend(tmp_path):
    # Two plates stacked in one step overlap in top view: both are listed
    # with their level and the plan carries the level legend.
    placements = [placement(0, 0, 0, "Red"), placement(0, 0, 1, "Blue"),
                  placement(2, 0, 0, "Red")]
    steps = [placements]
    out = instructions.write_pdf(placements, tmp_path / "stack.pdf", steps=steps)
    text = pdf_text(out.read_bytes())
    assert re.search(r"Step 1 of 1 .* layers 0-1", text)
    assert "Placement plan, plate levels 0-1" in text
    assert "Levels:" in text and "corner digit = plate level" in text
    assert "1.  1 x Plate 1 x 1 (3024) - Blue  [L1]" in text
    assert "2.  2 x Plate 1 x 1 (3024) - Red  [L0]" in text


def test_split_layer_steps_are_labelled_as_parts(tmp_path):
    placements = tower(1, 50)
    steps = instructions.plan_steps(placements, budget=30)
    assert len(steps) == 2
    out = instructions.write_pdf(placements, tmp_path / "split.pdf", steps=steps)
    text = pdf_text(out.read_bytes())
    assert re.search(r"Step 1 of 2 .* layer 0 \(part 1 of 2\)", text)
    assert re.search(r"Step 2 of 2 .* layer 0 \(part 2 of 2\)", text)


def test_inventory_page_only_when_callout_overflows(tmp_path):
    n_colors = len(palette())
    small = [placement(x, 0, 0, palette()[x].name) for x in range(4)]
    out = instructions.write_pdf(small, tmp_path / "small.pdf")
    data = out.read_bytes()
    assert count_pages(data) == 2
    assert pdf_text(data).count("Step 1: parts") == 1

    # One SKU per palette colour: too many rows for a callout box, so the
    # full inventory follows the plan on its own page(s).
    many = [pack.Placement(plate_1x1(), idx, idx, 0, 0, 1, 1, 0) for idx in range(n_colors)]
    out = instructions.write_pdf(many, tmp_path / "many.pdf", steps=[many])
    data = out.read_bytes()
    assert count_pages(data) >= 3
    text = pdf_text(data)
    assert "Step 1: placement plan" in text
    assert f"{n_colors}.  1 x Plate 1 x 1 (3024) - {palette()[n_colors - 1].name}" in text


def test_cover_has_parts_summary_table(tmp_path):
    names = [c.name for c in palette()[:20]]
    placements = [placement(x, 0, 0, name) for x, name in enumerate(names)]
    placements += [placement(0, 1, 0, "Red")] * 5
    out = instructions.write_pdf(placements, tmp_path / "cover.pdf")
    text = pdf_text(out.read_bytes())
    assert "Parts summary" in text
    assert "20 distinct parts (SKUs), top 15 listed" in text
    assert "Total parts" in text and str(len(placements)) in text
    assert "other parts (5 SKUs)" in text
    assert "Qty" in text and "Colour" in text


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
