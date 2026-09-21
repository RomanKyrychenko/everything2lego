"""Step-by-step PDF build instructions.

Approach:
- Sort placements bottom-up into build steps (one step per plate-layer).
- Render each step as an isometric painter's-algorithm image with the current
  model in full color and the "just added" pieces outlined in bright yellow.
- Assemble the sequence of PNGs into a PDF using reportlab.
"""

from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Iterable
from collections import Counter

from PIL import Image, ImageDraw
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

from . import bricks
from .colors import palette
from .pack import Placement


COS30 = math.cos(math.pi / 6)
SIN30 = math.sin(math.pi / 6)


def _project(x: float, y: float, z: float, scale: float,
             ox: float, oy: float) -> tuple[float, float]:
    u = (x - z) * COS30 * scale + ox
    v = (x + z) * SIN30 * scale - y * scale + oy
    return u, v


def _shade(rgb: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    return tuple(max(0, min(255, int(c * factor))) for c in rgb)  # type: ignore[return-value]


# Order of world-low directions cycled by 90° Y-rotations, matching pack.py
_LOW_DIR_CYCLE = ("+x", "-z", "-x", "+z")


def _slope_world_low(p: Placement) -> str | None:
    """World direction ("+x" / "-z" / "-x" / "+z" / "+x+z" / "+x-z" /
    "-x+z" / "-x-z") that this slope's low side points to, accounting for
    the part's default slope_dir + placement rotation. None if not a slope."""
    part = p.part
    if part.shape not in ("slope", "slope_inv", "slope_corner"):
        return None
    default = part.slope_dir or ("+x+z" if part.shape == "slope_corner" else "+x")
    rot_steps = (p.rotation_deg // 90) % 4
    if part.shape == "slope_corner":
        corner_cycle = ("+x+z", "+x-z", "-x-z", "-x+z")
        if default not in corner_cycle:
            default = "+x+z"
        idx = corner_cycle.index(default)
        return corner_cycle[(idx + rot_steps) % 4]
    if default not in _LOW_DIR_CYCLE:
        default = "+x"
    idx = _LOW_DIR_CYCLE.index(default)
    return _LOW_DIR_CYCLE[(idx + rot_steps) % 4]


def _draw_brick(draw: ImageDraw.ImageDraw, p: Placement, rgb: tuple[int, int, int],
                scale: float, ox: float, oy: float,
                *, is_new: bool = False) -> None:
    S = bricks.MM_PER_STUD
    P = bricks.MM_PER_PLATE
    x0, x1 = p.x_stud * S, (p.x_stud + p.w_world) * S
    z0, z1 = p.z_stud * S, (p.z_stud + p.d_world) * S
    y0, y1 = p.y_plate * P, (p.y_plate + p.part.height_units) * P

    outline = (255, 210, 40) if is_new else (0, 0, 0)
    outline_w = 3 if is_new else 1
    shape = p.part.shape
    low_dir = _slope_world_low(p)

    if shape in ("slope", "slope_inv", "slope_corner") and low_dir:
        # Slope: two top corners are low (at y=y0) and two/three are high.
        # Inverted: same idea but on the bottom face — flat top, tilted bottom.
        # We render the *silhouette* correctly by picking which corners of the
        # box's top / bottom faces to collapse.
        # Top face corners keyed by (x*, z*) location:
        top_y = {"E": y1, "F": y1, "G": y1, "H": y1}
        bot_y = {"A": y0, "B": y0, "C": y0, "D": y0}
        if shape == "slope":
            # Collapse the two TOP corners on the low-direction side to y0.
            if low_dir == "+x": top_y["F"] = top_y["G"] = y0
            elif low_dir == "-x": top_y["E"] = top_y["H"] = y0
            elif low_dir == "+z": top_y["G"] = top_y["H"] = y0
            elif low_dir == "-z": top_y["E"] = top_y["F"] = y0
        elif shape == "slope_inv":
            # Bottom face is the tilted one — the *low* corners on the bottom
            # face rise up to y1 on the opposite side from the slope's low-dir.
            if low_dir == "+x": bot_y["A"] = bot_y["D"] = y1
            elif low_dir == "-x": bot_y["B"] = bot_y["C"] = y1
            elif low_dir == "+z": bot_y["A"] = bot_y["B"] = y1
            elif low_dir == "-z": bot_y["C"] = bot_y["D"] = y1
        elif shape == "slope_corner":
            # Corner: one top corner drops to y0, the diagonally-opposite stays high.
            if low_dir == "+x+z": top_y["G"] = y0
            elif low_dir == "+x-z": top_y["F"] = y0
            elif low_dir == "-x-z": top_y["E"] = y0
            elif low_dir == "-x+z": top_y["H"] = y0

        corners = {
            "A": (x0, bot_y["A"], z0), "B": (x1, bot_y["B"], z0),
            "C": (x1, bot_y["C"], z1), "D": (x0, bot_y["D"], z1),
            "E": (x0, top_y["E"], z0), "F": (x1, top_y["F"], z0),
            "G": (x1, top_y["G"], z1), "H": (x0, top_y["H"], z1),
        }
        pt = {k: _project(*v, scale=scale, ox=ox, oy=oy) for k, v in corners.items()}
        # Draw side faces + top slanted surface.
        draw.polygon([pt["B"], pt["C"], pt["G"], pt["F"]],
                       fill=_shade(rgb, 0.65), outline=outline)
        draw.polygon([pt["D"], pt["C"], pt["G"], pt["H"]],
                       fill=_shade(rgb, 0.85), outline=outline)
        draw.polygon([pt["E"], pt["F"], pt["G"], pt["H"]],
                       fill=rgb, outline=outline)
        # Studs only on top corners that are still at y1 (visible top surface).
        if shape != "tile":
            stud_r = 2.4 * scale
            for dx in range(p.w_world):
                for dz in range(p.d_world):
                    sx = (p.x_stud + dx + 0.5) * S
                    sz = (p.z_stud + dz + 0.5) * S
                    # Where is the top surface at this stud?
                    # Bilinear on the four top-corner Y values.
                    fx = (sx - x0) / max(1e-6, x1 - x0)
                    fz = (sz - z0) / max(1e-6, z1 - z0)
                    y_top = ((1 - fx) * (1 - fz) * top_y["E"]
                              + fx * (1 - fz) * top_y["F"]
                              + fx * fz * top_y["G"]
                              + (1 - fx) * fz * top_y["H"])
                    # Skip studs whose top surface is below the brick's top —
                    # they'd sit on the sloped face, not a flat plate.
                    if y_top < y1 - 0.1:
                        continue
                    sy = y_top + 1.6
                    u, v = _project(sx, sy, sz, scale, ox, oy)
                    draw.ellipse([u - stud_r, v - stud_r * 0.55,
                                    u + stud_r, v + stud_r * 0.55],
                                 fill=_shade(rgb, 1.1), outline=(0, 0, 0))
        return

    # Default: full axis-aligned box (bricks, plates, tiles).
    corners = {
        "A": (x0, y0, z0), "B": (x1, y0, z0), "C": (x1, y0, z1), "D": (x0, y0, z1),
        "E": (x0, y1, z0), "F": (x1, y1, z0), "G": (x1, y1, z1), "H": (x0, y1, z1),
    }
    pt = {k: _project(*v, scale=scale, ox=ox, oy=oy) for k, v in corners.items()}

    draw.polygon([pt["B"], pt["C"], pt["G"], pt["F"]], fill=_shade(rgb, 0.65), outline=outline)
    draw.polygon([pt["D"], pt["C"], pt["G"], pt["H"]], fill=_shade(rgb, 0.85), outline=outline)
    draw.polygon([pt["E"], pt["F"], pt["G"], pt["H"]], fill=rgb, outline=outline)

    if is_new:
        for a, b in (("E", "F"), ("F", "G"), ("G", "H"), ("H", "E"),
                     ("B", "F"), ("C", "G"), ("D", "H")):
            draw.line([pt[a], pt[b]], fill=outline, width=outline_w)

    if p.part.shape == "tile":
        return
    stud_r = 2.4 * scale
    for dx in range(p.w_world):
        for dz in range(p.d_world):
            sx = (p.x_stud + dx + 0.5) * S
            sz = (p.z_stud + dz + 0.5) * S
            sy = y1 + 1.6
            u, v = _project(sx, sy, sz, scale, ox, oy)
            r = stud_r
            draw.ellipse([u - r, v - r * 0.55, u + r, v + r * 0.55],
                         fill=_shade(rgb, 1.1), outline=(0, 0, 0))


def _painter_order(placements: list[Placement]) -> list[Placement]:
    """Draw far-from-camera first. Camera is at (+X, +Y, +Z), so smaller
    (x + y + z) = further away."""
    return sorted(placements, key=lambda p: (p.y_plate, p.x_stud + p.z_stud))


def _bounds(placements: list[Placement]) -> tuple[float, float, float, float, float, float]:
    S, P = bricks.MM_PER_STUD, bricks.MM_PER_PLATE
    xs0 = [p.x_stud * S for p in placements]
    xs1 = [(p.x_stud + p.w_world) * S for p in placements]
    zs0 = [p.z_stud * S for p in placements]
    zs1 = [(p.z_stud + p.d_world) * S for p in placements]
    ys0 = [p.y_plate * P for p in placements]
    ys1 = [(p.y_plate + p.part.height_units) * P for p in placements]
    return min(xs0), max(xs1), min(ys0), max(ys1), min(zs0), max(zs1)


def _render_step(all_so_far: list[Placement], new: set[int], size: tuple[int, int],
                 view_bounds: tuple[float, float, float, float, float, float]
                 ) -> Image.Image:
    W, H = size
    img = Image.new("RGB", size, (240, 240, 240))
    draw = ImageDraw.Draw(img, "RGBA")

    x_min, x_max, y_min, y_max, z_min, z_max = view_bounds
    # Compute isometric bounding box in projected coords.
    corners_3d = [
        (x_min, y_min, z_min), (x_max, y_min, z_min),
        (x_min, y_min, z_max), (x_max, y_min, z_max),
        (x_min, y_max, z_min), (x_max, y_max, z_min),
        (x_min, y_max, z_max), (x_max, y_max, z_max),
    ]
    projected = [_project(x, y, z, scale=1.0, ox=0, oy=0) for x, y, z in corners_3d]
    us = [p[0] for p in projected]
    vs = [p[1] for p in projected]
    proj_w = max(us) - min(us)
    proj_h = max(vs) - min(vs)
    if proj_w <= 0 or proj_h <= 0:
        return img
    margin = 40
    scale = min((W - 2 * margin) / proj_w, (H - 2 * margin) / proj_h)
    ox = W / 2 - (max(us) + min(us)) / 2 * scale
    oy = H / 2 - (max(vs) + min(vs)) / 2 * scale

    pal = palette()
    for idx, p in enumerate(_painter_order(all_so_far)):
        rgb = pal[p.color_idx].rgb
        _draw_brick(draw, p, rgb, scale, ox, oy, is_new=(id(p) in new))
    return img


def write_pdf(placements: Iterable[Placement], out_path: str | Path,
              *, page_size: tuple[float, float] = letter,
              image_size: tuple[int, int] = (1400, 1000)) -> Path:
    """Write step-by-step PDF instructions grouped by plate-level."""
    placements = list(placements)
    if not placements:
        raise ValueError("No placements to write")

    view_bounds = _bounds(placements)

    # Group by y_plate; one step per layer.
    by_layer: dict[int, list[Placement]] = {}
    for p in placements:
        by_layer.setdefault(p.y_plate, []).append(p)
    layers = sorted(by_layer.keys())

    out_path = Path(out_path)
    c = canvas.Canvas(str(out_path), pagesize=page_size)
    page_w, page_h = page_size

    # Cover.
    c.setFont("Helvetica-Bold", 32)
    c.drawCentredString(page_w / 2, page_h - 2 * inch, "everything2lego")
    c.setFont("Helvetica", 14)
    c.drawCentredString(page_w / 2, page_h - 2.6 * inch,
                        f"{len(placements)} parts, {len(layers)} layers")
    c.setFont("Helvetica", 10)
    c.drawCentredString(page_w / 2, 1 * inch,
                        "Build bottom-up. Bricks added this step are highlighted in yellow.")
    c.showPage()

    running: list[Placement] = []
    for step_num, y in enumerate(layers, start=1):
        newly = by_layer[y]
        running.extend(newly)
        new_ids = {id(p) for p in newly}
        img = _render_step(running, new_ids, image_size, view_bounds)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        img_reader = _pil_to_reader(buf)

        # Page header.
        c.setFont("Helvetica-Bold", 16)
        c.drawString(0.6 * inch, page_h - 0.7 * inch,
                     f"Step {step_num} of {len(layers)} — layer {y}")
        c.setFont("Helvetica", 10)
        c.drawString(0.6 * inch, page_h - 0.95 * inch,
                     f"{len(newly)} new part(s) this step, {len(running)} total")

        # Image area.
        avail_w = page_w - 1.2 * inch
        avail_h = page_h - 2.5 * inch
        img_ratio = image_size[0] / image_size[1]
        if avail_w / avail_h > img_ratio:
            draw_h = avail_h
            draw_w = draw_h * img_ratio
        else:
            draw_w = avail_w
            draw_h = draw_w / img_ratio
        img_x = (page_w - draw_w) / 2
        img_y = 0.8 * inch
        c.drawImage(img_reader, img_x, img_y, width=draw_w, height=draw_h,
                    preserveAspectRatio=True, mask="auto")
        c.showPage()
        _write_plan(c, newly, running, y, step_num, view_bounds, page_w, page_h)

    c.save()
    return out_path


def _write_plan(c, newly, running, y, step, bounds, page_w, page_h):
    """A stud-coordinate plan and numbered inventory make hidden pieces locatable."""
    pal = palette()
    keys = sorted({(p.part.ldraw, p.color_idx) for p in newly})
    numbers = {key: i + 1 for i, key in enumerate(keys)}
    quantities = Counter((p.part.ldraw, p.color_idx) for p in newly)
    parts = {p.part.ldraw: p.part for p in newly}
    c.setFont("Helvetica-Bold", 16)
    c.drawString(42, page_h - 48, f"Step {step}: placement plan")
    c.setFont("Helvetica", 10)
    c.drawString(42, page_h - 66, f"Top view at plate level {y}. Numbers match the parts below.")
    width = (bounds[1] - bounds[0]) / bricks.MM_PER_STUD
    depth = (bounds[5] - bounds[4]) / bricks.MM_PER_STUD
    cell = min((page_w - 110) / width, 290 / depth)
    x0 = (page_w - width * cell) / 2
    top = page_h - 95
    for x in range(int(width)):
        c.setFillColorRGB(0.25, 0.25, 0.25)
        c.setFont("Helvetica", min(8, cell * 0.6))
        c.drawCentredString(x0 + (x + 0.5) * cell, top + 5, str(x))
    for z in range(int(depth)):
        c.drawRightString(x0 - 5, top - (z + 0.65) * cell, str(z))
    for p in running:
        if p.y_plate > y or p.y_plate + p.part.height_units <= y:
            continue
        is_new = p.y_plate == y
        rgb = pal[p.color_idx].rgb if is_new else (222, 226, 222)
        c.setFillColorRGB(*(v / 255 for v in rgb))
        c.setStrokeColorRGB(0.25, 0.25, 0.25)
        px = x0 + p.x_stud * cell - bounds[0] / bricks.MM_PER_STUD * cell
        pz = top - (p.z_stud + p.d_world) * cell + bounds[4] / bricks.MM_PER_STUD * cell
        c.rect(px, pz, p.w_world * cell, p.d_world * cell, fill=1)
        if is_new:
            brightness = sum(rgb) / 3
            c.setFillColorRGB(*((0, 0, 0) if brightness > 145 else (1, 1, 1)))
            c.setFont("Helvetica-Bold", min(10, cell * 0.6))
            c.drawCentredString(px + p.w_world * cell / 2, pz + p.d_world * cell / 2 - 3,
                                str(numbers[(p.part.ldraw, p.color_idx)]))
    cursor = top - depth * cell - 35
    for key in keys:
        if cursor < 55:
            c.showPage()
            cursor = page_h - 50
            c.setFillColorRGB(0, 0, 0)
            c.setFont("Helvetica-Bold", 14)
            c.drawString(42, cursor, f"Step {step}: parts continued")
            cursor -= 28
        part_id, color_idx = key
        c.setFillColorRGB(*(v / 255 for v in pal[color_idx].rgb))
        c.rect(42, cursor - 2, 10, 10, fill=1)
        c.setFillColorRGB(0, 0, 0)
        c.setFont("Helvetica", 10)
        c.drawString(62, cursor,
                     f"{numbers[key]}.  {quantities[key]} x {parts[part_id].name} ({parts[part_id].bl}) - {pal[color_idx].name}")
        cursor -= 17
    c.showPage()


def _pil_to_reader(buf: io.BytesIO):
    from reportlab.lib.utils import ImageReader
    return ImageReader(buf)
