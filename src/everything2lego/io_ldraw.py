"""LDraw .ldr writer.

LDraw type-1 line format:
    1 <colour> x y z  a b c  d e f  g h i  part.dat

- Y axis points DOWN.
- 1 stud = 20 LDU horizontally; 1 plate = 8 LDU vertically; 1 brick = 24 LDU.
- Part origin is at the CENTER of the part's footprint, at the TOP of the body
  (where studs meet the body). So a brick sitting on the ground has origin at
  y = -(brick_height_plates) * 8 in LDraw world coords.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

from .bricks import GLASS, GLASS_OFFSET, LDU_PER_PLATE, LDU_PER_STUD
from .colors import palette
from .pack import Placement
from .snot import SnotChild, render_child


# 3x3 rotation matrices (row-major, LDraw convention, right-handed Y-down).
_ROT: dict[int, tuple[int, ...]] = {
    0:   (1, 0, 0,  0, 1, 0,  0, 0, 1),
    90:  (0, 0, 1,  0, 1, 0, -1, 0, 0),
    180: (-1, 0, 0, 0, 1, 0,  0, 0, -1),
    270: (0, 0, -1, 0, 1, 0,  1, 0, 0),
}


def _fmt_num(n: int | float) -> str:
    if isinstance(n, float) and n.is_integer():
        n = int(n)
    return str(n)


def placement_to_line(p: Placement, children: Iterable[SnotChild] = ()) -> str:
    pal = palette()
    ldraw_color = pal[p.color_idx].code

    x = (2 * p.x_stud + p.w_world - 1) * (LDU_PER_STUD // 2)
    z = (2 * p.z_stud + p.d_world - 1) * (LDU_PER_STUD // 2)
    y = -(p.y_plate + p.part.height_units) * LDU_PER_PLATE

    m = _ROT[(p.rotation_deg + p.part.yaw) % 360]
    ox, oy, oz = p.part.origin
    if ox or oy or oz:                     # the origin is off the footprint's centre / body's top
        x -= m[0] * ox + m[2] * oz
        z -= m[6] * ox + m[8] * oz
        y -= oy
    matrix = " ".join(_fmt_num(v) for v in m)
    at = f"{_fmt_num(x)} {_fmt_num(y)} {_fmt_num(z)} {matrix}"
    line = f"1 {ldraw_color} {at} {p.part.ldraw}"
    if p.part.companion:                   # a wheel's tyre: black, as tyres are
        line += f"\n1 0 {at} {p.part.companion}"
    glass = GLASS.get(p.part.ldraw)
    if glass is not None and p.glass is not None:
        # the frame's glass, clipped in with the frame's own transform
        # (shifted where the part pair needs it)
        dx, dy, dz = GLASS_OFFSET.get(p.part.ldraw, (0, 0, 0))
        gx = x + m[0] * dx + m[1] * dy + m[2] * dz
        gy = y + m[3] * dx + m[4] * dy + m[5] * dz
        gz = z + m[6] * dx + m[7] * dy + m[8] * dz
        line += f"\n1 {p.glass} {_fmt_num(gx)} {_fmt_num(gy)} {_fmt_num(gz)} {matrix} {glass}"
    for ch in children:
        line += "\n" + render_child(m, int(x), int(y), int(z), ch)
    return line


def write_ldr(placements: Iterable[Placement], out_path: str | Path,
              *, model_name: str = "everything2lego model",
              steps: Iterable[Iterable[Placement]] | None = None,
              subassemblies: Iterable[Iterable[Placement]] | None = None,
              decorations: Mapping[Placement, Iterable[SnotChild]] | None = None) -> Path:
    """Write an LDraw model with one `0 STEP` marker closing each build step.

    `steps` is the build-step grouping (a list of lists of placements, as
    returned by `instructions.plan_steps`); it lets the .ldr follow the same
    steps as the PDF. When omitted the default grouping is computed here.
    Parts inside a step are written bottom-up so viewers show a natural
    build order.

    `subassemblies` turns the output into an MPD (multi-part document):
    the main model references one submodel per group, and each submodel's
    parts live in its own `0 FILE` section with its own build steps.  Real
    sets look like this; a monolithic LDraw doesn't.
    """
    placements = list(placements)
    out_path = Path(out_path)
    header = [
        f"0 {model_name}",
        "0 Name: model.ldr",
        "0 Author: everything2lego",
        "0 !LICENSE Redistributable under CCAL version 2.0",
        "",
    ]
    if subassemblies is not None:
        groups = [list(g) for g in subassemblies if g]
        lines: list[str] = ["0 FILE model.ldr"] + header
        identity = "1 0 0 0 1 0 0 0 1"
        for i, _ in enumerate(groups):
            lines.append(f"1 16 0 0 0 {identity} sub_{i:02d}.ldr")
            lines.append("0 STEP")
        for i, group in enumerate(groups):
            lines.append("")
            lines.append(f"0 FILE sub_{i:02d}.ldr")
            lines.append(f"0 Sub-assembly {i:02d}")
            lines.append(f"0 Name: sub_{i:02d}.ldr")
            lines.append("")
            if steps is None:
                from .instructions import plan_steps
                sub_steps = plan_steps(group)
            else:
                keep = set(id(p) for p in group)
                sub_steps = [[p for p in s if id(p) in keep] for s in steps]
            for step in sub_steps:
                step = sorted(step, key=lambda q: (q.y_plate, q.z_stud, q.x_stud))
                if not step:
                    continue
                lines.extend(placement_to_line(p, (decorations or {}).get(p, ())) for p in step)
                lines.append("0 STEP")
        out_path.write_text("\n".join(lines) + "\n")
        return out_path
    if steps is None:
        from .instructions import plan_steps
        steps = plan_steps(placements)
    lines = list(header)
    for step in steps:
        step = sorted(step, key=lambda q: (q.y_plate, q.z_stud, q.x_stud))
        if not step:
            continue
        lines.extend(placement_to_line(p) for p in step)
        lines.append("0 STEP")
    out_path.write_text("\n".join(lines) + "\n")
    return out_path
