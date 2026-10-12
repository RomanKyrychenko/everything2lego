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
from typing import Iterable

from .bricks import LDU_PER_PLATE, LDU_PER_STUD
from .colors import palette
from .pack import Placement


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


def placement_to_line(p: Placement) -> str:
    pal = palette()
    ldraw_color = pal[p.color_idx].code

    x = (2 * p.x_stud + p.w_world - 1) * (LDU_PER_STUD // 2)
    z = (2 * p.z_stud + p.d_world - 1) * (LDU_PER_STUD // 2)
    y = -(p.y_plate + p.part.height_units) * LDU_PER_PLATE

    m = _ROT[p.rotation_deg]
    matrix = " ".join(_fmt_num(v) for v in m)
    return f"1 {ldraw_color} {_fmt_num(x)} {_fmt_num(y)} {_fmt_num(z)} {matrix} {p.part.ldraw}"


def write_ldr(placements: Iterable[Placement], out_path: str | Path,
              *, model_name: str = "everything2lego model",
              steps: Iterable[Iterable[Placement]] | None = None) -> Path:
    """Write an LDraw model with one `0 STEP` marker closing each build step.

    `steps` is the build-step grouping (a list of lists of placements, as
    returned by `instructions.plan_steps`); it lets the .ldr follow the same
    steps as the PDF. When omitted the default grouping is computed here.
    Parts inside a step are written bottom-up so viewers show a natural
    build order.
    """
    placements = list(placements)
    if steps is None:
        from .instructions import plan_steps
        steps = plan_steps(placements)
    out_path = Path(out_path)
    lines: list[str] = [
        f"0 {model_name}",
        "0 Name: model.ldr",
        "0 Author: everything2lego",
        "0 !LICENSE Redistributable under CCAL version 2.0",
        "",
    ]
    for step in steps:
        step = sorted(step, key=lambda q: (q.y_plate, q.z_stud, q.x_stud))
        if not step:
            continue
        lines.extend(placement_to_line(p) for p in step)
        lines.append("0 STEP")
    out_path.write_text("\n".join(lines) + "\n")
    return out_path
