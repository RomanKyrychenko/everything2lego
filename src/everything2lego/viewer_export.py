"""Export the packed model to a JSON payload for the three.js viewer."""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path
from typing import Iterable

from . import bricks
from .colors import palette
from .pack import Placement


_LOW_DIR_CYCLE = ("+x", "-z", "-x", "+z")
_CORNER_LOW_CYCLE = ("+x+z", "+x-z", "-x-z", "-x+z")


def _slope_world_low(p: Placement) -> str | None:
    """World low direction for a slope placement — same logic as pack.py's
    slope refinement.  Returns e.g. "+x", "-z", or "+x+z" for corner slopes;
    None if the part isn't a slope."""
    part = p.part
    if part.shape not in ("slope", "slope_inv", "slope_corner"):
        return None
    rot_steps = (p.rotation_deg // 90) % 4
    if part.shape == "slope_corner":
        default = part.slope_dir or "+x+z"
        if default not in _CORNER_LOW_CYCLE:
            default = "+x+z"
        idx = _CORNER_LOW_CYCLE.index(default)
        return _CORNER_LOW_CYCLE[(idx + rot_steps) % 4]
    default = part.slope_dir or "+x"
    if default not in _LOW_DIR_CYCLE:
        default = "+x"
    idx = _LOW_DIR_CYCLE.index(default)
    return _LOW_DIR_CYCLE[(idx + rot_steps) % 4]


def _scene_dict(placements: Iterable[Placement]) -> dict:
    pal = palette()
    bricks_out: list[dict] = []
    for p in placements:
        bricks_out.append({
            "part": p.part.ldraw,
            "shape": p.part.shape,
            "low_dir": _slope_world_low(p),
            "rot": p.rotation_deg,
            "x": p.x_stud,
            "y": p.y_plate,
            "z": p.z_stud,
            "w": p.w_world,
            "d": p.d_world,
            "h": p.part.height_units,
            "rgb": list(pal[p.color_idx].rgb),
            "color": pal[p.color_idx].name,
        })
    return {
        "mm_per_stud": bricks.MM_PER_STUD,
        "mm_per_plate": bricks.MM_PER_PLATE,
        "bricks": bricks_out,
    }


def write_viewer(placements: Iterable[Placement], out_dir: str | Path, *, report: dict | None = None) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scene = _scene_dict(list(placements))
    scene["report"] = report or {}
    (out_dir / "scene.json").write_text(json.dumps(scene, indent=2))

    src = files("everything2lego.viewer")
    (out_dir / "main.js").write_text(src.joinpath("main.js").read_text())
    license_text = src.joinpath("THIRD_PARTY_LICENSES.txt").read_text()
    (out_dir / "THIRD_PARTY_LICENSES.txt").write_text(license_text)
    html = src.joinpath("index.html").read_text()
    payload = json.dumps(scene).replace("<", "\\u003c")
    html = html.replace("<!-- SCENE -->", f'<script id="scene-data" type="application/json">{payload}</script>')
    html = html.replace("<!-- VIEWER -->", "<!--\n" + license_text + "\n-->\n<script>" + src.joinpath("bundle.js").read_text() + "</script>")
    (out_dir / "index.html").write_text(html)

    return out_dir / "index.html"
