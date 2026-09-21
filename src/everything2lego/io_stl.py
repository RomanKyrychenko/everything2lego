"""Export LEGO placements as an axis-aligned box-mesh STL.

Each Placement becomes one trimesh box sized by its footprint (in mm) and
its part height (also mm). Boxes are concatenated (not booleaned — union is
slow and unnecessary since STL viewers happily render overlapping meshes).

Result is a single-body STL that shows the LEGO assembly's silhouette /
volume for use as a reference model, 3-D-print pre-check, or CAD import.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import trimesh

from . import bricks
from .colors import palette
from .pack import Placement


def _placement_box(p: Placement) -> trimesh.Trimesh:
    w_stud, d_stud = p.w_world, p.d_world
    h_plates = p.part.height_units
    # Anisotropic scale: studs → 8 mm, plates → 3.2 mm.
    sx = w_stud * bricks.MM_PER_STUD
    sz = d_stud * bricks.MM_PER_STUD
    sy = h_plates * bricks.MM_PER_PLATE
    box = trimesh.creation.box(extents=(sx, sy, sz))
    # Place: bottom-front-left corner at (x*stud, y*plate, z*stud).
    tx = (p.x_stud + w_stud / 2) * bricks.MM_PER_STUD
    ty = (p.y_plate + h_plates / 2) * bricks.MM_PER_PLATE
    tz = (p.z_stud + d_stud / 2) * bricks.MM_PER_STUD
    box.apply_translation((tx, ty, tz))
    # Colour: attach as visual so exporters that respect vertex colours can
    # keep the tint; STL itself is monochrome but this helps GLB / PLY.
    pal = palette()
    if 0 <= p.color_idx < len(pal):
        r, g, b = pal[p.color_idx].rgb
        box.visual.face_colors = np.array([r, g, b, 255], dtype=np.uint8)
    return box


def write_stl(placements: Iterable[Placement], path: str | Path) -> Path:
    """Write a merged STL of every placement to `path`. Returns the path."""
    boxes = [_placement_box(p) for p in placements]
    if not boxes:
        # Emit an empty degenerate STL rather than fail.
        mesh = trimesh.Trimesh()
    else:
        mesh = trimesh.util.concatenate(boxes)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path)
    return path
