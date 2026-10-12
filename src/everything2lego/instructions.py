"""Step-by-step PDF build instructions.

Approach:
- Sort placements bottom-up and group consecutive plate-layers into build
  steps with a part budget (see `plan_steps`). A layer is never split across
  steps, so every step's parts rest only on parts from earlier steps.
- Render each step as an isometric painter's-algorithm image with the current
  model in full color and the "just added" pieces outlined in bright yellow.
  Draw order is a topological sort of "is behind" over every pair of parts
  whose projections overlap, so tall and long parts occlude correctly.
- Give each step one combined top-view placement plan of all its new parts
  (plate level encoded as a corner digit and a lightness ramp), tiled across
  pages when the model is too wide to keep the cells legible, plus a numbered
  part inventory in a callout box on the render page. The plan sits beside
  the callout on the render page whenever it fits legibly there.
- Assemble the pages into a PDF using reportlab, opening with a cover that
  summarises the most used parts.
"""

from __future__ import annotations

import heapq
import io
import math
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from . import bricks
from .colors import palette
from .pack import Placement, slope_world_low


COS30 = math.cos(math.pi / 6)
SIN30 = math.sin(math.pi / 6)


def _project(x: float, y: float, z: float, scale: float,
             ox: float, oy: float) -> tuple[float, float]:
    u = (x - z) * COS30 * scale + ox
    v = (x + z) * SIN30 * scale - y * scale + oy
    return u, v


def _shade(rgb: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    return tuple(max(0, min(255, int(c * factor))) for c in rgb)  # type: ignore[return-value]


def _tint(rgb: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
    """Mix ``amount`` of white into ``rgb``."""
    return tuple(int(c + (255 - c) * amount) for c in rgb)  # type: ignore[return-value]


@lru_cache(maxsize=None)
def _tones(rgb: tuple[int, int, int]) -> dict[str, tuple[int, int, int]]:
    """Four tones per colour: lit top, two shaded sides and an outline drawn
    in a dark version of the part's own colour rather than flat black."""
    return {
        "top": _tint(rgb, 0.12),
        "left": _shade(rgb, 0.82),    # +z face
        "right": _shade(rgb, 0.62),   # +x face
        "line": _shade(rgb, 0.35),
    }


STUD_RADIUS = 2.4   # mm
STUD_HEIGHT = 1.7   # mm
# A horizontal circle of radius r projects to an ellipse with these semi-axes.
_ELLIPSE_U = math.sqrt(2) * COS30
_ELLIPSE_V = math.sqrt(2) * SIN30


def _draw_stud(draw: ImageDraw.ImageDraw, sx: float, sy: float, sz: float,
               tones: dict, scale: float, ox: float, oy: float) -> None:
    """A stud as a short cylinder: shaded side, lit top, toned outline."""
    a = STUD_RADIUS * _ELLIPSE_U * scale
    b = STUD_RADIUS * _ELLIPSE_V * scale
    u, v0 = _project(sx, sy, sz, scale, ox, oy)
    v1 = v0 - STUD_HEIGHT * scale
    base = [u - a, v0 - b, u + a, v0 + b]
    draw.pieslice(base, 0, 90, fill=tones["right"])
    draw.pieslice(base, 90, 180, fill=tones["left"])
    draw.arc(base, 0, 180, fill=tones["line"])
    draw.rectangle([u - a, v1, u, v0], fill=tones["left"])
    draw.rectangle([u, v1, u + a, v0], fill=tones["right"])
    draw.line([(u - a, v1), (u - a, v0)], fill=tones["line"])
    draw.line([(u + a, v1), (u + a, v0)], fill=tones["line"])
    draw.ellipse([u - a, v1 - b, u + a, v1 + b], fill=tones["top"], outline=tones["line"])


_slope_world_low = slope_world_low


def _stud_cells(p: Placement) -> list[tuple[int, int]]:
    """Stud cells of a part, back to front."""
    cells = [(dx, dz) for dx in range(p.w_world) for dz in range(p.d_world)]
    return sorted(cells, key=lambda c: c[0] + c[1])


def _draw_brick(draw: ImageDraw.ImageDraw, p: Placement, rgb: tuple[int, int, int],
                scale: float, ox: float, oy: float,
                *, is_new: bool = False) -> None:
    S = bricks.MM_PER_STUD
    P = bricks.MM_PER_PLATE
    x0, x1 = p.x_stud * S, (p.x_stud + p.w_world) * S
    z0, z1 = p.z_stud * S, (p.z_stud + p.d_world) * S
    y0, y1 = p.y_plate * P, (p.y_plate + p.part.height_units) * P

    tones = _tones(rgb)
    outline = (255, 210, 40) if is_new else tones["line"]
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
                       fill=tones["right"], outline=outline)
        draw.polygon([pt["D"], pt["C"], pt["G"], pt["H"]],
                       fill=tones["left"], outline=outline)
        draw.polygon([pt["E"], pt["F"], pt["G"], pt["H"]],
                       fill=tones["top"], outline=outline)
        # Studs only on top corners that are still at y1 (visible top surface).
        if shape != "tile":
            for dx, dz in _stud_cells(p):
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
                _draw_stud(draw, sx, y_top, sz, tones, scale, ox, oy)
        return

    # Default: full axis-aligned box (bricks, plates, tiles).
    corners = {
        "A": (x0, y0, z0), "B": (x1, y0, z0), "C": (x1, y0, z1), "D": (x0, y0, z1),
        "E": (x0, y1, z0), "F": (x1, y1, z0), "G": (x1, y1, z1), "H": (x0, y1, z1),
    }
    pt = {k: _project(*v, scale=scale, ox=ox, oy=oy) for k, v in corners.items()}

    draw.polygon([pt["B"], pt["C"], pt["G"], pt["F"]], fill=tones["right"], outline=outline)
    draw.polygon([pt["D"], pt["C"], pt["G"], pt["H"]], fill=tones["left"], outline=outline)
    draw.polygon([pt["E"], pt["F"], pt["G"], pt["H"]], fill=tones["top"], outline=outline)

    if is_new:
        for a, b in (("E", "F"), ("F", "G"), ("G", "H"), ("H", "E"),
                     ("B", "F"), ("C", "G"), ("D", "H")):
            draw.line([pt[a], pt[b]], fill=outline, width=outline_w)

    if p.part.shape == "tile":
        return
    for dx, dz in _stud_cells(p):
        sx = (p.x_stud + dx + 0.5) * S
        sz = (p.z_stud + dz + 0.5) * S
        _draw_stud(draw, sx, y1, sz, tones, scale, ox, oy)


_SLOPES = ("slope", "slope_inv", "slope_corner")

# A draw item is a whole part (cell None) or one stud x one plate cell of a
# part, (dx, dy, dz) in the part's own grid, used when parts overlap cyclically.
Cell = tuple[int, int, int]
Item = tuple[Placement, "Cell | None"]


def _draw_cell(draw: ImageDraw.ImageDraw, p: Placement, cell: Cell,
               rgb: tuple[int, int, int], scale: float, ox: float, oy: float,
               *, is_new: bool = False) -> None:
    """One stud x one plate cell of a box part: its faces on the part's
    surface, the part's outline where it runs along them, and its stud."""
    S, P = bricks.MM_PER_STUD, bricks.MM_PER_PLATE
    w, h, d = p.w_world, p.part.height_units, p.d_world
    dx, dy, dz = cell
    x0, y0, z0 = (p.x_stud + dx) * S, (p.y_plate + dy) * P, (p.z_stud + dz) * S
    x1, y1, z1 = x0 + S, y0 + P, z0 + S
    tones = _tones(rgb)
    line = (255, 210, 40) if is_new else tones["line"]
    width = 3 if is_new else 1

    def pt(x, y, z):
        return _project(x, y, z, scale, ox, oy)

    # Each visible face with its corners and, per edge, whether that edge
    # runs along the part's outline (both coordinates it holds fixed lie
    # on the part's own faces).
    faces = []
    if dx == w - 1:
        faces.append(("right", [pt(x1, y0, z0), pt(x1, y0, z1), pt(x1, y1, z1), pt(x1, y1, z0)],
                      (dy == 0, dz == d - 1, dy == h - 1, dz == 0)))
    if dz == d - 1:
        faces.append(("left", [pt(x0, y0, z1), pt(x1, y0, z1), pt(x1, y1, z1), pt(x0, y1, z1)],
                      (dy == 0, dx == w - 1, dy == h - 1, dx == 0)))
    if dy == h - 1:
        faces.append(("top", [pt(x0, y1, z0), pt(x1, y1, z0), pt(x1, y1, z1), pt(x0, y1, z1)],
                      (dz == 0, dx == w - 1, dz == d - 1, dx == 0)))
    for tone, quad, _ in faces:
        draw.polygon(quad, fill=tones[tone])
    for _, quad, outline in faces:
        for k, on_outline in enumerate(outline):
            if on_outline:
                draw.line([quad[k], quad[(k + 1) % 4]], fill=line, width=width)
    if dy == h - 1 and p.part.shape != "tile":
        _draw_stud(draw, x0 + S / 2, y1, z0 + S / 2, tones, scale, ox, oy)


def _boxes(items: list[Item]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per item, in mm as (x0, x1, y0, y1, z0, z1): the body, the slab its
    studs occupy (inset from the body's edges, one stud height tall), and
    whether it shows studs at all."""
    S, P = bricks.MM_PER_STUD, bricks.MM_PER_PLATE
    rows, has_studs = [], []
    for p, cell in items:
        x0, z0, y0 = p.x_stud * S, p.z_stud * S, p.y_plate * P
        studded = p.part.shape != "tile"
        if cell is None:
            rows.append((x0, x0 + p.w_world * S, y0, y0 + p.part.height_units * P,
                         z0, z0 + p.d_world * S))
        else:
            dx, dy, dz = cell
            rows.append((x0 + dx * S, x0 + (dx + 1) * S, y0 + dy * P, y0 + (dy + 1) * P,
                         z0 + dz * S, z0 + (dz + 1) * S))
            studded = studded and dy == p.part.height_units - 1
        has_studs.append(studded)
    body = np.array(rows, dtype=float).reshape(-1, 6)
    inset = S / 2 - STUD_RADIUS
    studs = body + [inset, -inset, 0.0, 0.0, inset, -inset]
    studs[:, 2] = body[:, 3]
    studs[:, 3] = body[:, 3] + STUD_HEIGHT
    return body, studs, np.array(has_studs, dtype=bool)


def _hexagon(b: np.ndarray) -> np.ndarray:
    """The projection of a box along (1, 1, 1) is the hexagon bounded by the
    ranges of x - z, x - y and y - z; returned per box as (lo, hi) of each.
    Two projections overlap iff all three ranges do."""
    x0, x1, y0, y1, z0, z1 = b.T
    return np.stack([x0 - z1, x1 - z0, x0 - y1, x1 - y0, y0 - z1, y1 - z0], axis=1)


def _overlap(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Whether hexagons a and b (rows of `_hexagon`) overlap."""
    hit = np.ones(len(a), dtype=bool)
    for k in (0, 2, 4):
        hit &= (a[:, k] < b[:, k + 1] - eps) & (b[:, k] < a[:, k + 1] - eps)
    return hit


def _behind(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Whether box a lies entirely on the far side of box b along some axis.
    The camera looks down (-1, -1, -1), so "far" means lower x, y or z."""
    return ((a[:, 1] <= b[:, 0] + eps) | (a[:, 3] <= b[:, 2] + eps)
            | (a[:, 5] <= b[:, 4] + eps))


def _pair_order(body: np.ndarray, studs: np.ndarray, has_studs: np.ndarray,
                i: np.ndarray, j: np.ndarray,
                hexes: tuple[np.ndarray, np.ndarray] | None = None
                ) -> tuple[np.ndarray, np.ndarray]:
    """For each pair, (i must be drawn first, j must be drawn first).

    An item is its body plus its stud slab.  Where two bodies overlap on
    screen, they decide; otherwise a stud slab against the other item
    does.  Overlapping projections of disjoint boxes always have exactly
    one far side, so each deciding pair gives an unambiguous answer.
    ``hexes`` are `_hexagon` of body and studs, if already computed.
    """
    body_hex, stud_hex = hexes if hexes is not None else (_hexagon(body), _hexagon(studs))
    i_first = np.zeros(len(i), dtype=bool)
    j_first = np.zeros(len(i), dtype=bool)
    for a, ah, b, bh, valid in (
            (body, body_hex, body, body_hex, None),
            (studs, stud_hex, body, body_hex, has_studs[i]),
            (body, body_hex, studs, stud_hex, has_studs[j]),
            (studs, stud_hex, studs, stud_hex, has_studs[i] & has_studs[j])):
        hit = _overlap(ah[i], bh[j])
        if valid is not None:
            hit &= valid
        ai, bj = a[i], b[j]
        fwd, bwd = _behind(ai, bj), _behind(bj, ai)
        open_ = hit & ~(i_first | j_first)
        i_first |= open_ & fwd & ~bwd
        j_first |= open_ & bwd & ~fwd
    return i_first, j_first


def _edges(body: np.ndarray, studs: np.ndarray, has_studs: np.ndarray
           ) -> tuple[np.ndarray, np.ndarray]:
    """Every "draw src before dst" constraint, found by bucketing items on
    two of their hexagon ranges and testing pairs that share a bucket."""
    n = len(body)
    grown = body.copy()
    grown[:, 3] = np.where(has_studs, studs[:, 3], body[:, 3])
    grown_hex = _hexagon(grown)
    hexes = (_hexagon(body), _hexagon(studs))
    cell = 4 * bricks.MM_PER_STUD
    lo = np.floor(grown_hex[:, [0, 2]] / cell).astype(int)
    hi = np.floor(grown_hex[:, [1, 3]] / cell).astype(int)
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for k in range(n):
        for ga in range(lo[k, 0], hi[k, 0] + 1):
            for gb in range(lo[k, 1], hi[k, 1] + 1):
                buckets[(ga, gb)].append(k)

    found = []
    for (ga, gb), members in buckets.items():
        if len(members) < 2:
            continue
        idx = np.asarray(members)
        ti, tj = np.triu_indices(len(idx), 1)
        i, j = idx[ti], idx[tj]
        # Test each pair once: in the first bucket the two items share.
        first = ((np.maximum(lo[i, 0], lo[j, 0]) == ga)
                 & (np.maximum(lo[i, 1], lo[j, 1]) == gb))
        i, j = i[first], j[first]
        near = _overlap(grown_hex[i], grown_hex[j])
        i, j = i[near], j[near]
        if not len(i):
            continue
        i_first, j_first = _pair_order(body, studs, has_studs, i, j, hexes)
        found.append(np.concatenate([i[i_first], j[j_first]]) * n
                     + np.concatenate([j[i_first], i[j_first]]))
    edges = np.sort(np.concatenate(found)) if found else np.zeros(0, dtype=int)
    return edges // n, edges % n


def _toposort(body: np.ndarray, src: np.ndarray, dst: np.ndarray) -> list[int]:
    """Kahn's algorithm.  Ties, and any cycle left over, fall back to the
    old layer-major key (bottom first, then x + z)."""
    n = len(body)
    starts = np.searchsorted(src, np.arange(n + 1))
    succ = dst.tolist()
    indeg = np.bincount(dst, minlength=n).tolist()
    tiebreak = np.lexsort((np.arange(n), body[:, 0] + body[:, 4], body[:, 2])).tolist()
    rank = [0] * n
    for r, k in enumerate(tiebreak):
        rank[k] = r
    ready = [(rank[k], k) for k in range(n) if indeg[k] == 0]
    heapq.heapify(ready)
    done = [False] * n
    order: list[int] = []
    pending = iter(tiebreak)
    while len(order) < n:
        if not ready:
            k = next(k for k in pending if not done[k])
            heapq.heappush(ready, (rank[k], k))
        _, k = heapq.heappop(ready)
        if done[k]:
            continue
        done[k] = True
        order.append(k)
        for d in succ[starts[k]:starts[k + 1]]:
            indeg[d] -= 1
            if indeg[d] == 0 and not done[d]:
                heapq.heappush(ready, (rank[d], d))
    return order


def _cyclic(n: int, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Items that sit on a cycle of the "drawn before" relation."""
    graph = csr_matrix((np.ones(len(src), dtype=np.int8), (src, dst)), shape=(n, n))
    _, labels = connected_components(graph, directed=True, connection="strong")
    return np.bincount(labels)[labels] > 1


def _visible_cells(p: Placement) -> list[Cell]:
    """Cells that carry a face the camera can see (top, +x or +z side)."""
    w, h, d = p.w_world, p.part.height_units, p.d_world
    return [(dx, dy, dz) for dx in range(w) for dy in range(h) for dz in range(d)
            if dx == w - 1 or dy == h - 1 or dz == d - 1]


def _painter_order(placements: Iterable[Placement]) -> list[Item]:
    """Back-to-front draw items that are correct for parts of any size.

    Sorting by a single key per part fails as soon as parts differ in
    height or length: a plate one layer up but behind a tall brick sorts
    after it and is painted over it.  Instead, for every pair whose
    projections overlap, decide which one is behind along a separating axis
    (`_pair_order`) and topologically sort that relation.

    Interlocking parts can overlap cyclically (A hides B hides C hides A),
    which no order of whole parts can draw.  Parts on a cycle are split
    into one stud x one plate cells and sorted again.  Cells of a common
    grid never form a cycle, because a line of sight only moves to higher
    x, y and z, so this settles after a split or two.  Slopes are kept
    whole and, if still cyclic, fall back to the tiebreak key.
    """
    items: list[Item] = [(p, None) for p in placements]
    if len(items) < 2:
        return items
    while True:
        body, studs, has_studs = _boxes(items)
        src, dst = _edges(body, studs, has_studs)
        cyclic = _cyclic(len(items), src, dst)
        split = [k for k in np.flatnonzero(cyclic)
                 if items[k][1] is None and items[k][0].part.shape not in _SLOPES]
        if not split:
            return [items[k] for k in _toposort(body, src, dst)]
        split_set = set(split)
        items = ([it for k, it in enumerate(items) if k not in split_set]
                 + [(items[k][0], c) for k in split for c in _visible_cells(items[k][0])])


def _bounds(placements: list[Placement]) -> tuple[float, float, float, float, float, float]:
    S, P = bricks.MM_PER_STUD, bricks.MM_PER_PLATE
    xs0 = [p.x_stud * S for p in placements]
    xs1 = [(p.x_stud + p.w_world) * S for p in placements]
    zs0 = [p.z_stud * S for p in placements]
    zs1 = [(p.z_stud + p.d_world) * S for p in placements]
    ys0 = [p.y_plate * P for p in placements]
    ys1 = [(p.y_plate + p.part.height_units) * P for p in placements]
    return min(xs0), max(xs1), min(ys0), max(ys1), min(zs0), max(zs1)


def _draw_shadow(img: Image.Image, placements: list[Placement], ground: float,
                 scale: float, ox: float, oy: float) -> None:
    """A soft contact shadow: every footprint projected onto the ground
    plane, grown a little and blurred, so the model sits on the page."""
    S = bricks.MM_PER_STUD
    grow = 1.5
    mask = Image.new("L", img.size, 0)
    draw = ImageDraw.Draw(mask)
    for p in placements:
        x0, x1 = p.x_stud * S - grow, (p.x_stud + p.w_world) * S + grow
        z0, z1 = p.z_stud * S - grow, (p.z_stud + p.d_world) * S + grow
        draw.polygon([_project(x, ground, z, scale, ox, oy)
                      for x, z in ((x0, z0), (x1, z0), (x1, z1), (x0, z1))], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(max(1.0, 3.0 * scale)))
    mask = mask.point(lambda v: v * 45 // 100)
    img.paste((150, 150, 150), mask=mask)


def _render_step(all_so_far: list[Placement], new: set[int], size: tuple[int, int],
                 view_bounds: tuple[float, float, float, float, float, float],
                 *, order: list[Item] | None = None) -> Image.Image:
    """Render the parts placed so far.  ``order`` is the draw order of the
    whole model from `_painter_order`, computed once; the items of any
    subset of parts are still in a valid order, so steps can share it."""
    W, H = size
    img = Image.new("RGB", size, (240, 240, 240))

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

    _draw_shadow(img, all_so_far, y_min, scale, ox, oy)
    draw = ImageDraw.Draw(img, "RGBA")
    if order is None:
        order = _painter_order(all_so_far)
    else:
        present = {id(p) for p in all_so_far}
        order = [it for it in order if id(it[0]) in present]
    pal = palette()
    for p, cell in order:
        rgb = pal[p.color_idx].rgb
        if cell is None:
            _draw_brick(draw, p, rgb, scale, ox, oy, is_new=(id(p) in new))
        else:
            _draw_cell(draw, p, cell, rgb, scale, ox, oy, is_new=(id(p) in new))
    return img




DEFAULT_STEP_BUDGET = 40
"""Smallest default part budget per build step (layers are merged up to it)."""

TARGET_STEPS = 30
"""The default budget grows with the model so it builds in about this many steps."""

MIN_PLAN_CELL = 14.0
"""Smallest stud cell (in points) drawn on a placement plan before tiling."""

MAX_PLAN_CELL = 24.0
"""Largest stud cell drawn on a placement plan (bigger only wastes paper)."""

COVER_TABLE_ROWS = 15
"""How many SKUs the parts summary on the cover lists."""


def default_budget(n_parts: int) -> int:
    """Parts per step so that a model takes at most about TARGET_STEPS steps,
    never below DEFAULT_STEP_BUDGET."""
    return max(DEFAULT_STEP_BUDGET, math.ceil(max(0, int(n_parts)) / TARGET_STEPS))


def _layers(placements: list[Placement]) -> list[list[Placement]]:
    """Placements grouped by plate level, bottom-up."""
    by_layer: dict[int, list[Placement]] = {}
    for p in placements:
        by_layer.setdefault(p.y_plate, []).append(p)
    return [by_layer[y] for y in sorted(by_layer)]


def _split_layer(layer: list[Placement], budget: int) -> list[list[Placement]]:
    """A layer bigger than the budget, cut into spatial halves.

    The cut runs across the layer's longer footprint axis (left/right halves
    by stud x, or front/back halves by stud z), each half holding the same
    number of parts, and repeats while a half is still over the budget. Both
    halves rest only on the complete layers below, so they are valid steps.
    """
    if len(layer) <= budget or len(layer) < 2:
        return [layer]
    cx = [p.x_stud + p.w_world / 2 for p in layer]
    cz = [p.z_stud + p.d_world / 2 for p in layer]
    by_x = (max(cx) - min(cx)) >= (max(cz) - min(cz))
    ordered = sorted(layer, key=lambda p: ((p.x_stud + p.w_world / 2, p.z_stud) if by_x
                                           else (p.z_stud + p.d_world / 2, p.x_stud)))
    half = len(ordered) // 2
    return _split_layer(ordered[:half], budget) + _split_layer(ordered[half:], budget)


def plan_steps(placements: Iterable[Placement],
               budget: int | None = None) -> list[list[Placement]]:
    """Group placements bottom-up into build steps of roughly `budget` parts.

    Consecutive plate layers are merged into one step while the step stays
    within the budget; `budget` defaults to `default_budget` of the part count
    so a model takes at most about TARGET_STEPS steps. A layer is never
    merged partially: a layer with more parts than the budget is cut into
    spatial halves (`_split_layer`) that each become a step of their own, and
    every other step holds whole layers. Because steps are made of consecutive
    layers (or halves of one layer over complete layers), each step's parts
    only rest on parts from earlier steps.
    """
    placements = list(placements)
    if budget is None:
        budget = default_budget(len(placements))
    budget = max(1, int(budget))
    steps: list[list[Placement]] = []
    current: list[Placement] = []
    for layer in _layers(placements):
        if len(layer) > budget:
            if current:
                steps.append(current)
                current = []
            steps.extend(_split_layer(layer, budget))
            continue
        if current and len(current) + len(layer) > budget:
            steps.append(current)
            current = []
        current.extend(layer)
    if current:
        steps.append(current)
    return steps


def _step_levels(step: list[Placement]) -> list[int]:
    return sorted({p.y_plate for p in step})


def _step_labels(steps: list[list[Placement]]) -> list[str]:
    """"layer 3", "layers 3-5" or, for a split layer, "layer 3 (part 1 of 2)"."""
    shares: dict[int, list[int]] = defaultdict(list)
    for k, step in enumerate(steps):
        for y in _step_levels(step):
            shares[y].append(k)
    labels = []
    for k, step in enumerate(steps):
        levels = _step_levels(step)
        text = (f"layer {levels[0]}" if len(levels) == 1
                else f"layers {levels[0]}-{levels[-1]}")
        owners = shares[levels[0]]
        if len(levels) == 1 and len(owners) > 1:
            text += f" (part {owners.index(k) + 1} of {len(owners)})"
        labels.append(text)
    return labels


# --- page layout (points; letter is 612 x 792) --------------------------------

_MARGIN = 42               # left/right page margin
_HEADER_H = 80             # page top to the bottom of the step header
_BOTTOM = 43               # lowest y any content may reach
_GAP = 14                  # vertical gap between blocks
_CALLOUT_ROW = 12          # inventory row height in a callout box
_CALLOUT_TITLE = 18        # title line of a callout box
_CALLOUT_MAX_H = 200       # tallest callout box under the render
_CALLOUT_MAX_W = 250       # widest callout box beside a plan
_MIN_RENDER_H = 260        # the render never shrinks below this on a step page
_SIDE_PLAN_MAX_H = 320     # tallest plan drawn beside the callout on a step page

_PLAN_PAGE_TOP = 50        # baseline of the page title
_PLAN_FLOW_TOP = 95        # distance from the page top to the first plan block
_PLAN_BOTTOM_MARGIN = 55   # lowest y a plan or inventory row may reach
_PLAN_BLOCK_HEADER = 30    # sub-header + stud-number row above each grid
_PLAN_BLOCK_GAP = 30
_INVENTORY_ROW = 17

_GREY = (222, 226, 222)
_LEVEL_RAMP = 0.5          # how much white the highest level of a step mixes in


class _Inventory:
    """The numbered parts list of one step: a number per (part, colour) SKU,
    with quantities and the plate levels it is used on."""

    def __init__(self, newly: list[Placement]):
        pal = palette()
        self.keys = sorted({(p.part.ldraw, p.color_idx) for p in newly})
        self.numbers = {key: i + 1 for i, key in enumerate(self.keys)}
        self.quantities = Counter((p.part.ldraw, p.color_idx) for p in newly)
        self.parts = {p.part.ldraw: p.part for p in newly}
        self.levels: dict[tuple[str, int], set[int]] = defaultdict(set)
        for p in newly:
            self.levels[(p.part.ldraw, p.color_idx)].add(p.y_plate)
        self.multi_level = len({p.y_plate for p in newly}) > 1
        self.pal = pal

    def number(self, p: Placement) -> int:
        return self.numbers[(p.part.ldraw, p.color_idx)]

    def rgb(self, key) -> tuple[int, int, int]:
        return self.pal[key[1]].rgb

    def line(self, key, *, short: bool = False) -> str:
        part_id, color_idx = key
        part, color = self.parts[part_id], self.pal[color_idx]
        text = f"{self.numbers[key]}.  {self.quantities[key]} x {part.name} ({part.bl}) - {color.name}"
        if self.multi_level:
            lv = sorted(self.levels[key])
            text += (f"  [L{lv[0]}]" if len(lv) == 1 else f"  [L{lv[0]}-{lv[-1]}]") if short \
                else (f"  - level {lv[0]}" if len(lv) == 1
                      else "  - levels " + ", ".join(map(str, lv)))
        return text


def _fit_text(c, text: str, font: str, size: float, max_w: float) -> str:
    """`text` cut (with an ellipsis) to fit `max_w` points."""
    from reportlab.pdfbase.pdfmetrics import stringWidth
    if stringWidth(text, font, size) <= max_w:
        return text
    while text and stringWidth(text + "...", font, size) > max_w:
        text = text[:-1]
    return text.rstrip() + "..."


def _callout_rows(n_keys: int, columns: int) -> int:
    return math.ceil(n_keys / max(1, columns))


def _callout_height(n_keys: int, columns: int) -> float:
    return _CALLOUT_TITLE + _callout_rows(n_keys, columns) * _CALLOUT_ROW + 8


def _draw_callout(c, inv: _Inventory, step: int, x: float, top: float,
                  width: float, columns: int) -> float:
    """The numbered inventory in a bordered box; returns the box's bottom y."""
    height = _callout_height(len(inv.keys), columns)
    c.setStrokeColorRGB(0.6, 0.6, 0.6)
    c.setLineWidth(0.6)
    c.roundRect(x, top - height, width, height, 4, stroke=1, fill=0)
    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(x + 8, top - 13, f"Step {step}: parts")
    rows = _callout_rows(len(inv.keys), columns)
    col_w = (width - 16) / columns
    for i, key in enumerate(inv.keys):
        col, row = divmod(i, rows)
        cx = x + 8 + col * col_w
        cy = top - _CALLOUT_TITLE - (row + 1) * _CALLOUT_ROW + 2
        c.setFillColorRGB(*(v / 255 for v in inv.rgb(key)))
        c.setStrokeColorRGB(0.3, 0.3, 0.3)
        c.rect(cx, cy - 1, 8, 8, fill=1, stroke=1)
        c.setFillColorRGB(0, 0, 0)
        c.setFont("Helvetica", 7.5)
        c.drawString(cx + 12, cy, _fit_text(c, inv.line(key, short=True),
                                            "Helvetica", 7.5, col_w - 16))
    return top - height


def _level_tint(y: int, levels: list[int]) -> float:
    """White mixed into a new part's colour: lowest level of the step 0,
    highest _LEVEL_RAMP, so stacked parts of one step read as a ramp."""
    if len(levels) < 2:
        return 0.0
    return _LEVEL_RAMP * levels.index(y) / (len(levels) - 1)


def _draw_plan_tile(c, newly, running, inv: _Inventory, tile, cell, x0, top,
                    x_origin, z_origin, *, title: str | None, tiled: bool):
    """One tile of a step's combined top-view plan with its top-left corner
    at (x0 - margin, top); returns the grid's bottom y.

    Parts placed in earlier steps are grey; the step's own parts are drawn
    bottom level first in their colour, lightened towards the highest level,
    numbered like the parts list and (for a multi-level step) marked with
    their plate level in the corner. A higher part stacked on a lower one
    within the step covers it, so both are also listed with their levels.
    """
    tx0, tx1, tz0, tz1 = tile
    tile_w, tile_d = tx1 - tx0, tz1 - tz0
    new_ids = {id(p) for p in newly}
    levels = _step_levels(newly)
    header = _PLAN_BLOCK_HEADER if title is not None else _PLAN_BLOCK_HEADER - 12
    if title is not None:
        c.setFillColorRGB(0, 0, 0)
        c.setFont("Helvetica-Bold", 11)
        if tiled:
            title += f"  —  studs x {tx0}-{tx1 - 1}, z {tz0}-{tz1 - 1}"
        c.drawString(x0 - 20, top - 10, title)
    grid_top = top - header
    c.setFillColorRGB(0.25, 0.25, 0.25)
    c.setFont("Helvetica", min(8, cell * 0.6))
    for x in range(tx0, tx1):
        c.drawCentredString(x0 + (x - tx0 + 0.5) * cell, grid_top + 4, str(x))
    for z in range(tz0, tz1):
        c.drawRightString(x0 - 5, grid_top - (z - tz0 + 0.65) * cell, str(z))
    grid_bottom = grid_top - tile_d * cell
    c.saveState()
    clip = c.beginPath()
    clip.rect(x0, grid_bottom, tile_w * cell, tile_d * cell)
    c.clipPath(clip, stroke=0, fill=0)
    c.setLineWidth(0.5)

    def visible(p):
        px_s, pz_s = p.x_stud - x_origin, p.z_stud - z_origin
        return not (px_s + p.w_world <= tx0 or px_s >= tx1
                    or pz_s + p.d_world <= tz0 or pz_s >= tz1)

    def box(p):
        px = x0 + (p.x_stud - x_origin - tx0) * cell
        pz = grid_top - (p.z_stud - z_origin - tz0 + p.d_world) * cell
        return px, pz, p.w_world * cell, p.d_world * cell

    old = sorted((p for p in running if id(p) not in new_ids and visible(p)),
                 key=lambda p: p.y_plate)
    c.setFillColorRGB(*(v / 255 for v in _GREY))
    c.setStrokeColorRGB(0.55, 0.55, 0.55)
    for p in old:
        c.rect(*box(p), fill=1)
    new = sorted((p for p in newly if visible(p)),
                 key=lambda p: (p.y_plate, p.x_stud + p.z_stud))
    num_font = min(10, cell * 0.6)
    lvl_font = max(4.0, min(6, cell * 0.38))
    for p in new:
        rgb = _tint(inv.rgb((p.part.ldraw, p.color_idx)), _level_tint(p.y_plate, levels))
        c.setFillColorRGB(*(v / 255 for v in rgb))
        c.setStrokeColorRGB(0.2, 0.2, 0.2)
        px, pz, pw, pd = box(p)
        c.rect(px, pz, pw, pd, fill=1)
        ink = (0, 0, 0) if sum(rgb) / 3 > 145 else (1, 1, 1)
        c.setFillColorRGB(*ink)
        vx0, vx1 = max(px, x0), min(px + pw, x0 + tile_w * cell)
        vz0, vz1 = max(pz, grid_bottom), min(pz + pd, grid_top)
        c.setFont("Helvetica-Bold", num_font)
        c.drawCentredString((vx0 + vx1) / 2, (vz0 + vz1) / 2 - num_font * 0.35,
                            str(inv.number(p)))
        if inv.multi_level:
            c.setFont("Helvetica", lvl_font)
            c.drawString(vx0 + 1.2, vz1 - lvl_font - 0.6, str(p.y_plate))
    c.restoreState()
    return grid_bottom


def _draw_level_legend(c, levels: list[int], x: float, y: float) -> None:
    """Swatches of the level ramp, lowest to highest, for a multi-level step."""
    if len(levels) < 2:
        return
    base = (90, 130, 200)
    c.setFont("Helvetica", 8)
    c.setFillColorRGB(0.2, 0.2, 0.2)
    c.drawString(x, y, "Levels:")
    cx = x + 36
    for y_lvl in levels:
        rgb = _tint(base, _level_tint(y_lvl, levels))
        c.setFillColorRGB(*(v / 255 for v in rgb))
        c.setStrokeColorRGB(0.2, 0.2, 0.2)
        c.rect(cx, y - 2, 11, 11, fill=1, stroke=1)
        c.setFillColorRGB(0.2, 0.2, 0.2)
        c.setFont("Helvetica", 8)
        c.drawString(cx + 14, y, str(y_lvl))
        cx += 18 + 7 * len(str(y_lvl))
    c.drawString(cx + 4, y, "(corner digit = plate level; lower is darker)")


def _plan_tiles(width: int, depth: int, avail_w: float, avail_h: float
                ) -> tuple[float, list[tuple[int, int, int, int]]]:
    """Cell size and the (x0, x1, z0, z1) stud ranges of each plan tile.

    The plan may use the whole available area (cells capped at MAX_PLAN_CELL
    so small models do not waste paper); when that would still shrink the
    cells below MIN_PLAN_CELL the grid is cut into tiles that each fit on one
    page.
    """
    width, depth = max(1, width), max(1, depth)
    cell = min(avail_w / width, avail_h / depth, MAX_PLAN_CELL)
    if cell >= MIN_PLAN_CELL:
        return cell, [(0, width, 0, depth)]
    cell = MIN_PLAN_CELL
    tile_w = max(1, int(avail_w // cell))
    tile_d = max(1, int(avail_h // cell))
    tiles = []
    for z0 in range(0, depth, tile_d):
        for x0 in range(0, width, tile_w):
            tiles.append((x0, min(width, x0 + tile_w), z0, min(depth, z0 + tile_d)))
    return cell, tiles


def _grid_extent(bounds) -> tuple[int, int, float, float]:
    """(width, depth) in studs and the (x, z) stud origin of the model."""
    width = int(round((bounds[1] - bounds[0]) / bricks.MM_PER_STUD))
    depth = int(round((bounds[5] - bounds[4]) / bricks.MM_PER_STUD))
    return width, depth, bounds[0] / bricks.MM_PER_STUD, bounds[4] / bricks.MM_PER_STUD


def _write_cover(c, placements, steps, page_w, page_h) -> None:
    """Title, totals and a parts summary table of the most used SKUs."""
    pal = palette()
    n_layers = len({p.y_plate for p in placements})
    c.setFont("Helvetica-Bold", 32)
    c.drawCentredString(page_w / 2, page_h - 1.6 * inch, "everything2lego")
    c.setFont("Helvetica", 14)
    c.drawCentredString(page_w / 2, page_h - 2.1 * inch,
                        f"{len(placements)} parts in {len(steps)} steps "
                        f"({n_layers} plate layers)")

    counts = Counter((p.part.ldraw, p.color_idx) for p in placements)
    parts = {p.part.ldraw: p.part for p in placements}
    top = counts.most_common(COVER_TABLE_ROWS)
    y = page_h - 2.9 * inch
    c.setFont("Helvetica-Bold", 13)
    c.drawString(_MARGIN + 30, y, "Parts summary")
    c.setFont("Helvetica", 9)
    c.setFillColorRGB(0.35, 0.35, 0.35)
    c.drawRightString(page_w - _MARGIN - 30, y,
                      f"{len(counts)} distinct parts (SKUs), top {len(top)} listed")
    y -= 20
    col_qty, col_part, col_color = _MARGIN + 30, _MARGIN + 80, page_w / 2 + 60
    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica-Bold", 10)
    c.drawRightString(col_qty + 30, y, "Qty")
    c.drawString(col_part, y, "Part")
    c.drawString(col_color, y, "Colour")
    y -= 6
    c.setStrokeColorRGB(0.4, 0.4, 0.4)
    c.setLineWidth(0.8)
    c.line(_MARGIN + 30, y, page_w - _MARGIN - 30, y)
    y -= 15
    c.setFont("Helvetica", 10)
    for (part_id, color_idx), n in top:
        part, color = parts[part_id], pal[color_idx]
        c.setFillColorRGB(0, 0, 0)
        c.drawRightString(col_qty + 30, y, str(n))
        c.drawString(col_part, y, _fit_text(c, f"{part.name} ({part.bl})", "Helvetica", 10,
                                            col_color - col_part - 12))
        c.setFillColorRGB(*(v / 255 for v in color.rgb))
        c.setStrokeColorRGB(0.3, 0.3, 0.3)
        c.rect(col_color, y - 1, 10, 10, fill=1, stroke=1)
        c.setFillColorRGB(0, 0, 0)
        c.drawString(col_color + 16, y, color.name)
        y -= 16
    rest = len(placements) - sum(n for _, n in top)
    if rest > 0:
        c.setFillColorRGB(0.35, 0.35, 0.35)
        c.drawRightString(col_qty + 30, y, str(rest))
        c.drawString(col_part, y, f"other parts ({len(counts) - len(top)} SKUs)")
        y -= 16
    y -= 4
    c.setStrokeColorRGB(0.4, 0.4, 0.4)
    c.line(_MARGIN + 30, y, page_w - _MARGIN - 30, y)
    y -= 15
    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica-Bold", 10)
    c.drawRightString(col_qty + 30, y, str(len(placements)))
    c.drawString(col_part, y, "Total parts")

    c.setFont("Helvetica", 10)
    c.drawCentredString(page_w / 2, 1 * inch,
                        "Build bottom-up. Bricks added this step are highlighted in yellow.")
    c.showPage()


def write_pdf(placements: Iterable[Placement], out_path: str | Path,
              *, page_size: tuple[float, float] = letter,
              image_size: tuple[int, int] = (1400, 1000),
              order: list[Item] | None = None,
              steps: list[list[Placement]] | None = None,
              budget: int | None = None) -> Path:
    """Write step-by-step PDF instructions.

    `order` (from `_painter_order`) reuses a draw order computed once;
    `steps` (from `plan_steps`) lets callers share one grouping between the
    PDF and the LDraw file.  Both are computed here when omitted.

    Each step gets a render page with its numbered parts in a callout box
    and one combined top-view plan of all its new parts. The plan is drawn
    beside the callout on the render page when it fits legibly there, and on
    its own page (tiled, for wide models) otherwise; the inventory moves to
    a page of its own only when it overflows the callout.
    """
    placements = list(placements)
    if not placements:
        raise ValueError("No placements to write")
    if steps is None:
        steps = plan_steps(placements, budget)
    steps = [list(s) for s in steps if s]
    if not steps:
        raise ValueError("No steps to write")

    view_bounds = _bounds(placements)
    if order is None:
        order = _painter_order(placements)
    labels = _step_labels(steps)

    out_path = Path(out_path)
    c = canvas.Canvas(str(out_path), pagesize=page_size)
    page_w, page_h = page_size
    _write_cover(c, placements, steps, page_w, page_h)

    width, depth, x_origin, z_origin = _grid_extent(view_bounds)
    content_w = page_w - 2 * _MARGIN
    img_ratio = image_size[0] / image_size[1]

    running: list[Placement] = []
    for step_num, newly in enumerate(steps, start=1):
        running.extend(newly)
        new_ids = {id(p) for p in newly}
        img = _render_step(running, new_ids, image_size, view_bounds, order=order)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        img_reader = _pil_to_reader(buf)
        inv = _Inventory(newly)
        levels = _step_levels(newly)

        c.setFillColorRGB(0, 0, 0)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(_MARGIN, page_h - 50, f"Step {step_num} of {len(steps)} — {labels[step_num - 1]}")
        c.setFont("Helvetica", 10)
        c.drawString(_MARGIN, page_h - 68,
                     f"{len(newly)} new part(s) this step, {len(running)} total")
        top = page_h - _HEADER_H
        band_h = top - _BOTTOM

        # Layout 1: plan (one tile) and the callout side by side under the render.
        side_plan_w = content_w - _CALLOUT_MAX_W - _GAP
        side_plan_h = min(_SIDE_PLAN_MAX_H, band_h - _MIN_RENDER_H - _GAP)
        side_cell = min((side_plan_w - 40) / max(1, width),
                        (side_plan_h - _PLAN_BLOCK_HEADER) / max(1, depth), MAX_PLAN_CELL)
        side_callout_h = _callout_height(len(inv.keys), 1)
        side_ok = (side_cell >= MIN_PLAN_CELL and side_plan_h > 0
                   and side_callout_h <= side_plan_h)
        if side_ok:
            lower_h = max(_PLAN_BLOCK_HEADER + depth * side_cell + 14, side_callout_h)
            render_h = band_h - lower_h - _GAP
            draw_w = min(content_w, render_h * img_ratio)
            draw_h = draw_w / img_ratio
            c.drawImage(img_reader, (page_w - draw_w) / 2, top - draw_h,
                        width=draw_w, height=draw_h, preserveAspectRatio=True, mask="auto")
            lower_top = top - render_h - _GAP
            grid_x = _MARGIN + 30 + (side_plan_w - 40 - width * side_cell) / 2
            _draw_plan_tile(c, newly, running, inv, (0, width, 0, depth), side_cell,
                            grid_x, lower_top, x_origin, z_origin,
                            title=_plan_title(levels), tiled=False)
            _draw_level_legend(c, levels, _MARGIN + 10, lower_top - lower_h + 2)
            _draw_callout(c, inv, step_num, page_w - _MARGIN - _CALLOUT_MAX_W, lower_top,
                          _CALLOUT_MAX_W, 1)
            c.showPage()
            continue

        # Layout 2: callout under the render (two columns), plan on its own page(s).
        callout_cols = 1 if len(inv.keys) <= 6 else 2
        callout_h = _callout_height(len(inv.keys), callout_cols)
        callout_fits = callout_h <= min(_CALLOUT_MAX_H, band_h - _MIN_RENDER_H - _GAP)
        render_h = band_h - (callout_h + _GAP if callout_fits else 0)
        draw_w = min(content_w, render_h * img_ratio)
        draw_h = draw_w / img_ratio
        c.drawImage(img_reader, (page_w - draw_w) / 2, top - draw_h,
                    width=draw_w, height=draw_h, preserveAspectRatio=True, mask="auto")
        if callout_fits:
            _draw_callout(c, inv, step_num, _MARGIN, top - draw_h - _GAP, content_w, callout_cols)
        c.showPage()
        _write_plan(c, newly, running, levels, step_num, view_bounds, page_w, page_h,
                    inventory=not callout_fits)

    c.save()
    return out_path


def _plan_title(levels: list[int]) -> str:
    return (f"Placement plan, plate level {levels[0]}" if len(levels) == 1
            else f"Placement plan, plate levels {levels[0]}-{levels[-1]}")


def _write_plan(c, newly, running, levels, step, bounds, page_w, page_h,
                *, inventory: bool = True):
    """A stud-coordinate plan makes hidden pieces locatable.

    One combined top-view plan shows every part of the step with its plate
    level encoded (see `_draw_plan_tile`); a model too wide for legible cells
    is cut into tiles whose stud ranges are named in their headers, flowing
    down the page and onto further pages. With `inventory` the numbered parts
    list follows the last plan (when the callout on the render page could not
    hold it), continuing on extra pages when a step has many SKUs.
    """
    if isinstance(levels, int):
        levels = [levels]
    inv = _Inventory(newly)
    width, depth, x_origin, z_origin = _grid_extent(bounds)
    avail_h = page_h - _PLAN_FLOW_TOP - _PLAN_BLOCK_HEADER - _PLAN_BOTTOM_MARGIN
    cell, tiles = _plan_tiles(width, depth, page_w - 110, avail_h)
    tiled = len(tiles) > 1

    subtitle = "Top view of this step's parts. Grey = already placed; numbers match the parts list."

    def page_title():
        c.setFillColorRGB(0, 0, 0)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(_MARGIN, page_h - 48, f"Step {step}: placement plan")
        c.setFont("Helvetica", 10)
        c.drawString(_MARGIN, page_h - 66, subtitle)
        _draw_level_legend(c, levels, _MARGIN, page_h - 82)
        return page_h - _PLAN_FLOW_TOP

    cursor = page_title()
    for tile in tiles:
        block_h = _PLAN_BLOCK_HEADER + (tile[3] - tile[2]) * cell
        if cursor < page_h - _PLAN_FLOW_TOP and cursor - block_h < _PLAN_BOTTOM_MARGIN:
            c.showPage()
            cursor = page_title()
        x0 = (page_w - (tile[1] - tile[0]) * cell) / 2
        cursor = _draw_plan_tile(c, newly, running, inv, tile, cell, x0, cursor,
                                 x_origin, z_origin, title=_plan_title(levels),
                                 tiled=tiled) - _PLAN_BLOCK_GAP

    if not inventory:
        c.showPage()
        return
    # Inventory: under the last plan when at least a few rows fit there.
    if cursor - 28 - 3 * _INVENTORY_ROW < _PLAN_BOTTOM_MARGIN:
        c.showPage()
        cursor = page_h - _PLAN_PAGE_TOP
    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(_MARGIN, cursor, f"Step {step}: parts")
    cursor -= 28
    for key in inv.keys:
        if cursor < _PLAN_BOTTOM_MARGIN:
            c.showPage()
            cursor = page_h - _PLAN_PAGE_TOP
            c.setFillColorRGB(0, 0, 0)
            c.setFont("Helvetica-Bold", 14)
            c.drawString(_MARGIN, cursor, f"Step {step}: parts continued")
            cursor -= 28
        c.setFillColorRGB(*(v / 255 for v in inv.rgb(key)))
        c.rect(_MARGIN, cursor - 2, 10, 10, fill=1)
        c.setFillColorRGB(0, 0, 0)
        c.setFont("Helvetica", 10)
        c.drawString(_MARGIN + 20, cursor, inv.line(key))
        cursor -= _INVENTORY_ROW
    c.showPage()


def _pil_to_reader(buf: io.BytesIO):
    from reportlab.lib.utils import ImageReader
    return ImageReader(buf)
