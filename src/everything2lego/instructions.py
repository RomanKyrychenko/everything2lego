"""Step-by-step PDF build instructions.

Approach:
- Sort placements bottom-up into build steps (one step per plate-layer).
- Render each step as an isometric painter's-algorithm image with the current
  model in full color and the "just added" pieces outlined in bright yellow.
  Draw order is a topological sort of "is behind" over every pair of parts
  whose projections overlap, so tall and long parts occlude correctly.
- Assemble the sequence of PNGs into a PDF using reportlab.
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


def write_pdf(placements: Iterable[Placement], out_path: str | Path,
              *, page_size: tuple[float, float] = letter,
              image_size: tuple[int, int] = (1400, 1000),
              order: list[Item] | None = None) -> Path:
    """Write step-by-step PDF instructions grouped by plate-level.  Pass
    ``order`` (from `_painter_order`) to reuse a draw order computed once."""
    placements = list(placements)
    if not placements:
        raise ValueError("No placements to write")

    view_bounds = _bounds(placements)
    if order is None:
        order = _painter_order(placements)

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
        img = _render_step(running, new_ids, image_size, view_bounds, order=order)

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
