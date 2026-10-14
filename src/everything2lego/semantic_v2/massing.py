"""massing — walls, roofs and decks read off the SOLID voxel volume.

A segmenter working on a photo reconstruction slices one facade into a
dozen regions and one roof face into strips; recipes fed those specs build
a dozen walls and a roof of lean-tos.  The voxel volume has none of that
noise: fill the shell, and its plan at every course IS the floor plan, its
top surface IS the roof.  This module reads the architecture from there,
the way a set designer reads it off the reference photo:

  walls   every axis-aligned run of boundary cells of the plan at each
          course, accumulated into one wall per plane line
  pitches connected patches of the height map with a steady gradient
          along one axis (stepped voxels smoothed over ±2 studs)
  decks   flat patches of the height map (a flat roof, a terrace, the
          ground slab around the house)

All three come out as the specs setdesign.py already builds, so the recipes
(bonded courses, slope rows, ridge tiles, corner interlock) are shared.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..bricks import Part
from ..pack import Placement
from ..templates.base import part_by_ldraw
from .mesh_target import MeshTarget
from .openings import WallFrame, find_openings, forced_openings, reserved_cells
from .regions import PlanarRegion, RegionLabel
from .setdesign import (COURSE, MIN_PITCH_RUN, MIN_PITCH_SPAN, MIN_WALL_STUDS, PitchSpec,
                        WallSpec)

MIN_DECK_CELLS = 4
MIN_PITCH_CELLS = 8
# Height-map gradient (plates per stud) that reads as a pitched roof: a 33°
# row rises 3 plates in 2 studs, a 45° row 3 in 1; below the floor it is a
# flat deck with sampling ripple, above the ceiling a wall.
PITCH_MIN_GRADIENT = 0.75
PITCH_MAX_GRADIENT = 4.5


# ---------------------------------------------------------------------------
# Volume + height map
# ---------------------------------------------------------------------------

SOLID_MAX_GAP = 6          # plates: a hollow band this thin between solid is inside the building


def solid_volume(occ: np.ndarray) -> np.ndarray:
    """The occupied shell with its enclosed interior filled.

    Three passes, because a voxelised reconstruction is never quite
    closed: 3-D cavities first; then each plate's plan on its own (a shell
    open at the top or bottom is still a closed ring in plan); then any
    cell with solid at most SOLID_MAX_GAP plates above AND below it — the
    band where a roof meets the walls often leaks in both passes and
    would leave a storey hollow, with no floor and no plan to read the
    walls from.  The gap limit keeps an open porch open."""
    from scipy.ndimage import binary_closing, binary_fill_holes
    H = occ.shape[0]
    # A one-voxel gap in the shell (the join of a tower and a roof, a
    # window reveal) leaks every fill; close such gaps first.  Closing
    # never opens a hole, and the result is only used to decide inside.
    closed = binary_closing(occ, iterations=1) | occ
    solid = binary_fill_holes(closed)
    bracketed = np.zeros_like(solid)
    for y in range(H):
        solid[y] |= binary_fill_holes(closed[y]) | _xray_fill(closed[y])
        bracketed[y] = _xray_fill(closed[y], either=True)
    # Covered from above and standing on something below, and walled in
    # along at least one axis: the inside of a block whose shell lost a
    # whole face (a reconstruction's far side, a facade the voxeliser
    # thinned away).  A porch under a roof between two posts also counts
    # — a set would close it too.
    above_any = np.cumsum(solid[::-1], axis=0)[::-1] > 0
    below_any = np.cumsum(solid, axis=0) > 0
    solid |= above_any & below_any & bracketed
    above = np.zeros_like(solid)
    below = np.zeros_like(solid)
    for k in range(1, SOLID_MAX_GAP + 1):
        above[:H - k] |= solid[k:]
        below[k:] |= solid[:H - k]
    solid |= above & below
    return solid


def _xray_fill(sl: np.ndarray, *, either: bool = False) -> np.ndarray:
    """Cells of a plan slice that lie between occupied cells along x AND
    along z (or along either axis, with `either`).  A window or door is a
    hole in the shell big enough to leak the flood fills, but the room
    behind it is still bracketed by walls in both directions; the hole
    itself is bracketed in one only and stays open.  An L-shaped plan's
    notch is open on one side too."""
    X, Z = sl.shape
    inside_x = np.zeros_like(sl)
    for z in range(Z):
        xs = np.flatnonzero(sl[:, z])
        if len(xs):
            inside_x[xs.min():xs.max() + 1, z] = True
    inside_z = np.zeros_like(sl)
    for x in range(X):
        zs = np.flatnonzero(sl[x])
        if len(zs):
            inside_z[x, zs.min():zs.max() + 1] = True
    return (inside_x | inside_z) if either else (inside_x & inside_z)


def height_map(solid: np.ndarray) -> np.ndarray:
    """Top solid plate index per (x, z), -1 where the column is empty."""
    H = solid.shape[0]
    top = np.where(solid.any(axis=0), H - 1 - np.argmax(solid[::-1], axis=0), -1)
    return top.astype(np.int32)


def _smooth_gradient(h: np.ndarray, axis: int, reach: int = 2) -> np.ndarray:
    """Central difference of the height map over ±reach cells along `axis`.

    A neighbour counts only when it is occupied AND within a roof's reach
    in height (PITCH_MAX_GRADIENT per stud): a drop to the ground next to
    a ledge, or from a tower top to the roof below, is a cliff — the edge
    of this surface, not its slope — so the difference is taken one-sided
    from the other side there."""
    valid = h >= 0
    hf = h.astype(np.float32)
    g = np.zeros_like(hf)
    n = np.zeros_like(hf)
    for r in range(1, reach + 1):
        fwd = np.roll(hf, -r, axis=axis)
        fv = np.roll(valid, -r, axis=axis)
        bwd = np.roll(hf, r, axis=axis)
        bv = np.roll(valid, r, axis=axis)
        # roll wraps: kill the wrapped border
        idx = [slice(None)] * h.ndim
        idx[axis] = slice(-r, None)
        fv[tuple(idx)] = False
        idx[axis] = slice(0, r)
        bv[tuple(idx)] = False
        cliff = PITCH_MAX_GRADIENT * r
        fv &= np.abs(fwd - hf) <= cliff
        bv &= np.abs(bwd - hf) <= cliff
        both = valid & fv & bv
        g[both] += (fwd[both] - bwd[both]) / (2 * r)
        n[both] += 1
        only_f = valid & fv & ~bv
        g[only_f] += (fwd[only_f] - hf[only_f]) / r
        n[only_f] += 1
        only_b = valid & bv & ~fv
        g[only_b] += (hf[only_b] - bwd[only_b]) / r
        n[only_b] += 1
    out = np.zeros_like(hf)
    out[n > 0] = g[n > 0] / n[n > 0]
    return out


# ---------------------------------------------------------------------------
# Roofs and decks from the height map
# ---------------------------------------------------------------------------

@dataclass
class DeckSpec:
    """A flat top: plates in two layers whose top is `y` (plates)."""
    cells: np.ndarray          # (N, 2) int — (x, z)
    y: int
    colour: int

    @property
    def layers(self) -> list[int]:
        return [v for v in (self.y - 1, self.y - 2) if v >= 0]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    from scipy.ndimage import label
    lab, n = label(mask)
    return [np.argwhere(lab == i) for i in range(1, n + 1)]


def _colour_of(palette_idx: np.ndarray, ys: np.ndarray, xs: np.ndarray, zs: np.ndarray) -> int:
    vals = palette_idx[ys, xs, zs]
    vals = vals[vals >= 0]
    return int(Counter(vals.tolist()).most_common(1)[0][0]) if len(vals) else 0


def pitches_and_decks(solid: np.ndarray, palette_idx: np.ndarray, *, ground_y: int
                      ) -> tuple[list[PitchSpec], list[DeckSpec], np.ndarray]:
    """Classify every top cell as sloped along x, sloped along z, or flat;
    connected patches become PitchSpecs and DeckSpecs.  Also returns the
    roof SKIN: the cells of every pitch within a course of the surface, so
    the wall pass leaves the steps of the slope and the gable triangle to
    the roof recipe — and nothing more: a tower rising through a roof
    keeps its walls above it."""
    h = height_map(solid)
    valid = h >= 0
    gx = _smooth_gradient(h, 0)
    gz = _smooth_gradient(h, 1)
    ax, az = np.abs(gx), np.abs(gz)
    in_x = valid & (ax >= PITCH_MIN_GRADIENT) & (ax <= PITCH_MAX_GRADIENT)
    in_z = valid & (az >= PITCH_MIN_GRADIENT) & (az <= PITCH_MAX_GRADIENT)
    # A cell on the edge of a face sees a cliff across the other axis (the
    # drop to the wall top next to it); that is not a pitch, so a gradient
    # outside the pitch range never outvotes one inside it.
    sloped_x = in_x & (~in_z | (ax >= az))
    sloped_z = in_z & (~in_x | (az > ax))
    flat = valid & ~sloped_x & ~sloped_z & (ax < PITCH_MIN_GRADIENT) & (az < PITCH_MIN_GRADIENT)

    pitches: list[PitchSpec] = []
    owned = np.zeros_like(solid)
    stepped: list[np.ndarray] = []      # sloped patches that are not roofs: stairs, steep bays
    for axis, sloped, grad in (("x", sloped_x, gx), ("z", sloped_z, gz)):
        for sign in (1, -1):
            for cells in _components(sloped & ((grad > 0) if sign > 0 else (grad < 0))):
                if len(cells) < MIN_PITCH_CELLS:
                    continue
                along = cells[:, 0] if axis == "x" else cells[:, 1]
                across = cells[:, 1] if axis == "x" else cells[:, 0]
                lo, hi = int(along.min()), int(along.max()) + 1
                span = (int(across.min()), int(across.max()) + 1)
                run = hi - lo
                tops = h[cells[:, 0], cells[:, 1]]
                # Percentiles, not extremes: a pilaster or a hole at the
                # edge of a face must not set its ridge or eave.
                y_bottom = int(np.percentile(tops, 5)) + 1
                y_top = int(np.percentile(tops, 95)) + 1
                overall = (y_top - y_bottom) / max(1, run)
                # A roof starts above the ground and rises like a roof.  A
                # flight of steps at ground level, or a bay that climbs
                # nine plates a stud, is stepped geometry — built as decks
                # at each height instead.
                if (y_bottom <= ground_y + COURSE or overall > PITCH_MAX_GRADIENT
                        or overall < PITCH_MIN_GRADIENT * 0.5):
                    stepped.append(cells)
                    continue
                if span[1] - span[0] < MIN_PITCH_SPAN or run < MIN_PITCH_RUN:
                    continue
                eave, ridge = (float(lo), float(hi)) if sign > 0 else (float(hi), float(lo))
                colour = _colour_of(palette_idx, tops, cells[:, 0], cells[:, 1])
                pitches.append(PitchSpec(region=None, axis=axis, sign=sign, eave=eave, ridge=ridge,
                                         span=span, y_bottom=y_bottom, y_top=y_top, colour=colour,
                                         cells=cells))
                # The skin: each cell of the face from a course under its
                # surface to the surface.
                for (cx, cz), t in zip(cells, tops):
                    owned[max(0, int(t) - COURSE):int(t) + 1, cx, cz] = True

    decks: list[DeckSpec] = []

    def add_decks(cells: np.ndarray, min_cells: int) -> None:
        tops = h[cells[:, 0], cells[:, 1]]
        # One deck per height: a terrace and the roof above it are two.
        for t in np.unique(tops):
            sel = cells[tops == t]
            if len(sel) < min_cells:
                continue
            top = int(t) + 1
            y = ground_y + int(round((top - ground_y) / COURSE)) * COURSE
            y = max(ground_y + 1, y)
            decks.append(DeckSpec(cells=sel, y=y,
                                  colour=_colour_of(palette_idx, np.full(len(sel), int(t)),
                                                    sel[:, 0], sel[:, 1])))

    for cells in _components(flat):
        if len(cells) >= MIN_DECK_CELLS:
            add_decks(cells, MIN_DECK_CELLS)
    for cells in stepped:
        add_decks(cells, 2)
    return pitches, decks, owned


# ---------------------------------------------------------------------------
# Walls from the plan at every course
# ---------------------------------------------------------------------------

def _synthetic_region(frame: WallFrame, target: MeshTarget, colour_rgb: np.ndarray) -> PlanarRegion:
    """A PlanarRegion standing in for the wall so the opening finders (which
    sample the mesh surface near the region's plane within its bounds) can
    run on a wall that came from the volume, not from a segment."""
    inv = np.linalg.inv(target.mesh_to_voxel.astype(np.float64))
    if frame.axis_x:
        lo_v = np.array([frame.origin, frame.y0, frame.other + 0.5, 1.0])
        hi_v = np.array([frame.origin + frame.length, frame.y0 + frame.height, frame.other + 0.5, 1.0])
        normal = np.array([0.0, 0.0, 1.0], np.float32)
    else:
        lo_v = np.array([frame.other + 0.5, frame.y0, frame.origin, 1.0])
        hi_v = np.array([frame.other + 0.5, frame.y0 + frame.height, frame.origin + frame.length, 1.0])
        normal = np.array([1.0, 0.0, 0.0], np.float32)
    lo_mm = (inv @ lo_v)[:3]
    hi_mm = (inv @ hi_v)[:3]
    mn, mx = np.minimum(lo_mm, hi_mm).astype(np.float32), np.maximum(lo_mm, hi_mm).astype(np.float32)
    origin = ((mn + mx) / 2).astype(np.float32)
    empty = np.zeros((0, 3), np.float32)
    return PlanarRegion(label=RegionLabel.PLANAR_WALL, points=empty, normals=empty,
                        colors=colour_rgb.reshape(1, 3).astype(np.uint8), bounds=(mn, mx),
                        plane_normal=normal, plane_origin=origin)


WALL_SNAP_STUDS = 2        # a facade reconstructed from a photo wobbles this much
WALL_SNAP_WEAKER = 0.6     # a line this much weaker than a neighbour is its jitter
WALL_SNAP_SHARED = 0.7     # ...when this share of its cells sit in courses the neighbour spans


def _snap_wall_lines(masks: dict, colours: dict) -> tuple[dict, dict]:
    """Fold the jitter of a reconstructed facade onto one plane line.

    Face cells of one wall land on two or three adjacent coordinates; each
    would become its own thin, ragged wall.  Per axis, take the lines from
    the most populated down and absorb every weaker line within
    WALL_SNAP_STUDS that has under WALL_SNAP_WEAKER of its cells and lives
    in the same courses (WALL_SNAP_SHARED of its cells in courses the
    stronger line occupies).  A bay that really stands two studs proud
    keeps its own line because it is as populated as the facade next to
    it; a ledge or a plinth keeps its own because its courses are its
    own."""
    out_m: dict = {}
    out_c: dict = {}

    def alternations(a: np.ndarray, b: np.ndarray) -> int:
        """How often, going up the courses, the wall switches between
        line a and line b.  Jitter switches many times; a real step (a
        plinth, a set-back storey) switches once."""
        ca, cb = a.any(axis=0), b.any(axis=0)
        seq = [1 if (x and not y) else -1 if (y and not x) else 0 for x, y in zip(ca, cb)]
        seq = [s for s in seq if s]
        return sum(1 for s, t in zip(seq, seq[1:]) if s != t)

    for axis_x in (True, False):
        lines = {other: m for (ax, other), m in masks.items() if ax == axis_x}
        strength = {o: int(m.sum()) for o, m in lines.items()}
        absorbed: dict[int, int] = {}
        for o in sorted(lines, key=lambda k: -strength[k]):
            if o in absorbed:
                continue
            absorbed[o] = o
            courses_o = lines[o].any(axis=0)
            for d in range(1, WALL_SNAP_STUDS + 1):
                for q in (o - d, o + d):
                    if q not in lines or q in absorbed:
                        continue
                    weaker = strength[q] < WALL_SNAP_WEAKER * strength[o]
                    shared = lines[q][:, courses_o].sum() / max(1, strength[q])
                    if d == 1:
                        # Walls are at least two studs apart or they are
                        # one wall: a one-stud step is below the design's
                        # resolution (a facade is one plane; a bay stands
                        # two or more studs proud).
                        absorbed[q] = o
                    elif weaker and shared >= WALL_SNAP_SHARED:
                        absorbed[q] = o                     # same courses: jitter
        for q, o in absorbed.items():
            key = (axis_x, o)
            if key not in out_m:
                out_m[key] = np.zeros_like(lines[o])
                out_c[key] = Counter()
            out_m[key] |= lines[q]
            out_c[key].update(colours[(axis_x, q)])
    return out_m, out_c


DROP_EVERY = 8             # a raised wall gets a column down at its ends and every so many studs


def _drop_orphan_courses(masks: dict, solid: np.ndarray, ground_y: int) -> int:
    """A single course of wall cells with nothing under it — no wall cell,
    no solid — is not a wall: it is the underside of an eave or a sill
    read as a face.  The roof or the course above carries that; drop it.
    Longer unsupported runs (a bay, a jetty) are kept for the structure
    pass to brace."""
    H, X, Z = solid.shape
    dropped = 0
    for (axis_x, other), mask in masks.items():
        L, C = mask.shape
        # Specks: a cell with no neighbour along the wall in its course
        # and none in the course above or below is a jagged edge of the
        # plan, not a wall.
        padded = np.pad(mask, 1)
        alone = mask & ~padded[:-2, 1:-1] & ~padded[2:, 1:-1] & ~padded[1:-1, :-2] & ~padded[1:-1, 2:]
        dropped += int(alone.sum())
        mask &= ~alone
        # Strips: a column whose cells never have a neighbour column built
        # in the same course is a one-stud-wide ribbon of jitter, not a
        # wall; dropped when nothing solid stands under it.
        for u in range(L):
            col = mask[u]
            if not col.any():
                continue
            side = (mask[u - 1] if u > 0 else np.zeros_like(col)) | (mask[u + 1] if u + 1 < L else np.zeros_like(col))
            if (col & side).any():
                continue
            c_lo = int(np.flatnonzero(col).min())
            x, z = (u, other) if axis_x else (other, u)
            if c_lo == 0:
                continue
            y0 = ground_y + (c_lo - 1) * COURSE
            under = solid[y0:min(H, y0 + COURSE), x, z]
            if under.sum() < min(2, len(under)):
                dropped += int(col.sum())
                col[:] = False
        for u in range(L):
            col = mask[u]
            c = 0
            while c < C:
                if not col[c]:
                    c += 1
                    continue
                c1 = c
                while c1 < C and col[c1]:
                    c1 += 1
                run = c1 - c
                if run == 1 and c > 0:
                    x, z = (u, other) if axis_x else (other, u)
                    y0 = ground_y + (c - 1) * COURSE
                    under = solid[y0:min(H, y0 + COURSE), x, z]
                    if under.sum() < min(2, len(under)):
                        col[c] = False
                        dropped += 1
                c = c1
    return dropped


def drop_wall_supports(walls: list[WallSpec], solid: np.ndarray, ground_y: int,
                       floor_cells: dict[int, np.ndarray]) -> int:
    """A wall that starts above the ground — a tower rising out of a roof,
    a dormer, an upper storey set back — needs something under it.  The
    plan only showed its faces where they are boundaries, but the volume
    under it is solid: continue the wall's end columns and every DROP_EVERY
    studs between straight down through that solid until a floor (a plate
    sheet at that course covering the cell) or the ground picks them up.
    Those columns are part of the wall — same colour, bonded into it.
    Returns the number of courses added."""
    H = solid.shape[0]
    added = 0
    for w in walls:
        f = w.frame
        c_base = (f.y0 - ground_y) // COURSE          # the wall's first course, globally
        cols = np.flatnonzero(w.mask.any(axis=1))
        if len(cols) == 0 or c_base <= 0:
            continue
        picks = sorted({int(cols.min()), int(cols.max())}
                       | set(range(int(cols.min()) + DROP_EVERY, int(cols.max()), DROP_EVERY)))
        drops: dict[int, int] = {}                      # u → number of courses to add below
        for u in picks:
            built = np.flatnonzero(w.mask[u])
            if len(built) == 0:
                continue
            x, z = (f.origin + u, f.other) if f.axis_x else (f.other, f.origin + u)
            c = c_base + int(built.min()) - 1
            n = 0
            while c >= 0:
                fc = floor_cells.get(c)
                if fc is not None and fc[x, z]:
                    break                               # stands on this floor
                y0 = ground_y + c * COURSE
                block = solid[y0:min(H, y0 + COURSE), x, z]
                if block.sum() < min(2, len(block)):
                    break
                n += 1
                c -= 1
            if n:
                drops[u] = n
        if not drops:
            continue
        k = max(drops.values())
        new_mask = np.zeros((f.length, k + w.mask.shape[1]), dtype=bool)
        new_mask[:, k:] = w.mask
        for u, n in drops.items():
            first = int(np.flatnonzero(w.mask[u]).min()) + k
            new_mask[u, first - n:first] = True
        w.mask = new_mask
        if w.reserved is not None:
            w.reserved = np.concatenate([np.zeros((f.length, k * COURSE), bool), w.reserved], axis=1)
        w.frame = WallFrame(f.axis_x, f.origin, f.other, f.y0 - k * COURSE, f.length,
                            f.height + k * COURSE)
        w.plate_courses = {c + k for c in w.plate_courses}
        added += sum(drops.values())
    return added


def _recess_lab(palette_idx: np.ndarray, frame: WallFrame, recess: np.ndarray,
                depth: int = 3) -> Optional[np.ndarray]:
    """The mean colour (Lab) the photo gave the cells in a wall's window
    reveals — what shows through the glass: the coloured cells within
    `depth` studs of the wall plane, on either side, behind each recessed
    (u, y) cell."""
    from ..colors import _rgb_to_lab, palette
    H, X, Z = palette_idx.shape
    us, ys = np.nonzero(recess)
    if len(us) == 0:
        return None
    pal = np.array([c.rgb for c in palette()], np.float32)
    rgb = []
    for u, y in zip(us, ys):
        yy, along = frame.y0 + y, frame.origin + u
        if not 0 <= yy < H:
            continue
        for k in range(-depth, depth + 1):
            across = frame.other + k
            x, z = (along, across) if frame.axis_x else (across, along)
            if 0 <= x < X and 0 <= z < Z and palette_idx[yy, x, z] >= 0:
                rgb.append(pal[palette_idx[yy, x, z]])
    if not rgb:
        return None
    return _rgb_to_lab(np.array(rgb) / 255.0).mean(axis=0)


def _opening_colour(palette_idx: np.ndarray, frame: WallFrame) -> int:
    """The colour of a window or door on this wall: the darkest colour the
    wall's own cells carry, if it covers a few percent of them (frames
    read dark against a facade), else the first palette entry."""
    H, X, Z = palette_idx.shape
    if frame.axis_x:
        cells = palette_idx[frame.y0:frame.y0 + frame.height,
                            max(0, frame.origin):frame.origin + frame.length, frame.other] \
            if 0 <= frame.other < Z else np.array([])
    else:
        cells = palette_idx[frame.y0:frame.y0 + frame.height, frame.other,
                            max(0, frame.origin):frame.origin + frame.length] \
            if 0 <= frame.other < X else np.array([])
    vals = cells[cells >= 0] if cells.size else cells
    if vals.size == 0:
        return 0
    from ..colors import palette as _palette
    pal = _palette()
    counts = Counter(vals.tolist())
    total = sum(counts.values())
    dark = sorted((sum(pal[c].rgb) / 3, c) for c, n in counts.items()
                  if n >= 0.03 * total and 0 <= c < len(pal))
    return int(dark[0][1]) if dark else int(counts.most_common(1)[0][0])


def walls_from_solid(solid: np.ndarray, palette_idx: np.ndarray, target: MeshTarget, *,
                     ground_y: int, exclude: Optional[np.ndarray] = None,
                     detect_openings: bool = True, opening_delta_e: float = 22.0,
                     relief=None) -> list[WallSpec]:
    """One WallSpec per plane line (axis, coordinate) holding every course
    in which the plan has a boundary run there.  A course counts where at
    least two of its three plates are solid.  Cells in `exclude` (a roof's
    prism) are not walls."""
    from ..colors import palette as _palette
    H, X, Z = solid.shape
    n_courses = max(0, (H - ground_y + 1) // COURSE)
    masks: dict[tuple[bool, int], np.ndarray] = {}
    colours: dict[tuple[bool, int], Counter] = {}

    def note(axis_x: bool, other: int, u: int, c: int, colour: int) -> None:
        key = (axis_x, other)
        if key not in masks:
            masks[key] = np.zeros((X if axis_x else Z, n_courses), dtype=bool)
            colours[key] = Counter()
        masks[key][u, c] = True
        if colour >= 0:
            colours[key][colour] += 1

    for c in range(n_courses):
        y0 = ground_y + c * COURSE
        block = solid[y0:y0 + COURSE]
        if block.shape[0] == 0:
            break
        sl = block.sum(axis=0) >= min(2, block.shape[0])
        if exclude is not None:
            ex = exclude[y0:y0 + COURSE].any(axis=0)
            sl_wall = sl & ~ex
        else:
            sl_wall = sl
        if not sl_wall.any():
            continue
        pad = np.pad(sl, 1)
        ymid = min(H - 1, y0 + 1)
        # faces toward -z / +z → walls running along x at that z
        for dz, nb in ((-1, pad[1:-1, :-2]), (1, pad[1:-1, 2:])):
            face = sl_wall & ~nb
            for x, z in np.argwhere(face):
                note(True, int(z), int(x), c, int(palette_idx[ymid, x, z]))
        for dx, nb in ((-1, pad[:-2, 1:-1]), (1, pad[2:, 1:-1])):
            face = sl_wall & ~nb
            for x, z in np.argwhere(face):
                note(False, int(x), int(z), c, int(palette_idx[ymid, x, z]))

    masks, colours = _snap_wall_lines(masks, colours)
    _drop_orphan_courses(masks, solid, ground_y)
    pal = _palette()
    walls: list[WallSpec] = []
    for (axis_x, other), mask in masks.items():
        cols = np.flatnonzero(mask.any(axis=1))
        rows = np.flatnonzero(mask.any(axis=0))
        if len(cols) < MIN_WALL_STUDS or mask.sum() < 2 * MIN_WALL_STUDS:
            continue
        origin, last = int(cols.min()), int(cols.max())
        c0, c1 = int(rows.min()), int(rows.max())
        sub = mask[origin:last + 1, c0:c1 + 1]
        frame = WallFrame(axis_x, origin, other, ground_y + c0 * COURSE, last - origin + 1,
                          (c1 - c0 + 1) * COURSE)
        colour = colours[(axis_x, other)].most_common(1)[0][0] if colours[(axis_x, other)] else 0
        rgb = np.array(pal[colour].rgb, dtype=np.uint8) if 0 <= colour < len(pal) else np.zeros(3, np.uint8)
        region = _synthetic_region(frame, target, rgb)
        openings: list[Placement] = []
        boxes = target.metadata.get("forced_openings") or []
        measured = None
        if relief is not None:
            # a measured front: openings are the reveals behind the face
            from ..facade import wall_recess
            from .openings import openings_from_relief
            fit, raw_front = relief
            recess = wall_recess(fit, raw_front, axis_x=axis_x, other=other, origin=origin,
                                 length=frame.length, y0=frame.y0, height=frame.height)
            if recess is not None:
                # only where the wall is actually built (its courses)
                built = np.repeat(sub, COURSE, axis=1)[:, :frame.height]
                recess &= built
                measured = openings_from_relief(frame, recess, _opening_colour(palette_idx, frame),
                                                glass_lab=_recess_lab(palette_idx, frame, recess))
        try:
            if boxes:
                openings = forced_openings(region, target, frame, boxes)
            elif detect_openings:
                openings = find_openings(region, target, frame=frame, min_delta_e=opening_delta_e)
        except Exception:      # opening detection is a nicety; a wall without is still a wall
            openings = []
        if measured is not None:
            # Depth is the authority where it speaks (a reveal is a reveal);
            # colour fills in the flush openings depth cannot see (a bay's
            # glazing), where it does not collide with a measured one.
            def box(p):
                return (p.x_stud, p.x_stud + p.w_world, p.z_stud, p.z_stud + p.d_world,
                        p.y_plate, p.y_plate + p.part.height_units)

            def collide(a, b):
                return not (a[1] <= b[0] or b[1] <= a[0] or a[3] <= b[2] or b[3] <= a[2]
                            or a[5] <= b[4] or b[5] <= a[4])
            extra = [p for p in openings if not any(collide(box(p), box(q)) for q in measured)]
            openings = measured + extra
        reserved = reserved_cells(openings, frame) if openings else None
        walls.append(WallSpec(frame=frame, colour=int(colour), mask=sub, openings=openings,
                              reserved=reserved, region=region))
    return walls


# ---------------------------------------------------------------------------
# Piers: small solid blocks (a chimney, a pillar, a plinth) built solid
# ---------------------------------------------------------------------------

PIER_MAX_CELLS = 36        # a plan this small is a pier, not a room to leave hollow
PLINTH_COURSES = 4         # at ground level anything unowned is a plinth or steps: built solid


@dataclass
class PierSpec:
    """Solid brick courses over `cells` from course `c0` to `c1` (inclusive);
    one spec per connected small plan patch per run of courses."""
    cells: np.ndarray          # (N, 2) (x, z)
    y0: int
    n_courses: int
    colour: int


def piers_from_solid(solid: np.ndarray, palette_idx: np.ndarray, *, ground_y: int,
                     exclude: np.ndarray,
                     floor_cells: Optional[dict[int, np.ndarray]] = None) -> list[PierSpec]:
    """Course by course, the solid cells nobody owns (not a wall, not a
    roof, not a floor) that form a patch of at most PIER_MAX_CELLS: a
    chimney stack, a porch pillar, the core of a thin tower.  Larger
    patches are rooms and stay hollow — the walls, floors and roofs tie
    them.  A pier starts on the ground or on a floor; a patch that would
    hang in a room is not built.  Vertically adjacent patches with the
    same footprint become one spec so the bricks bond up the stack."""
    from scipy.ndimage import binary_closing, binary_fill_holes
    H, X, Z = solid.shape
    n_courses = max(0, (H - ground_y + 1) // COURSE)
    specs: list[PierSpec] = []
    open_specs: dict[tuple, PierSpec] = {}
    for c in range(n_courses):
        y0 = ground_y + c * COURSE
        block = solid[y0:y0 + COURSE]
        if block.shape[0] == 0:
            break
        owned = exclude[y0:y0 + COURSE].any(axis=0)
        sl = (block.sum(axis=0) >= min(2, block.shape[0])) & ~owned
        # what the walls of this course enclose is a room (a doorway's gap
        # bridged first)
        enclosed = binary_fill_holes(binary_closing(owned, iterations=2)) & ~owned
        nxt: dict[tuple, PierSpec] = {}
        for cells in _components(sl):
            # A big unowned patch is a room — left hollow, the walls and
            # floors carry it — unless it is at ground level outside the
            # walls: a plinth, a flight of steps, a terrace base, built
            # solid like a set does.
            if len(cells) > PIER_MAX_CELLS and (
                    c >= PLINTH_COURSES or enclosed[cells[:, 0], cells[:, 1]].mean() > 0.5):
                continue
            key = tuple(map(tuple, cells))
            if key in open_specs:
                spec = open_specs[key]
                spec.n_courses += 1
            else:
                if c > 0:
                    under = floor_cells.get(c - 1) if floor_cells else None
                    if under is None or under[cells[:, 0], cells[:, 1]].mean() < 0.5:
                        continue                 # nothing to stand on
                ys = np.full(len(cells), min(H - 1, y0 + 1))
                spec = PierSpec(cells=cells, y0=y0, n_courses=1,
                                colour=_colour_of(palette_idx, ys, cells[:, 0], cells[:, 1]))
                specs.append(spec)
            nxt[key] = spec
        open_specs = nxt
    return specs


def _brick_catalogue() -> list[tuple[int, int, Part]]:
    from ..bricks import catalog
    out = [(p.footprint[0], p.footprint[1], p) for p in catalog()
           if p.shape == "brick" and p.height_units == COURSE]
    out.sort(key=lambda t: -(t[0] * t[1]))
    return out


def build_pier(spec: PierSpec, claimed: Optional[np.ndarray], grid_shape: tuple[int, int, int]
               ) -> list[Placement]:
    H, X, Z = grid_shape
    parts = _brick_catalogue()
    if not parts:
        return []
    x0, z0 = int(spec.cells[:, 0].min()), int(spec.cells[:, 1].min())
    x1, z1 = int(spec.cells[:, 0].max()) + 1, int(spec.cells[:, 1].max()) + 1
    base = np.zeros((x1 - x0, z1 - z0), dtype=bool)
    base[spec.cells[:, 0] - x0, spec.cells[:, 1] - z0] = True
    out: list[Placement] = []
    for c in range(spec.n_courses):
        y = spec.y0 + c * COURSE
        if y + COURSE > H:
            break
        mask = base.copy()
        if claimed is not None:
            mask &= ~claimed[y:y + COURSE, x0:x1, z0:z1].any(axis=0)
        out += _tile_mask(mask, y, spec.colour, parts, x0=x0, z0=z0, mirror=(c % 2 == 1))
    return out


# ---------------------------------------------------------------------------
# Floors and pillars: the structure inside the rooms
# ---------------------------------------------------------------------------

FLOOR_EVERY = 10           # courses per storey at set scale (30 plates)
# A building's foot: the courses whose plan together is its footprint (the
# base plate) and within which a ground wall's ragged foot is closed.
FOOT_COURSES = 4
SPINE_EVERY = 8            # studs between the hidden walls carrying a floor
MIN_FLOOR_CELLS = 16
MIN_SETBACK = 6            # wall cells starting over the void in one course that earn a floor


@dataclass
class FloorSpec:
    """One plate layer at plate `y` over `cells` (the plan at that course:
    rooms AND wall tops, so the sheet ties the walls together).  The walls
    build that course as two plates under it.  A `base` floor (the ground
    course) is solid: two more plate layers under it, seams crossed."""
    cells: np.ndarray
    y: int
    colour: int
    base: bool = False
    colours: Optional[np.ndarray] = None   # per cell, the top layer's colour (a visible deck)


def floors_and_spines(solid: np.ndarray, walls: list[WallSpec], gables: list, *,
                      ground_y: int, structural_colour: int,
                      decks: Optional[list] = None,
                      ) -> tuple[list[FloorSpec], list[WallSpec], dict[int, np.ndarray]]:
    """Where a real set has a floor: under every roof (its ceiling, at the
    course the roof seats on), under every wall that starts above the
    ground (the base plate under the house, the sheet an upper storey
    stands on), and every FLOOR_EVERY courses up a block.  The floor is
    the top plate of that course; the walls there switch to two plate
    layers (`plate_courses`).  A floor covers only cells with solid under
    them (a sheet needs a room or a wall beneath, not a roof's overhang).

    Under each floor, SPINE walls — hidden 1-stud brick walls across the
    room every SPINE_EVERY studs, from the floor below (or the ground) up
    to this floor — carry the sheet: every floor plate 8 long crosses one.
    They are ordinary WallSpecs (bonded, plate course under the floor).
    Also returns, per floor course, the (X, Z) mask the floor covers."""
    H, X, Z = solid.shape
    n_courses = max(0, (H - ground_y + 1) // COURSE)
    floor_courses: set[int] = set()
    for g in gables:
        c = (g.y_bottom - ground_y) // COURSE - 1
        if c >= 0:
            floor_courses.add(c)
    # Under a run of wall that starts above the ground with nothing solid
    # beneath it — a set-back storey, a jetty, a wall snapped onto a line
    # the storey below doesn't reach — there is a floor to stand on; but
    # only where that happens along a real stretch of wall (MIN_SETBACK
    # cells in one course), not under every stray cell of jitter.
    starts: Counter = Counter()
    for w in walls:
        f = w.frame
        c0 = (f.y0 - ground_y) // COURSE
        for u in range(f.length):
            col = w.mask[u]
            x, z = (f.origin + u, f.other) if f.axis_x else (f.other, f.origin + u)
            if not (0 <= x < X and 0 <= z < Z):
                continue
            for cw in np.flatnonzero(col & ~np.concatenate([[False], col[:-1]])):
                c = c0 + int(cw)
                if c <= 0:
                    continue
                yb = ground_y + (c - 1) * COURSE
                under = solid[yb:yb + COURSE, x, z]
                if under.sum() < min(2, len(under)):
                    starts[c - 1] += 1
    for c, n in starts.items():
        if n >= MIN_SETBACK:
            floor_courses.add(c)
    for c in range(FLOOR_EVERY, n_courses, FLOOR_EVERY):
        if not any(abs(c - f) <= 2 for f in floor_courses):
            floor_courses.add(c)
    # A flat top (a flat roof, a terrace, a tread of the steps) is the top
    # floor of its block: same sheet, same spines under it.
    deck_cells: dict[int, np.ndarray] = {}
    deck_colour: dict[int, np.ndarray] = {}          # per course, the deck colour per cell (-1 none)
    for d in decks or []:
        c = (d.y - ground_y) // COURSE - 1
        if c < 0:
            continue
        m = deck_cells.setdefault(c, np.zeros((X, Z), dtype=bool))
        m[d.cells[:, 0], d.cells[:, 1]] = True
        col = deck_colour.setdefault(c, np.full((X, Z), -1, dtype=int))
        col[d.cells[:, 0], d.cells[:, 1]] = d.colour
        floor_courses.add(c)

    floors: list[FloorSpec] = []
    spines: list[WallSpec] = []
    floor_cells: dict[int, np.ndarray] = {}

    # The built plan: the solid plus the walls' own cells (a wall snapped
    # onto a line the solid misses at some course still needs its floor).
    wall_plan: dict[int, np.ndarray] = {}
    for w in walls:
        f = w.frame
        c0 = (f.y0 - ground_y) // COURSE
        for cw in range(w.mask.shape[1]):
            us = np.flatnonzero(w.mask[:, cw])
            if len(us) == 0:
                continue
            m = wall_plan.setdefault(c0 + cw, np.zeros((X, Z), dtype=bool))
            xs = f.origin + us if f.axis_x else np.full(len(us), f.other)
            zs = np.full(len(us), f.other) if f.axis_x else f.origin + us
            ok = (xs >= 0) & (xs < X) & (zs >= 0) & (zs < Z)
            m[xs[ok], zs[ok]] = True

    def plan_at(c: int) -> np.ndarray:
        y0 = ground_y + c * COURSE
        if c < 0 or y0 >= H:
            return np.zeros((X, Z), dtype=bool)
        block = solid[y0:y0 + COURSE]
        m = block.sum(axis=0) >= min(2, block.shape[0])
        if c in wall_plan:
            m = m | wall_plan[c]
        return m
    below_floor_course: dict[tuple[int, int], int] = {}      # (x, z) → last floor course here
    # A storey floor never runs inside a roof: cells under a gable's
    # footprint at or above its base belong to the roof recipe — except
    # the ceiling course of a gable itself, which is kept over that
    # gable's whole footprint even where another, lower roof covers it
    # (a cross-gable's rows rise out of the main roof's attic and need
    # their own sheet to stand on).
    def roofed(y0: int, c: int) -> np.ndarray:
        m = np.zeros((X, Z), dtype=bool)
        own = np.zeros((X, Z), dtype=bool)
        for g in gables:
            faces = [g.a] + ([g.b] if g.b is not None else [])
            lo = int(np.floor(min(min(f.eave, f.ridge) for f in faces)))
            hi = int(np.ceil(max(max(f.eave, f.ridge) for f in faces)))
            r0 = min(f.span[0] for f in faces)
            r1 = max(f.span[1] for f in faces)
            if (g.y_bottom - ground_y) // COURSE - 1 == c:
                # a stud wider each way: slope parts round past the eave line
                if g.a.axis == "x":
                    own[max(0, lo - 1):hi + 1, max(0, r0):r1] = True
                else:
                    own[max(0, r0):r1, max(0, lo - 1):hi + 1] = True
            if y0 + COURSE <= g.y_bottom:
                continue
            if g.a.axis == "x":
                m[max(0, lo):hi, max(0, r0):r1] = True
            else:
                m[max(0, r0):r1, max(0, lo):hi] = True
        return m & ~own

    from scipy.ndimage import binary_dilation
    last_floor = np.full((X, Z), -1, dtype=int)      # per cell, the last floor course below
    for c in sorted(floor_courses):
        y0 = ground_y + c * COURSE
        if y0 + COURSE > H:
            break
        if c == 0:
            # the base plate is the building's footprint: a
            # reconstruction's foot is ragged (plants, shadow, a matte bite),
            # so the plan of the foot courses together, its inside filled
            from scipy.ndimage import binary_fill_holes
            foot = np.zeros((X, Z), dtype=bool)
            for k in range(0, FOOT_COURSES + 1):
                if ground_y + (k + 1) * COURSE <= H:
                    foot |= plan_at(k)
            sl = binary_fill_holes(foot) & ~roofed(y0, c)
        else:
            sl = plan_at(c) & ~roofed(y0, c)
        # Walls that start right above this course stand on it, roof or
        # no roof: a tower rising out of the attic gets its floor over the
        # footprint those walls enclose (their bounding box in the plan).
        above = wall_plan.get(c + 1)
        if above is not None and above.any():
            for cells in _components(binary_dilation(above, iterations=2)):
                x0, x1 = int(cells[:, 0].min()), int(cells[:, 0].max()) + 1
                z0, z1 = int(cells[:, 1].min()), int(cells[:, 1].max()) + 1
                box = np.zeros((X, Z), dtype=bool)
                box[x0:x1, z0:z1] = True
                sl |= plan_at(c) & box
        if c >= 1:
            # on the storey below — or one stud out from it where a wall
            # of the course above stands there: a plate may ledge a stud
            # past the wall under it to carry a storey that stands a stud
            # proud, but not hang over the void for nothing
            from scipy.ndimage import binary_dilation
            under = plan_at(c - 1)
            above_walls = wall_plan.get(c + 1)
            ledge = binary_dilation(under, iterations=1) & ~under
            if above_walls is not None:
                ledge &= above_walls
            else:
                ledge[:] = False
            sl &= under | ledge
        tops = deck_cells.get(c)
        storey = c in floor_courses and (tops is None or c in {
            (g.y_bottom - ground_y) // COURSE - 1 for g in gables} | {
            (w.frame.y0 - ground_y) // COURSE - 1 for w in walls} | set(
            range(FLOOR_EVERY, n_courses, FLOOR_EVERY)))
        if tops is not None and not storey:
            sl &= tops                       # a deck only: just the flat top itself
        covered = np.zeros((X, Z), dtype=bool)
        for cells in _components(sl):
            on_deck = tops is not None and tops[cells[:, 0], cells[:, 1]].any()
            if len(cells) < (2 if on_deck else MIN_FLOOR_CELLS):
                continue
            covered[cells[:, 0], cells[:, 1]] = True
            # A visible top (a deck) keeps the model's colour there; a
            # hidden storey floor is structural.  A low sheet (the ground
            # slab, a tread of the steps) is solid plates down to the ground.
            # One sheet, one colour (its plates bridge rim and room alike,
            # like a baseplate): the colour of whatever part of it shows —
            # the deck cells — else structural.
            colour = structural_colour
            dc = deck_colour.get(c)
            if dc is not None:
                vals = dc[cells[:, 0], cells[:, 1]]
                vals = vals[vals >= 0]
                if len(vals):
                    colour = int(Counter(vals.tolist()).most_common(1)[0][0])
            floors.append(FloorSpec(cells=cells, y=y0 + COURSE - 1, base=(c == 0), colour=colour))
            inside = np.zeros((X, Z), dtype=bool)
            inside[cells[:, 0], cells[:, 1]] = True
            # Spines run along z (floor plates then run along x across
            # them) at every SPINE_EVERY-th x inside the room, from the
            # course above the last floor there up to this one.
            xs = range(int(cells[:, 0].min()) + SPINE_EVERY // 2, int(cells[:, 0].max()), SPINE_EVERY)
            for x in xs:
                zs = np.flatnonzero(inside[x])
                if len(zs) < 3:
                    continue
                c_lo = int(last_floor[x, zs].max()) + 1
                if c_lo > c:
                    continue
                mask = np.zeros((len(zs), c - c_lo + 1), dtype=bool)
                for i, z in enumerate(zs):
                    for cc in range(c_lo, c + 1):
                        if plan_at(cc)[x, z]:
                            mask[i, cc - c_lo] = True
                # trim to the run of courses that is solid all the way up
                if not mask.any():
                    continue
                frame = WallFrame(False, int(zs.min()), x, ground_y + c_lo * COURSE,
                                  int(zs.max() - zs.min() + 1), (c - c_lo + 1) * COURSE)
                full = np.zeros((frame.length, mask.shape[1]), dtype=bool)
                full[zs - zs.min()] = mask
                spines.append(WallSpec(frame=frame, colour=structural_colour, mask=full,
                                       plate_courses={c - c_lo}))
        if covered.any():
            floor_cells[c] = covered
            last_floor[covered] = c
            # the walls the floor runs over build this course as plates
            # under it (not a wall elsewhere in the model at this height)
            for w in walls:
                cw = c - (w.frame.y0 - ground_y) // COURSE
                if not (0 <= cw < w.mask.shape[1]):
                    continue
                us = np.flatnonzero(w.mask[:, cw])
                if len(us) == 0:
                    continue
                f = w.frame
                xs = f.origin + us if f.axis_x else np.full(len(us), f.other)
                zs = np.full(len(us), f.other) if f.axis_x else f.origin + us
                ok = (xs >= 0) & (xs < X) & (zs >= 0) & (zs < Z)
                if ok.any() and covered[xs[ok], zs[ok]].mean() >= 0.5:
                    w.plate_courses.add(cw)
    return floors, spines, floor_cells


def build_floor(spec: FloorSpec, claimed: Optional[np.ndarray], grid_shape: tuple[int, int, int]
                ) -> list[Placement]:
    H, X, Z = grid_shape
    parts = _plate_catalogue()
    if not parts or not (0 <= spec.y < H):
        return []
    x0, z0 = int(spec.cells[:, 0].min()), int(spec.cells[:, 1].min())
    x1, z1 = int(spec.cells[:, 0].max()) + 1, int(spec.cells[:, 1].max()) + 1
    base = np.zeros((x1 - x0, z1 - z0), dtype=bool)
    base[spec.cells[:, 0] - x0, spec.cells[:, 1] - z0] = True
    out: list[Placement] = []
    layers = [spec.y] + ([spec.y - 1, spec.y - 2] if spec.base else [])
    for i, y in enumerate(layers):
        if y < 0:
            continue
        mask = base.copy()
        if claimed is not None:
            mask &= ~claimed[y, x0:x1, z0:z1]
        # Long side along x: the spines run along z, so a plate 8 long
        # always crosses one.  Base layers alternate so their seams cross.
        groups: list[tuple[int, np.ndarray]] = [(spec.colour, mask)]
        if i == 0 and spec.colours is not None:
            groups = []
            for colour in np.unique(spec.colours):
                m = np.zeros_like(mask)
                sel = spec.cells[spec.colours == colour]
                m[sel[:, 0] - x0, sel[:, 1] - z0] = True
                groups.append((int(colour), mask & m))
        for colour, m in groups:
            out += _tile_mask(m, y, colour, parts, x0=x0, z0=z0, mirror=(i % 2 == 1),
                              long_axis="x" if i % 2 == 0 else "z")
    return out


# ---------------------------------------------------------------------------
# Sculpture: course-by-course brick building for shapes that are not
# buildings (a figure, a vehicle, a sphere) — how a set builds organic form
# ---------------------------------------------------------------------------

SCULPT_SHELL = 2           # studs of brick behind the surface; the core stays hollow
BUILDING_WALL_COVERAGE = 0.5   # share of boundary cells the wall recipe must claim
SCULPT_CORNER_SHARE = 0.3  # outline corner density above which a shape is sculpted, not built


def outline_corner_share(solid: np.ndarray, *, ground_y: int) -> float:
    """How jagged the plan outlines are: the share of boundary cells with
    two or more open sides, over all courses.  A building's outline is
    long straight runs with a corner every so often (a house: ~0.18, a
    plain gable box: ~0.06); a sphere's is a staircase (~0.46)."""
    H = solid.shape[0]
    n_courses = max(0, (H - ground_y + 1) // COURSE)
    corners = boundary = 0
    for c in range(n_courses):
        y0 = ground_y + c * COURSE
        block = solid[y0:y0 + COURSE]
        if block.shape[0] == 0:
            break
        sl = block.sum(axis=0) >= min(2, block.shape[0])
        if not sl.any():
            continue
        pad = np.pad(sl, 1)
        open_n = ((~pad[1:-1, :-2]).astype(int) + (~pad[1:-1, 2:]) + (~pad[:-2, 1:-1]) + (~pad[2:, 1:-1]))
        boundary += int((sl & (open_n > 0)).sum())
        corners += int((sl & (open_n >= 2)).sum())
    return corners / max(1, boundary)


def wall_coverage(solid: np.ndarray, walls: list[WallSpec], *, ground_y: int) -> float:
    """What share of the solid's boundary cells (per course) the walls
    account for.  A house's facades are long straight runs: high.  A
    sphere's outline is short stepped runs: low."""
    H, X, Z = solid.shape
    n_courses = max(0, (H - ground_y + 1) // COURSE)
    boundary = 0
    for c in range(n_courses):
        y0 = ground_y + c * COURSE
        block = solid[y0:y0 + COURSE]
        if block.shape[0] == 0:
            break
        sl = block.sum(axis=0) >= min(2, block.shape[0])
        pad = np.pad(sl, 1)
        edge = sl & ~(pad[1:-1, :-2] & pad[1:-1, 2:] & pad[:-2, 1:-1] & pad[2:, 1:-1])
        boundary += int(edge.sum())
    covered = sum(int(w.mask.sum()) for w in walls if w.region is not None)
    return covered / max(1, boundary)


def _slope_rot_for_outward(direction: tuple[int, int]) -> int:
    """Rotation putting a slope's low side toward `direction` (dx, dz);
    the catalogue's slopes have their low side at +x at rotation 0."""
    dx, dz = direction
    if dx > 0:
        return 0
    if dx < 0:
        return 180
    return 270 if dz > 0 else 90      # +z low side = 270, -z = 90 (see _slope_rotation)


def sculpt_solid(solid: np.ndarray, claimed: np.ndarray, palette_idx: np.ndarray, *,
                 ground_y: int, shell: int = SCULPT_SHELL) -> list[Placement]:
    """Build the solid course by course: a brick shell `shell` studs thick
    plus whatever the course above stands on, packed largest-first with
    alternating scan direction so courses bond; where the course above
    steps in by one stud the edge cells are 45° slopes facing out (two
    studs if it steps in two: 33°), where it steps OUT by one the course
    above gets inverted slopes hanging over.  Colours per part from the
    voxel palette.  Everything respects `claimed` and marks it."""
    from scipy.ndimage import binary_erosion
    from ..templates.base import part_by_ldraw
    H, X, Z = solid.shape
    n_courses = max(0, (H - ground_y + 1) // COURSE)
    bricks = _brick_catalogue()
    slopes45 = {w: part_by_ldraw(l) for w, l in ((8, "4445.dat"), (4, "3037.dat"), (3, "3038.dat"),
                                                 (2, "3039.dat"), (1, "3040.dat"))}
    slopes33 = {w: part_by_ldraw(l) for w, l in ((6, "3939.dat"), (4, "3297.dat"), (3, "4161.dat"),
                                                 (2, "3298.dat"), (1, "4286.dat"))}
    inv = part_by_ldraw("3665.dat")
    out: list[Placement] = []
    from ..bricks import made_in
    from ..colors import nearest_where, palette as _palette
    pal = _palette()

    def made_widths(table: dict, colour: int) -> dict:
        return {k: p for k, p in table.items() if p is not None and made_in(p, pal[colour].code)}

    def plan_at(c: int) -> np.ndarray:
        y0 = ground_y + c * COURSE
        if c < 0 or y0 >= H:
            return np.zeros((X, Z), dtype=bool)
        block = solid[y0:y0 + COURSE]
        return block.sum(axis=0) >= min(2, block.shape[0])

    def colour_at(y: int, x: int, z: int) -> int:
        v = int(palette_idx[min(H - 1, y), x, z])
        return v if v >= 0 else 0

    def free(y: int, x: int, z: int, h: int = COURSE, w: int = 1, d: int = 1) -> bool:
        if x < 0 or z < 0 or x + w > X or z + d > Z or y + h > H:
            return False
        return not claimed[y:y + h, x:x + w, z:z + d].any()

    def put(part, colour, x, z, y, w, d, rot) -> None:
        out.append(Placement(part=part, color_idx=colour, x_stud=x, z_stud=z, y_plate=y,
                             w_world=w, d_world=d, rotation_deg=rot))
        claimed[y:y + part.height_units, x:x + w, z:z + d] = True

    for c in range(n_courses):
        y = ground_y + c * COURSE
        if y + COURSE > H:
            break
        here, above, below = plan_at(c), plan_at(c + 1), plan_at(c - 1)
        if not here.any():
            continue
        pad = np.pad(here, 1)
        edge = here & ~(pad[1:-1, :-2] & pad[1:-1, 2:] & pad[:-2, 1:-1] & pad[2:, 1:-1])
        # Which edge cells become slopes: the course above is absent here
        # but present one (or two) studs inward in a consistent direction.
        done = np.zeros((X, Z), dtype=bool)
        for (dx, dz) in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            # the outward direction is (dx, dz); inward is the opposite
            ix, iz = -dx, -dz
            cand = np.zeros((X, Z), dtype=bool)
            for x, z in np.argwhere(edge & ~above & ~done):
                xo, zo = x + dx, z + dz
                xi, zi = x + ix, z + iz
                if 0 <= xi < X and 0 <= zi < Z and not (0 <= xo < X and 0 <= zo < Z and here[xo, zo]) \
                        and here[xi, zi]:
                    cand[x, z] = True
            if not cand.any():
                continue
            # runs along the edge (perpendicular to the direction)
            along_x = dz != 0            # edge runs along x when facing ±z
            for line in range(Z if along_x else X):
                cells = np.flatnonzero(cand[:, line] if along_x else cand[line, :])
                if len(cells) == 0:
                    continue
                runs, start = [], cells[0]
                for a, b in zip(cells, cells[1:]):
                    if b != a + 1:
                        runs.append((start, a)); start = b
                runs.append((start, cells[-1]))
                for r0, r1 in runs:
                    pos = r0
                    while pos <= r1:
                        n = r1 - pos + 1
                        # step size: does the course above sit 1 or 2 studs in?
                        x, z = (pos, line) if along_x else (line, pos)
                        two = (0 <= x + 2 * ix < X and 0 <= z + 2 * iz < Z and above[x + 2 * ix, z + 2 * iz]
                               and not above[x + ix, z + iz])
                        table, depth = (slopes33, 3) if two else (slopes45, 2)
                        # only widths LEGO makes in this colour; when the
                        # family isn't made in it at all, the slope takes
                        # the nearest colour it is made in
                        col = colour_at(y + 1, x, z)
                        made = made_widths(table, col)
                        if not made:
                            one = table.get(1) or slopes45[1]
                            col = nearest_where(col, lambda code: made_in(one, code))
                            made = made_widths(table, col)
                        w = next((k for k in sorted(made, reverse=True) if k <= n), 1)
                        part = made.get(w) or table.get(w) or slopes45[1]
                        # footprint: w along the edge, depth inward from the edge cell
                        if along_x:
                            xs, zs = x, min(z, z + iz * (depth - 1))
                            W, D = w, depth
                        else:
                            xs, zs = min(x, x + ix * (depth - 1)), z
                            W, D = depth, w
                        if free(y, xs, zs, COURSE, W, D) and all(
                                here[xs + a, zs + b] for a in range(W) for b in range(D)):
                            put(part, col, xs, zs, y, W, D, _slope_rot_for_outward((dx, dz)))
                            for a in range(W):
                                for b in range(D):
                                    done[xs + a, zs + b] = True
                        pos += w
        # Inverted slopes under an overhang: the course above extends one
        # stud past this course's edge.
        if inv is not None and c + 1 < n_courses:
            y_up = y + COURSE
            for (dx, dz) in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                for x, z in np.argwhere(above & ~here):
                    xi, zi = x - dx, z - dz           # the cell inward of the overhang
                    if not (0 <= xi < X and 0 <= zi < Z) or not here[xi, zi] or not above[xi, zi]:
                        continue
                    xs, zs = min(x, xi), min(z, zi)
                    W, D = (2, 1) if dx else (1, 2)
                    if free(y_up, xs, zs, COURSE, W, D):
                        col = nearest_where(colour_at(y_up + 1, x, z), lambda code: made_in(inv, code))
                        put(inv, col, xs, zs, y_up, W, D,
                            _slope_rot_for_outward((dx, dz)))
        # Fill: the shell, plus what the course above needs under it.
        core = binary_erosion(here, iterations=shell) if shell > 0 else np.zeros_like(here)
        fill = (here & ~core) | (here & above)
        fill &= ~done
        if not fill.any():
            continue
        cols = np.full((X, Z), -1)
        for x, z in np.argwhere(fill):
            cols[x, z] = colour_at(y + 1, x, z)
        # bricks where the whole course is free; colour groups packed apart
        whole = fill & ~claimed[y:y + COURSE].any(axis=0)
        for colour in np.unique(cols[whole]) if whole.any() else []:
            m = whole & (cols == colour)
            out_c = _tile_mask(m, y, int(colour), bricks, x0=0, z0=0, mirror=(c % 2 == 1),
                               long_axis="x" if c % 2 == 0 else "z")
            for p in out_c:
                claimed[p.y_plate:p.y_plate + p.part.height_units, p.x_stud:p.x_stud + p.w_world,
                        p.z_stud:p.z_stud + p.d_world] = True
            out.extend(out_c)
        # plates where only part of the course is free (the seam over a
        # stand that ends between course lines, under a slope's tip)
        plates = _plate_catalogue()
        for dy in range(COURSE):
            layer = fill & ~claimed[y + dy]
            if not layer.any():
                continue
            for colour in np.unique(cols[layer]):
                m = layer & (cols == colour)
                out_p = _tile_mask(m, y + dy, int(colour), plates, x0=0, z0=0, mirror=(dy % 2 == 1))
                for p in out_p:
                    claimed[p.y_plate, p.x_stud:p.x_stud + p.w_world, p.z_stud:p.z_stud + p.d_world] = True
                out.extend(out_p)
    return out


# ---------------------------------------------------------------------------
# Stand: the pedestal under a figure
# ---------------------------------------------------------------------------

STAND_STEEP_STUDS = 2      # a course that widens by this much per course needs a stand under it


def stand_mask(solid: np.ndarray, *, ground_y: int, margin: int = 1) -> np.ndarray:
    """Where a figure needs a pedestal: the columns under the parts of its
    underside that widen by STAND_STEEP_STUDS or more from one course to
    the next (inverted slopes carry a one-stud overhang; anything steeper
    hangs).  Those columns, grown by `margin` and with holes filled, make
    the stand's footprint; the stand fills each column from the base
    plate up to the figure's lowest solid cell.  Returns a (H, X, Z) mask
    of stand cells — empty when the figure stands on its own."""
    from scipy.ndimage import binary_dilation, binary_fill_holes, distance_transform_cdt
    H, X, Z = solid.shape
    n_courses = max(0, (H - ground_y + 1) // COURSE)

    def plan_at(c: int) -> np.ndarray:
        y0 = ground_y + c * COURSE
        if c < 0 or y0 >= H:
            return np.zeros((X, Z), dtype=bool)
        block = solid[y0:y0 + COURSE]
        return block.sum(axis=0) >= min(2, block.shape[0])

    steep = np.zeros((X, Z), dtype=bool)
    for c in range(n_courses - 1):
        here, above = plan_at(c), plan_at(c + 1)
        if not above.any():
            continue
        if not here.any():
            steep |= above                   # nothing at all below: everything hangs
            continue
        dist = distance_transform_cdt(~here, metric="taxicab")
        steep |= above & (dist >= STAND_STEEP_STUDS)
    if not steep.any():
        return np.zeros_like(solid)
    foot = binary_fill_holes(binary_dilation(steep, iterations=margin))
    # the figure's lowest solid plate per column
    any_col = solid.any(axis=0)
    lowest = np.where(any_col, np.argmax(solid, axis=0), H)
    out = np.zeros_like(solid)
    for x, z in np.argwhere(foot):
        top = int(lowest[x, z]) if any_col[x, z] else H
        if top > ground_y + COURSE:          # above the base plate
            out[ground_y + COURSE:top, x, z] = True
    return out


def build_stand(mask: np.ndarray, claimed: np.ndarray, colour: int, *, ground_y: int
                ) -> list[Placement]:
    """Solid brick courses over the stand mask, bonded by alternating the
    packing direction; the top of each column is finished in plates where
    the figure's underside is not on a course line."""
    H, X, Z = mask.shape
    bricks = _brick_catalogue()
    plates = _plate_catalogue()
    out: list[Placement] = []
    c = 0
    while True:
        y = ground_y + COURSE + c * COURSE
        if y >= H:
            break
        block = mask[y:min(H, y + COURSE)]
        if not block.any():
            if not mask[y:].any():
                break
            c += 1
            continue
        full = block.all(axis=0) if block.shape[0] == COURSE else np.zeros((X, Z), dtype=bool)
        full &= ~claimed[y:y + COURSE].any(axis=0)
        ps = _tile_mask(full, y, colour, bricks, x0=0, z0=0, mirror=(c % 2 == 1),
                        long_axis="x" if c % 2 == 0 else "z")
        for p in ps:
            claimed[p.y_plate:p.y_plate + COURSE, p.x_stud:p.x_stud + p.w_world,
                    p.z_stud:p.z_stud + p.d_world] = True
        out.extend(ps)
        # partial columns (the underside between course lines): plates
        for dy in range(block.shape[0]):
            layer = block[dy] & ~full & ~claimed[y + dy]
            if not layer.any():
                continue
            pl = _tile_mask(layer, y + dy, colour, plates, x0=0, z0=0, mirror=(dy % 2 == 1))
            for p in pl:
                claimed[p.y_plate, p.x_stud:p.x_stud + p.w_world, p.z_stud:p.z_stud + p.d_world] = True
            out.extend(pl)
        c += 1
    return out


# ---------------------------------------------------------------------------
# Deck recipe
# ---------------------------------------------------------------------------

_PLATE_FOOTPRINTS = ((8, 8), (6, 8), (6, 6), (4, 8), (4, 6), (4, 4), (2, 8), (2, 6), (2, 4),
                     (2, 3), (2, 2), (1, 8), (1, 6), (1, 4), (1, 3), (1, 2), (1, 1))


def _plate_catalogue() -> list[tuple[int, int, Part]]:
    from ..bricks import catalog
    out = []
    for p in catalog():
        if p.shape == "plate":
            out.append((p.footprint[0], p.footprint[1], p))
    out.sort(key=lambda t: -(t[0] * t[1]))
    return out


def _tile_mask(mask: np.ndarray, y: int, colour: int, parts: list[tuple[int, int, Part]],
               *, x0: int, z0: int, mirror: bool, long_axis: Optional[str] = None
               ) -> list[Placement]:
    """Greedy largest-first cover of a boolean (X, Z) mask with plates;
    `mirror` scans from the far corner so a second layer's seams cross
    the first's; `long_axis` tries each part with its long side along
    that axis first.  Only parts LEGO makes in `colour` are used."""
    from ..bricks import made_in
    from ..colors import palette
    code = palette()[colour].code
    parts = [t for t in parts if made_in(t[2], code)]
    free = mask.copy()
    W, D = free.shape
    out: list[Placement] = []
    xs = range(W - 1, -1, -1) if mirror else range(W)
    zs = range(D - 1, -1, -1) if mirror else range(D)
    for x in xs:
        for z in zs:
            if not free[x, z]:
                continue
            for fw, fd, part in parts:
                if fw == fd:
                    orients = ((fw, fd, 0),)
                else:
                    a, b = max(fw, fd), min(fw, fd)
                    orients = ((a, b, 0 if fw >= fd else 90), (b, a, 90 if fw >= fd else 0)) \
                        if long_axis == "x" else \
                        ((b, a, 90 if fw >= fd else 0), (a, b, 0 if fw >= fd else 90)) \
                        if long_axis == "z" else ((fw, fd, 0), (fd, fw, 90))
                for w, d, rot in orients:
                    xa, za = (x - w + 1, z - d + 1) if mirror else (x, z)
                    if xa < 0 or za < 0 or xa + w > W or za + d > D:
                        continue
                    if free[xa:xa + w, za:za + d].all():
                        free[xa:xa + w, za:za + d] = False
                        out.append(Placement(part=part, color_idx=colour, x_stud=x0 + xa,
                                             z_stud=z0 + za, y_plate=y, w_world=w, d_world=d,
                                             rotation_deg=rot))
                        break
                else:
                    continue
                break
    return out


def build_deck(spec: DeckSpec, claimed: Optional[np.ndarray], grid_shape: tuple[int, int, int]
               ) -> list[Placement]:
    H, X, Z = grid_shape
    parts = _plate_catalogue()
    if not parts:
        return []
    x0, z0 = int(spec.cells[:, 0].min()), int(spec.cells[:, 1].min())
    x1, z1 = int(spec.cells[:, 0].max()) + 1, int(spec.cells[:, 1].max()) + 1
    base = np.zeros((x1 - x0, z1 - z0), dtype=bool)
    base[spec.cells[:, 0] - x0, spec.cells[:, 1] - z0] = True
    out: list[Placement] = []
    for i, y in enumerate(spec.layers):
        if not (0 <= y < H):
            continue
        mask = base.copy()
        if claimed is not None:
            mask &= ~claimed[y, x0:x1, z0:z1]
        out += _tile_mask(mask, y, spec.colour, parts, x0=x0, z0=z0, mirror=(i == 1))
    return out
