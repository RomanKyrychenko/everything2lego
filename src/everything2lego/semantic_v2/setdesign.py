"""setdesign — build the architectural elements of a model the way a set
designer would, instead of packing whatever cells the voxel grid holds.

A LEGO set is a quantised *design*: a wall is "12 studs long, 5 bricks
high, window at stud 4", built in bonded courses from a vocabulary of
parts; a pitched roof is rows of slopes stepping in per course with plates
between and a ridge on top.  The voxel pipeline produces a quantised
*surface* and then packs it, which is why its walls follow mesh noise, its
roofs are staircases and its windows are holes.

This module sits between the region detector and the builders:

    regions ──► specs (snapped to the stud/brick grid) ──► recipes ──► placements

  WallSpec / build_wall      vertical planar regions, course by course
  GableSpec / build_gable    tilted planar regions, paired across a ridge
                             (an unpaired pitch becomes a lean-to)

Everything the recipes can't claim (detail, round elements, the base) goes
on through the ordinary builders.  Recipes respect the shared `claimed`
mask so openings placed first and slabs built earlier are never overwritten.

Slope vocabulary is 3040 (2x1) and 3039 (2x2) until the catalogue's
footprint/`slope_dir` convention for the 2x3/2x4 and 33° families is
render-verified; a wrong orientation would put every large slope on
backwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..bricks import Part, made_in
from ..pack import Placement
from ..templates.base import part_by_ldraw
from .mesh_target import MeshTarget
from .openings import WallFrame, find_openings, forced_openings, reserved_cells, wall_frame
from .regions import FoundationRegion, PlanarRegion, Region, RegionLabel

COURSE = 3   # plates per brick

# 1xN parts, longest first — the lengths the catalogue ships.
_BRICKS = ((16, "2465.dat"), (12, "6112.dat"), (8, "3008.dat"), (6, "3009.dat"),
           (4, "3010.dat"), (3, "3622.dat"), (2, "3004.dat"), (1, "3005.dat"))
_PLATES = ((8, "3460.dat"), (6, "3666.dat"), (4, "3710.dat"), (3, "3623.dat"),
           (2, "3023.dat"), (1, "3024.dat"))
_TILES = ((8, "4162.dat"), (6, "6636.dat"), (4, "2431.dat"), (2, "3069b.dat"), (1, "3070b.dat"))
# Roof slopes by (ridge-axis width, part): every family in the catalogue has
# its low side at +x and its pitch along footprint[0] (render-verified), so
# one rotation rule serves all.  45° parts are 2 deep and step 1 stud per
# brick row; 33° parts are 3 deep and step 2.
_SLOPES_45 = ((8, "4445.dat"), (4, "3037.dat"), (3, "3038.dat"), (2, "3039.dat"), (1, "3040.dat"))
_SLOPES_33 = ((6, "3939.dat"), (4, "3297.dat"), (3, "4161.dat"), (2, "3298.dat"), (1, "4286.dat"))
_SLOPE_DEPTH = {1: 2, 2: 3}       # step → part depth along the pitch


def _parts(table) -> list[tuple[int, Part]]:
    out = []
    for length, ldraw in table:
        p = part_by_ldraw(ldraw)
        if p is not None:
            out.append((length, p))
    return out


def _in_colour(table: list[tuple[int, Part]], colour: int) -> list[tuple[int, Part]]:
    """The entries of a part table LEGO makes in palette colour `colour`."""
    from ..colors import palette
    code = palette()[colour].code
    return [(L, p) for L, p in table if made_in(p, code)]


def _colour_with(colour: int, *tables: list[tuple[int, Part]],
                 lengths: tuple[int, ...] = (1, 2, 4)) -> int:
    """`colour`, or the nearest palette colour in which each table's parts
    of these lengths are made.  The 1-long part finishes any run; the 2-
    and 4-long ones are what a run is mostly made of — a family that comes
    in a colour only as 1-long parts would build the run from hundreds of
    them, so a designer picks the nearest colour it really comes in."""
    from ..colors import nearest_where
    need = [p for t in tables for L, p in t if L in lengths]
    return nearest_where(colour, lambda code: all(made_in(p, code) for p in need))


def _settle_colours(plan: "SetPlan") -> "SetPlan":
    """Make every roof's colours ones its parts are made in: each face the
    nearest colour its slope family comes in, the hips, fill and ridge
    (g.colour) one the slopes and the ridge tiles both come in.  Walls,
    floors and the rest are plain bricks and plates, which every palette
    colour comes in."""
    slopes = {1: _parts(_SLOPES_45), 2: _parts(_SLOPES_33)}
    tiles = _parts(_TILES)
    for g in plan.gables:
        faces = [g.a] + ([g.b] if g.b is not None else [])
        g.colour = _colour_with(g.colour, *(slopes[g.step(f)] for f in faces), tiles)
        for f in faces:
            f.colour = _colour_with(f.colour, slopes[g.step(f)])
    return plan


def _dominant_colour(colors: np.ndarray) -> int:
    from .builders import _dominant_palette
    return int(_dominant_palette(colors))


def _runs(free: np.ndarray) -> list[tuple[int, int]]:
    """(start, length) of each maximal run of True."""
    out, i, n = [], 0, len(free)
    while i < n:
        if not free[i]:
            i += 1
            continue
        j = i
        while j < n and free[j]:
            j += 1
        out.append((i, j - i))
        i = j
    return out


def _fewest_parts(length: int, lengths: tuple[int, ...],
                  cost: Optional[dict[int, float]] = None) -> list[int]:
    """Widths tiling `length`.  Without `cost`: the fewest parts, and among
    those the fewest 1-wide ones (so 7 is 4+3, not 6+1).  With `cost` (per
    width, see `_run_costs`): the least total cost — the split real sets
    make most likely — and among equal costs the fewest parts.  Lengths
    are small, so a plain DP."""
    best: list[Optional[list[int]]] = [None] * (length + 1)
    best[0] = []

    def key(ws):
        if cost is None:
            return (len(ws), sum(1 for w in ws if w == 1))
        return (round(sum(cost[w] for w in ws), 9), len(ws))

    for n in range(1, length + 1):
        for L in sorted(lengths, reverse=True):
            if L <= n and best[n - L] is not None:
                cand = [L] + best[n - L]
                if best[n] is None or key(cand) < key(best[n]):
                    best[n] = cand
    return best[length] or []


def _run_costs(table: list[tuple[int, Part]], role: Optional[str]) -> Optional[dict[int, float]]:
    """Per width, -log how often real sets use that part for `role` among
    the table's parts (usage.part_costs); None without a role."""
    if role is None or not table:
        return None
    from ..usage import part_costs
    c = part_costs([p.ldraw for _, p in table], role)
    return {L: c[p.ldraw] for L, p in table}


def _pack_run(length: int, table: list[tuple[int, Part]], *, lead_short: bool = False,
              role: Optional[str] = None) -> list[tuple[int, int]]:
    """(offset, width) parts tiling `length`: the split real sets make most
    likely for `role` (the fewest parts without one).  `lead_short` starts
    with a 2 so the seams land mid-part on the course below."""
    lengths = tuple(L for L, _ in table)
    if not lengths:
        return []
    out, pos = [], 0
    if lead_short and length >= 4 and 2 in lengths:
        out.append((0, 2))
        pos = 2
    for w in _fewest_parts(length - pos, lengths, _run_costs(table, role)):
        out.append((pos, w))
        pos += w
    return out


# ---------------------------------------------------------------------------
# Walls
# ---------------------------------------------------------------------------

@dataclass
class WallSpec:
    """A wall as a designer would specify it: courses on the stud grid,
    each course covering the columns the model occupies there, with its
    openings already chosen as parts."""
    frame: WallFrame                     # y0 on a course line, height = n·COURSE
    colour: int
    # (length, n_courses) — build this column in this course; None = all.
    mask: Optional[np.ndarray] = None
    openings: list[Placement] = field(default_factory=list)
    reserved: Optional[np.ndarray] = None   # (length, height) cells the openings use
    region: Optional[PlanarRegion] = None
    # Courses built as two plate layers instead of a brick, leaving the
    # top plate of the course to a floor that covers wall and room alike.
    plate_courses: set = field(default_factory=set)

    @property
    def top(self) -> int:
        return self.frame.y0 + self.frame.height


def wall_spec(region: PlanarRegion, target: MeshTarget, *, ground_y: int,
              detect_openings: bool = True, opening_delta_e: float = 22.0,
              ) -> Optional[WallSpec]:
    """Snap a vertical region to courses.

    Bottom on a course line counted from the model's ground; top from the
    REGION's own bounds (not the voxel columns — a gable triangle above a
    wall shares its columns and would make the wall a tower), rounded up to
    a whole brick.  A column is built in a course when at least two of the
    course's three plates are occupied there, so ragged tops snap per
    column and a gable end comes out as a stepped triangle.  Openings come
    from the labeller's boxes when present, else from colour."""
    raw = wall_frame(region, target)
    occ = target.voxel_grid.occupied()
    H, X, Z = occ.shape
    mn = target.mm_point_to_voxel(region.bounds[0])
    mx = target.mm_point_to_voxel(region.bounds[1])
    top = int(np.ceil(max(float(mn[1]), float(mx[1])) - 1e-4))
    bottom = min(float(mn[1]), float(mx[1]))
    y0 = max(0, ground_y + int(round((bottom - ground_y) / COURSE)) * COURSE)
    height = int(np.ceil((top - y0) / COURSE)) * COURSE
    height = max(COURSE, min(height, ((H - y0) // COURSE) * COURSE))
    if height < COURSE or raw.length < 1:
        return None
    frame = WallFrame(raw.axis_x, raw.origin, raw.other, y0, raw.length, height)

    n_courses = height // COURSE
    mask = _wall_mask(region, target, frame, n_courses)
    if mask is None or not mask.any():
        return None
    # The region's bounds reach into whatever overlaps it (the roof's eave,
    # a slab), so the real top is the last course with anything built.
    n_courses = int(np.flatnonzero(mask.any(axis=0)).max()) + 1
    mask = mask[:, :n_courses]
    frame = WallFrame(raw.axis_x, raw.origin, raw.other, y0, raw.length, n_courses * COURSE)

    openings: list[Placement] = []
    boxes = target.metadata.get("forced_openings") or []
    if boxes:
        openings = forced_openings(region, target, frame, boxes)
    elif detect_openings:
        openings = find_openings(region, target, frame=frame, min_delta_e=opening_delta_e)
    reserved = reserved_cells(openings, frame) if openings else None
    return WallSpec(frame=frame, colour=_dominant_colour(region.colors), mask=mask,
                    openings=openings, reserved=reserved, region=region)


def _wall_mask(region: PlanarRegion, target: MeshTarget, frame: WallFrame, n_courses: int
               ) -> Optional[np.ndarray]:
    """(length, n_courses) — which columns the wall covers in each course,
    from the mesh SURFACE near the wall's plane inside the region's bounds
    (the target's surface samples, plus the region's own points, which on
    an authored mesh are just the facet corners).

    The voxel column under the frame's plane is the wrong source: a real
    facade wobbles across two or three voxel layers, so one layer misses
    most of it, and the column above the wall holds the roof's interior.
    Samples are sparse per cell, so each column is filled from its lowest
    to its highest course (a wall has no vertical holes — openings are
    reserved separately) and empty columns between two filled neighbours
    take their shared span.  The top course needs the surface to reach at
    least half way up it, otherwise it rounds down."""
    samples = [np.asarray(region.points, dtype=np.float32).reshape(-1, 3)]
    sp = getattr(target, "surface_points", None)
    if sp is not None and len(sp) and region.plane_normal is not None:
        n = region.plane_normal / (np.linalg.norm(region.plane_normal) + 1e-9)
        dist = np.abs((sp - region.plane_origin) @ n)
        tol = 0.5 / max(1e-6, float(target.mesh_to_voxel[0, 0]))     # half a stud, in mm
        lo_b, hi_b = region.bounds
        inside = (dist <= tol) & (sp >= lo_b - tol).all(axis=1) & (sp <= hi_b + tol).all(axis=1)
        samples.append(sp[inside])
    mm = np.vstack(samples)
    if len(mm) == 0:
        return None
    pts = (np.hstack([mm, np.ones((len(mm), 1), np.float32)]) @ target.mesh_to_voxel.T)[:, :3]
    u = np.floor(pts[:, 0] if frame.axis_x else pts[:, 2]).astype(int) - frame.origin
    y = pts[:, 1] - frame.y0
    ok = (u >= 0) & (u < frame.length) & (y >= -0.5) & (y < n_courses * COURSE + 0.5)
    u, y = u[ok], np.clip(y[ok], 0, n_courses * COURSE - 1e-3)
    if len(u) == 0:
        return None
    mask = np.zeros((frame.length, n_courses), dtype=bool)
    lo = np.full(frame.length, np.inf)
    hi = np.full(frame.length, -np.inf)
    np.minimum.at(lo, u, y)
    np.maximum.at(hi, u, y)
    for col in range(frame.length):
        if not np.isfinite(lo[col]):
            continue
        c0 = int(lo[col] // COURSE)
        c1 = int(hi[col] // COURSE)
        if (hi[col] - c1 * COURSE) < COURSE / 2 and c1 > c0:
            c1 -= 1
        mask[col, c0:c1 + 1] = True
    # Close sampling gaps: an empty column between filled ones (within two
    # studs either side) gets the courses both neighbours share.
    filled = mask.any(axis=1)
    for col in np.flatnonzero(~filled):
        left = next((k for k in range(col - 1, max(-1, col - 3), -1) if filled[k]), None)
        right = next((k for k in range(col + 1, min(frame.length, col + 3)) if filled[k]), None)
        if left is not None and right is not None:
            mask[col] = mask[left] & mask[right]
    return mask


def _bonded_split(length: int, table: list[tuple[int, Part]], cost: Optional[dict[int, float]],
                  avoid: set[int]) -> list[tuple[int, int]]:
    """(offset, width) parts tiling `length` at the least total cost, where
    a joint at an offset in `avoid` (a joint of the course below) costs as
    much as the dearest part: a wall bonds — its joints stagger — before
    it is made of the likeliest parts.  Without `cost` every part costs 1."""
    lengths = tuple(L for L, _ in table)
    c = cost or {L: 1.0 for L in lengths}
    penalty = max(c.values())
    best: list[Optional[tuple[float, int, list[int]]]] = [None] * (length + 1)
    best[0] = (0.0, 0, [])
    for n in range(1, length + 1):
        for L in lengths:
            if L > n or best[n - L] is None:
                continue
            prev_cost, prev_n, ws = best[n - L]
            joint = penalty if (n - L) > 0 and (n - L) in avoid else 0.0
            cand = (round(prev_cost + c[L] + joint, 9), prev_n + 1, ws + [L])
            if best[n] is None or cand[:2] < best[n][:2]:
                best[n] = cand
    out, pos = [], 0
    for w in (best[length][2] if best[length] else []):
        out.append((pos, w))
        pos += w
    return out


def bond_course(length: int, course: int, free: np.ndarray,
                lengths: tuple[int, ...] = (16, 12, 8, 6, 4, 3, 2, 1), *,
                parts: Optional[dict[int, Part]] = None, role: Optional[str] = None,
                below: Optional[set[int]] = None) -> list[tuple[int, int]]:
    """(start, width) bricks for one course over the free cells of a wall.

    With the joints of the course `below` (wall offsets where two of its
    parts meet), each run of free cells is split so its joints don't land
    on those — a running bond — at the least cost: the fewest parts, or,
    given the `parts` and a `role`, the split real sets make most likely
    (`_run_costs`).  Without `below`, odd courses lead each run with a
    2-stud brick so the seams land mid-brick on the course below."""
    table = [(L, parts[L] if parts else None) for L in lengths]
    cost = _run_costs(table, role) if parts else None
    out: list[tuple[int, int]] = []
    for start, run in _runs(free):
        if below is None:
            split = _pack_run(run, table, lead_short=(course % 2 == 1), role=role if parts else None)
        else:
            avoid = {j - start for j in below if start < j < start + run}
            split = _bonded_split(run, table, cost, avoid)
        out.extend((start + off, w) for off, w in split)
    return out


def _joints(course: list[tuple[int, int]]) -> set[int]:
    """Offsets where two parts of a course meet."""
    ends = {u + w for u, w in course}
    return {u for u, _ in course if u in ends}


def interlock_corners(walls: list[WallSpec]) -> int:
    """Where two walls meet at a corner, alternate which one owns the
    corner column course by course — the bond every brick building has.
    Without it the first wall built takes the column on every course and
    the second stops a stud short, so the two never share a stud.

    Only true corners (the column is an end of both walls) or same-colour
    T-junctions are interlocked; a different-coloured wall butting into the
    middle of another would show its end bricks in that face.  Returns the
    number of corners found."""
    n = 0
    for i, A in enumerate(walls):
        for B in walls[i + 1:]:
            fa, fb = A.frame, B.frame
            if fa.axis_x == fb.axis_x:
                continue
            uA = fb.other - fa.origin
            uB = fa.other - fb.origin
            if not (0 <= uA < fa.length and 0 <= uB < fb.length):
                continue
            end_a = uA <= 0 or uA >= fa.length - 1
            end_b = uB <= 0 or uB >= fb.length - 1
            if not ((end_a and end_b) or ((end_a or end_b) and A.colour == B.colour)):
                continue
            n += 1
            for cA in range(A.mask.shape[1]):
                y = fa.y0 + cA * COURSE
                cB = (y - fb.y0) // COURSE
                if not (0 <= cB < B.mask.shape[1]) or not (A.mask[uA, cA] and B.mask[uB, cB]):
                    continue
                if (y // COURSE) % 2 == 1:
                    A.mask[uA, cA] = False
                else:
                    B.mask[uB, cB] = False
    return n


def build_wall(spec: WallSpec, claimed: Optional[np.ndarray] = None) -> list[Placement]:
    """Bonded brick courses over the spec's mask, around the openings;
    cells already claimed (a slab, another wall's corner) are skipped."""
    f = spec.frame
    bricks = dict(_in_colour(_parts(_BRICKS), spec.colour))
    plates = dict(_in_colour(_parts(_PLATES), spec.colour))
    # Window frames and doors trim the wall in its own colour (the nearest
    # they are made in), as sets do: the colour the photo shows inside an
    # opening is the room and the glass, not the frame.
    from ..colors import nearest_where
    from dataclasses import replace

    def trimmed(o: Placement) -> Placement:
        return replace(o, color_idx=nearest_where(spec.colour, lambda code: made_in(o.part, code)))
    openings = [trimmed(o) for o in spec.openings]
    if not bricks:
        return openings
    out: list[Placement] = openings
    n_courses = f.height // COURSE
    below: set[int] = set()            # joints of the course just laid

    def opening_cols(c: int) -> np.ndarray:
        if spec.reserved is None or not 0 <= c < n_courses:
            return np.zeros(f.length, bool)
        return spec.reserved[:, c * COURSE:(c + 1) * COURSE].any(axis=1)

    def sill_and_lintel(c: int) -> set[int]:
        """Offsets where course c must not have a joint: across and at the
        edges of an opening that starts on it (its sill) or ends under it
        (its lintel) — the brick there spans past the opening, or the
        opening and the brick under it hang on each other alone."""
        here = opening_cols(c)
        out_: set[int] = set()
        for edge in (opening_cols(c + 1) & ~here, opening_cols(c - 1) & ~here):
            for a, n in _runs(edge):
                out_.update(range(a, a + n + 1))
        return out_

    for c in range(n_courses):
        y = f.y0 + c * COURSE
        plated = c in spec.plate_courses and plates
        h = 2 if plated else COURSE
        free = np.ones(f.length, bool) if spec.mask is None else spec.mask[:, c].copy()
        if spec.reserved is not None:
            free &= ~spec.reserved[:, c * COURSE:(c + 1) * COURSE].any(axis=1)
        if claimed is not None:
            H, X, Z = claimed.shape
            for u in np.flatnonzero(free):
                x, z = (f.origin + u, f.other) if f.axis_x else (f.other, f.origin + u)
                if not (0 <= x < X and 0 <= z < Z) or y + h > H \
                        or claimed[y:y + h, x, z].any():
                    free[u] = False
        if plated:
            # two plate layers, the upper one bonded against the lower
            for layer in (0, 1):
                course = bond_course(f.length, c + layer, free, tuple(sorted(plates, reverse=True)),
                                     parts=plates, role="wall", below=below | sill_and_lintel(c))
                for u, w in course:
                    out.append(f.placement(plates[w], spec.colour, u, c * COURSE + layer, w))
                below = _joints(course)
        else:
            course = bond_course(f.length, c, free, tuple(sorted(bricks, reverse=True)),
                                 parts=bricks, role="wall", below=below | sill_and_lintel(c))
            for u, w in course:
                out.append(f.placement(bricks[w], spec.colour, u, c * COURSE, w))
            below = _joints(course)
    return out


# ---------------------------------------------------------------------------
# Pitched roofs
# ---------------------------------------------------------------------------

@dataclass
class PitchSpec:
    """One tilted face, in voxel units: it rises from `eave` toward `ridge`
    along `axis`; `sign` is +1 when the ridge is at the higher coordinate."""
    region: Optional[PlanarRegion]
    axis: str                 # "x" | "z"  — the pitch (up-slope) axis
    sign: int                 # +1 / -1
    eave: float               # pitch-axis coordinate of the eave
    ridge: float              # pitch-axis coordinate of the ridge
    span: tuple[int, int]     # [r0, r1) along the ridge axis
    y_bottom: int             # eave height, plates
    y_top: int                # ridge height, plates
    colour: int
    cells: Optional[np.ndarray] = None   # (N, 2) (x, z) top cells, when read off a height map

    @property
    def run(self) -> float:
        return abs(self.ridge - self.eave)


def pitch_spec(region: PlanarRegion, target: MeshTarget) -> Optional[PitchSpec]:
    """Geometry from the region's POINTS (robust to normal winding): ridge is
    the top band of points, eave the bottom band; the pitch axis is the
    horizontal axis along which they differ most."""
    pts = np.array([target.mm_point_to_voxel(p) for p in region.points], dtype=np.float32)
    if len(pts) < 4:
        return None
    y_min, y_max = float(pts[:, 1].min()), float(pts[:, 1].max())
    if y_max - y_min < 1.0:
        return None
    band = max(1.0, (y_max - y_min) * 0.15)
    top = pts[pts[:, 1] > y_max - band]
    bot = pts[pts[:, 1] < y_min + band]
    if len(top) < 2 or len(bot) < 2:
        return None
    tc, bc = top.mean(axis=0), bot.mean(axis=0)
    dx, dz = float(tc[0] - bc[0]), float(tc[2] - bc[2])
    if abs(dx) < 0.5 and abs(dz) < 0.5:
        return None
    if abs(dx) >= abs(dz):
        axis, sign, eave, ridge = "x", (1 if dx > 0 else -1), float(bc[0]), float(tc[0])
        r0, r1 = float(pts[:, 2].min()), float(pts[:, 2].max())
    else:
        axis, sign, eave, ridge = "z", (1 if dz > 0 else -1), float(bc[2]), float(tc[2])
        r0, r1 = float(pts[:, 0].min()), float(pts[:, 0].max())
    lo = int(np.floor(r0 + 1e-4))
    span = (lo, max(lo + 1, int(np.ceil(r1 - 1e-4))))
    return PitchSpec(region=region, axis=axis, sign=sign, eave=eave, ridge=ridge, span=span,
                     y_bottom=int(np.floor(y_min)), y_top=int(np.ceil(y_max)),
                     colour=_dominant_colour(region.colors))


def seat_pitches_on_walls(pitches: list[PitchSpec], walls: list[WallSpec], *,
                          ground_y: int) -> None:
    """A roof sits ON the wall under its eave.  Meshes overlap the two (and
    the eave line is a fractional plate), so the first slope row would land
    inside the wall and be skipped as claimed.  Seat each pitch on the top
    of the tallest wall running along its eave when that is within a brick
    either way; otherwise snap its base to the course grid."""
    for p in pitches:
        eave_stud = int(np.floor(p.eave + 1e-4))
        tops = [w.top for w in walls
                if w.frame.axis_x == (p.axis == "z") and abs(w.frame.other - eave_stud) <= 1]
        top = max(tops) if tops else None
        if top is not None and abs(top - p.y_bottom) <= COURSE:
            p.y_bottom = top
        else:
            p.y_bottom = ground_y + int(round((p.y_bottom - ground_y) / COURSE)) * COURSE


def clip_walls_under_roofs(walls: list[WallSpec], gables: list[GableSpec]) -> None:
    """A gable-end wall stops at the roof base; the triangle above it is the
    roof recipe's job (bricks between the slope rows), otherwise the wall's
    top courses poke through the slopes.  Only the columns under the roof's
    run are clipped, so a wall that continues past the roof keeps its
    height there."""
    for g in gables:
        faces = [g.a] + ([g.b] if g.b is not None else [])
        lo = min(min(f.eave, f.ridge) for f in faces)
        hi = max(max(f.eave, f.ridge) for f in faces)
        r0 = min(f.span[0] for f in faces)
        r1 = max(f.span[1] for f in faces)
        for w in walls:
            f = w.frame
            along_pitch = f.axis_x == (g.a.axis == "x")
            if not along_pitch or not (r0 - 1 <= f.other <= r1):
                continue
            keep = (g.y_bottom - f.y0) // COURSE
            if keep >= w.mask.shape[1]:
                continue
            under = np.array([lo - 0.5 <= f.origin + u <= hi + 0.5 for u in range(f.length)])
            w.mask[under, max(0, keep):] = False
            built = np.flatnonzero(w.mask.any(axis=0))
            n = int(built.max()) + 1 if len(built) else 0
            w.mask = w.mask[:, :n]
            w.frame = WallFrame(f.axis_x, f.origin, f.other, f.y0, f.length, n * COURSE)
            if w.reserved is not None:
                w.reserved = w.reserved[:, :n * COURSE]
                w.openings = [o for o in w.openings
                              if o.y_plate + o.part.height_units <= w.top]


@dataclass
class GableSpec:
    """Two pitches meeting at a ridge, or one pitch (lean-to) when `b` is
    None.  Each face steps in `step` studs per brick row."""
    a: PitchSpec
    b: Optional[PitchSpec]
    y_bottom: int
    n_rows: int
    colour: int
    gable_colour: int         # the brick-filled triangle at each end of the ridge
    # A hipped end slopes too: the span steps in there each row and the
    # end carries slope parts facing out along the ridge, not a brick
    # triangle.  lo/hi are the ends at span[0] and span[1].
    hip_lo: bool = False
    hip_hi: bool = False

    def step(self, face: PitchSpec) -> int:
        return int(np.clip(round(face.run / max(1, self.n_rows)), 1, 2))


MERGE_GAP_STUDS = 4        # a cross-gable or a chimney cuts a face into pieces this far apart


def merge_pitches(pitches: list[PitchSpec]) -> list[PitchSpec]:
    """One roof face, one pitch.  A segmenter slices a long face into
    pieces along the ridge (by colour, by a chimney, by sampling noise),
    and a cross-gable or the main ridge cuts the height map's view of a
    face in two; pieces on the same plane — same axis and sign, eave and
    ridge within 1.5 studs, heights within a brick — that overlap or lie
    within MERGE_GAP_STUDS along the ridge are one face with the union
    span."""
    todo = sorted(pitches, key=lambda p: p.span[0])
    out: list[PitchSpec] = []
    for p in todo:
        for q in out:
            if q.axis != p.axis or q.sign != p.sign:
                continue
            # The ridge is the one line both pieces see whole; the eave of a
            # piece is wherever a cross-gable or a dormer stopped hiding it.
            if abs(q.eave - p.eave) > 3.0 or abs(q.ridge - p.ridge) > 1.5:
                continue
            if abs(q.y_top - p.y_top) > 2 * COURSE:
                continue
            if p.span[0] > q.span[1] + MERGE_GAP_STUDS or p.span[1] < q.span[0] - MERGE_GAP_STUDS:
                continue
            wp = p.span[1] - p.span[0]
            wq = q.span[1] - q.span[0]
            w = max(1, wp + wq)
            q.span = (min(q.span[0], p.span[0]), max(q.span[1], p.span[1]))
            q.eave = (q.eave * wq + p.eave * wp) / w
            q.ridge = (q.ridge * wq + p.ridge * wp) / w
            q.y_bottom = min(q.y_bottom, p.y_bottom)
            q.y_top = max(q.y_top, p.y_top)
            if wp > wq:
                q.colour, q.region = p.colour, p.region
            break
        else:
            out.append(PitchSpec(**vars(p)))
    return out


def _gable_footprint(g: GableSpec) -> tuple[int, int, int, int]:
    """(x0, x1, z0, z1) of the roof in studs, half-open."""
    faces = [g.a] + ([g.b] if g.b is not None else [])
    lo = int(np.floor(min(min(f.eave, f.ridge) for f in faces)))
    hi = int(np.ceil(max(max(f.eave, f.ridge) for f in faces)))
    r0 = min(f.span[0] for f in faces)
    r1 = max(f.span[1] for f in faces)
    return (lo, hi, r0, r1) if g.a.axis == "x" else (r0, r1, lo, hi)


def _prism_top(g: GableSpec, X: int, Z: int) -> np.ndarray:
    """(X, Z) top plate of the stepped prism a paired gable's recipe lays
    (row i one course up, `step` studs in from each eave), -1 outside."""
    a, b = g.a, g.b
    lo_face, hi_face = (a, b) if a.sign > 0 else (b, a)
    e0, e1 = lo_face.eave, hi_face.eave
    out = np.full((X, Z), -1, dtype=int)
    r0 = max(0, min(a.span[0], b.span[0]))
    r1 = min(max(a.span[1], b.span[1]), X if a.axis == "z" else Z)
    s_lo, s_hi = g.step(lo_face), g.step(hi_face)
    for t in range(max(0, int(np.floor(e0))), int(np.ceil(e1))):
        d_lo, d_hi = t - np.floor(e0), np.ceil(e1) - 1 - t
        if d_lo < 0 or d_hi < 0 or t >= (Z if a.axis == "z" else X):
            continue
        n = int(min(g.n_rows, d_lo // s_lo + 1, d_hi // s_hi + 1))
        if a.axis == "z":
            out[r0:r1, t] = g.y_bottom + n * COURSE - 1
        else:
            out[t, r0:r1] = g.y_bottom + n * COURSE - 1
    return out


def one_roof_per_block(gables: list[GableSpec], solid: np.ndarray, *,
                       share: float = 0.75) -> list[GableSpec]:
    """A block has one roof.  The height map of a squarish block with a
    terraced or hipped top reads as two paired roofs at right angles over
    the same footprint; each would lay its slope rows through the other's.
    Where two paired gables of crossed axes at about the same height
    overlap over `share` of BOTH their footprints, the one whose prism fits
    the model's surface better stays.  (A cross gable — a T or L — overlaps
    each wing only in part, and keeps both.)"""
    from .massing import height_map
    paired = [g for g in gables if g.b is not None]
    if len(paired) < 2:
        return gables
    H, X, Z = solid.shape
    top = height_map(solid)
    prisms = {id(g): _prism_top(g, X, Z) for g in paired}

    def misfit(g: GableSpec) -> float:
        m = prisms[id(g)] >= 0
        return float(np.mean(np.abs(top[m] - prisms[id(g)][m]))) if m.any() else 1e9

    dropped: set[int] = set()
    for i, g in enumerate(paired):
        for h in paired[i + 1:]:
            if id(g) in dropped or id(h) in dropped or g.a.axis == h.a.axis:
                continue
            if abs(g.y_bottom - h.y_bottom) > 2 * COURSE:
                continue
            mg, mh = prisms[id(g)] >= 0, prisms[id(h)] >= 0
            both = (mg & mh).sum()
            if both < share * mg.sum() or both < share * mh.sum():
                continue
            dropped.add(id(h) if misfit(g) <= misfit(h) else id(g))
    return [g for g in gables if id(g) not in dropped]


def absorb_hips(gables: list[GableSpec]) -> list[GableSpec]:
    """A hip roof reads off the height map as two roofs: faces pitched
    along x and faces pitched along z over one footprint.  Where an x-roof
    and a z-roof overlap by at least half the smaller, the larger keeps
    the footprint and the other's faces become its hipped ends — the end
    the smaller's eave sits at.  A half-hip (one hipped end, one gable
    end) comes out when only one face of the smaller reaches an end."""
    gables = list(gables)
    changed = True
    while changed:
        changed = False
        for i, g in enumerate(gables):
            for j, h in enumerate(gables):
                if j <= i or g.a.axis == h.a.axis:
                    continue
                gx0, gx1, gz0, gz1 = _gable_footprint(g)
                hx0, hx1, hz0, hz1 = _gable_footprint(h)
                ox = max(0, min(gx1, hx1) - max(gx0, hx0))
                oz = max(0, min(gz1, hz1) - max(gz0, hz0))
                ag, ah = (gx1 - gx0) * (gz1 - gz0), (hx1 - hx0) * (hz1 - hz0)
                if ox * oz < 0.5 * max(1, min(ag, ah)):
                    continue
                main, other = (g, h) if ag >= ah else (h, g)
                # A hip end is one face with no opposite partner: a roof
                # that pairs two faces across its own ridge is a cross
                # gable, never a hip end, however it overlaps.
                if other.b is not None:
                    continue
                # the other's face pitches along main's ridge axis; a face
                # whose eave lies at main's span end hips that end — if
                # main's own faces are absent there (a hip end replaces
                # them with the end face; under a cross-gable they go on)
                r0, r1 = main.a.span
                main_faces = [main.a] + ([main.b] if main.b is not None else [])
                ridge_idx = 1 if main.a.axis == "x" else 0

                pitch_idx = 1 - ridge_idx
                ridge_p = float(np.mean([f.ridge for f in main_faces]))
                half = max(1.0, main.a.run / 2)

                def main_covers(lo_r: float, hi_r: float) -> float:
                    """Share of the central band (within half a run of the
                    ridge) of the end strip that main's own faces cover.
                    A hip end's face takes that band over (low share); a
                    cross-gable keeps its ridge there and loses only the
                    eaves to the valley (high share)."""
                    cells = [f.cells for f in main_faces if f.cells is not None]
                    if not cells:
                        return 0.0
                    allc = np.vstack(cells)
                    inside = ((allc[:, ridge_idx] >= lo_r) & (allc[:, ridge_idx] < hi_r)
                              & (np.abs(allc[:, pitch_idx] + 0.5 - ridge_p) <= half))
                    width = max(1.0, hi_r - lo_r)
                    return float(inside.sum()) / (width * 2 * half)

                for f in [other.a] + ([other.b] if other.b is not None else []):
                    if f.sign > 0 and abs(f.eave - r0) <= 2 and main_covers(r0, r0 + f.run) < 0.5:
                        main.hip_lo = True
                    elif f.sign < 0 and abs(f.eave - r1) <= 2 and main_covers(r1 - f.run, r1) < 0.5:
                        main.hip_hi = True
                if not (main.hip_lo or main.hip_hi):
                    continue
                main.y_bottom = min(main.y_bottom, other.y_bottom)
                gables.remove(other)
                changed = True
                break
            if changed:
                break
    return gables


def pair_pitches(pitches: list[PitchSpec], *, wall_colour: Optional[int] = None) -> list[GableSpec]:
    """Greedy pairing: same axis, opposite signs facing each other (each
    rises toward the other's eave), ridges within 2 studs or a quarter of
    the gable's width — a reconstructed ridge often runs a few degrees
    off its walls, so each face's ridge estimate lands to one side of the
    true one — spans overlapping by at least half the shorter, ridge
    heights within three bricks.  A pair shares one ridge, midway between
    the two estimates.  Leftovers become lean-tos."""
    used = [False] * len(pitches)
    out: list[GableSpec] = []
    for i, p in enumerate(pitches):
        if used[i]:
            continue
        best, best_overlap = None, 0.0
        for j, q in enumerate(pitches):
            if j == i or used[j] or q.axis != p.axis or q.sign == p.sign:
                continue
            lo_face, hi_face = (p, q) if p.sign > 0 else (q, p)
            if not lo_face.eave < hi_face.eave:          # not facing each other
                continue
            tol = max(2.0, 0.25 * abs(hi_face.eave - lo_face.eave))
            if abs(q.ridge - p.ridge) > tol or abs(q.y_top - p.y_top) > 3 * COURSE:
                continue
            lo, hi = max(p.span[0], q.span[0]), min(p.span[1], q.span[1])
            overlap = hi - lo
            shorter = min(p.span[1] - p.span[0], q.span[1] - q.span[0])
            if overlap >= 0.5 * shorter and overlap > best_overlap:
                best, best_overlap = j, overlap
        used[i] = True
        b = None
        if best is not None:
            used[best] = True
            b = pitches[best]
            if abs(b.ridge - p.ridge) > 1e-6:
                mid = 0.5 * (p.ridge + b.ridge)
                p.ridge = b.ridge = mid
        y_bottom = p.y_bottom if b is None else max(p.y_bottom, b.y_bottom)
        y_top = p.y_top if b is None else max(p.y_top, b.y_top)
        n_rows = max(1, int(round((y_top - y_bottom) / COURSE)))
        # The shallowest stock slope climbs a course per 2 studs.  A roof
        # shallower than that, built in its measured rows, stops short of
        # the ridge and leaves a flat top; a set designer builds the stock
        # slope all the way up instead (the roof stands a little taller).
        run = max(f.run for f in ([p] if b is None else [p, b]))
        if run > 2 * n_rows:
            n_rows = int(np.ceil(run / 2.0))
        out.append(GableSpec(a=p, b=b, y_bottom=y_bottom, n_rows=n_rows, colour=p.colour,
                             gable_colour=wall_colour if wall_colour is not None else p.colour))
    return out


def _slope_rotation(axis: str, sign: int) -> int:
    """3040/3039 have their LOW side at +x by default.  The low side must
    face OUTWARD, i.e. away from the ridge: -sign along the pitch axis."""
    if axis == "x":
        return 180 if sign > 0 else 0
    return 90 if sign > 0 else 270


def _pack_run_supported(length: int, table: list[tuple[int, Part]], unsup: np.ndarray,
                        *, role: Optional[str] = None) -> list[tuple[int, int]]:
    """Like `_pack_run`, but no part may lie entirely on unsupported cells
    (`unsup`, one flag per cell of the run): an eave that overhangs the
    wall is carried by parts that also sit on the wall, never by a loose
    part over the void.  Falls back to the plain packing when the whole
    run is unsupported (nothing can help it)."""
    base = _pack_run(length, table, role=role)
    cost = _run_costs(table, role)
    if not unsup.any() or unsup.all():
        return base
    lengths = tuple(L for L, _ in table)
    best = None
    # Try every first width that covers the unsupported prefix plus one
    # supported cell, and likewise mirrored for the suffix.
    pre = int(np.argmax(~unsup)) if unsup[0] else 0
    suf = int(np.argmax(~unsup[::-1])) if unsup[-1] else 0
    for w0 in sorted({L for L in lengths if L >= pre + 1 and L <= length} | ({0} if pre == 0 else set())):
        for w1 in sorted({L for L in lengths if L >= suf + 1 and w0 + L <= length} | ({0} if suf == 0 else set())):
            mid = length - w0 - w1
            if mid < 0:
                continue
            inner = _fewest_parts(mid, lengths, cost) if mid else []
            cand = ([w0] if w0 else []) + inner + ([w1] if w1 else [])
            if sum(cand) != length:
                continue
            # a part entirely over unsupported cells anywhere in the middle?
            pos, ok = 0, True
            for w in cand:
                if unsup[pos:pos + w].all():
                    ok = False
                    break
                pos += w
            if not ok:
                continue
            key = ((round(sum(cost[w] for w in cand), 9), len(cand)) if cost is not None
                   else (len(cand), sum(1 for w in cand if w == 1)))
            if best is None or key < best[0]:
                best = (key, cand)
    if best is None:
        return base
    out, pos = [], 0
    for w in best[1]:
        out.append((pos, w))
        pos += w
    return out


def build_gable(g: GableSpec, claimed: Optional[np.ndarray] = None,
                grid_shape: Optional[tuple[int, int, int]] = None,
                surface: Optional[np.ndarray] = None,
                solid: Optional[np.ndarray] = None,
                others: Optional[np.ndarray] = None,
                avoid: Optional[np.ndarray] = None) -> list[Placement]:
    """Slope rows on each face; between them plates (studded, so the next
    row attaches) with bricks in the gable colour at the two ends of the
    span; a tile deck on the last row if the faces haven't met; tiles on
    the ridge where they have.

    With `surface` (top plate per x, z) a part is skipped when every cell
    of it ends more than a course under the model's surface there — the
    half of a cross-gable under the main roof, the rows a tower stands
    through — so intersecting roofs share one skin instead of building
    through each other.

    With `solid` (the filled volume) the roof knows which of its columns
    stand on something: the eave rows overhang the void only with parts
    that also sit on the wall, and the gable-end bricks go on the first
    supported column in from each end, not on the overhang."""
    slopes = {1: _parts(_SLOPES_45), 2: _parts(_SLOPES_33)}
    if not slopes[1]:
        return []
    if not slopes[2]:
        slopes[2] = slopes[1]
    bricks, plates, tiles = _parts(_BRICKS), _parts(_PLATES), _parts(_TILES)
    # Each run packs only with parts LEGO makes in the run's colour; the
    # family (45° or 33°, its depth and step) is the roof's geometry and
    # doesn't depend on colour.
    gable_bricks = _in_colour(bricks, g.gable_colour)
    bricks, tiles = _in_colour(bricks, g.colour), _in_colour(tiles, g.colour)
    out: list[Placement] = []
    a, b = g.a, g.b
    axis = a.axis
    faces = [a] + ([b] if b is not None else [])
    r0 = min(f.span[0] for f in faces)
    r1 = max(f.span[1] for f in faces)
    H, X, Z = grid_shape if grid_shape is not None else (10**6, 10**6, 10**6)
    taken: set[tuple[int, int, int]] = set()               # cells this roof placed

    def cell_free(y: int, p: int, r: int, h: int = 1) -> bool:
        x, z = (p, r) if axis == "x" else (r, p)
        if not (0 <= x < X and 0 <= z < Z and 0 <= y and y + h <= H):
            return False
        if claimed is not None and claimed[y:y + h, x, z].any():
            return False
        if avoid is not None and avoid[y:y + h, x, z].any():
            return False
        return all((yy, x, z) not in taken for yy in range(y, y + h))

    ridge_height = g.y_bottom + g.n_rows * COURSE

    def hidden(y: int, p: int, r: int, h: int) -> bool:
        """Deep under the model's surface AND in a column that belongs to
        something else — a tower or higher block (surface well above this
        roof's ridge), or another roof's footprint (`others`).  Where the
        column is this roof's own, every row is built: the rows above
        need it to stand on."""
        if surface is None:
            return False
        x, z = (p, r) if axis == "x" else (r, p)
        if not (0 <= x < surface.shape[0] and 0 <= z < surface.shape[1]):
            return False
        top = int(surface[x, z]) + 1
        if y + h >= top - COURSE:
            return False
        if top > ridge_height + COURSE:
            return True
        return others is not None and bool(others[x, z])

    def put(part: Part, p: int, r: int, y: int, along_pitch: int, along_ridge: int,
            rot: int, colour: int, *, may_hide: bool = True) -> bool:
        h = part.height_units
        for dp in range(along_pitch):
            for dr in range(along_ridge):
                if not cell_free(y, p + dp, r + dr, h):
                    return False
        if may_hide and surface is not None and all(
                hidden(y, p + dp, r + dr, h)
                for dp in range(along_pitch) for dr in range(along_ridge)):
            return False
        x, z = (p, r) if axis == "x" else (r, p)
        w, d = (along_pitch, along_ridge) if axis == "x" else (along_ridge, along_pitch)
        out.append(Placement(part=part, color_idx=colour, x_stud=x, z_stud=z, y_plate=y,
                             w_world=w, d_world=d, rotation_deg=rot))
        for dp in range(along_pitch):
            for dr in range(along_ridge):
                xx, zz = (p + dp, r + dr) if axis == "x" else (r + dr, p + dp)
                for yy in range(y, y + h):
                    taken.add((yy, xx, zz))
        return True

    def lay_along_ridge(y: int, p: int, table: list[tuple[int, Part]], colour: int) -> None:
        """A 1xN run along the ridge axis at pitch-coordinate p."""
        hgt = table[0][1].height_units
        free = np.array([cell_free(y, p, r, hgt) for r in range(r0, r1)], dtype=bool)
        rot = 90 if axis == "x" else 0
        for start, run in _runs(free):
            for off, w in _pack_run(run, table, role="roof"):
                part = next(pt for L, pt in table if L == w)
                put(part, p, r0 + start + off, y, 1, w, rot, colour)

    def lay_along_pitch(y: int, r: int, lo: int, hi: int, table: list[tuple[int, Part]],
                        colour: int, *, may_hide: bool = True) -> None:
        """A 1xN run along the pitch axis at ridge-coordinate r (gable-end
        and fill bricks: packed as a wall's are)."""
        free = np.array([cell_free(y, p, r, table[0][1].height_units) for p in range(lo, hi)],
                        dtype=bool)
        rot = 0 if axis == "x" else 90
        for start, run in _runs(free):
            for off, w in _pack_run(run, table, role="wall"):
                part = next(pt for L, pt in table if L == w)
                put(part, lo + start + off, r, y, w, 1, rot, colour, may_hide=may_hide)

    def footprint(face: PitchSpec, row: int) -> tuple[int, int]:
        """[lo, hi) along the pitch of this face's slope parts in `row`."""
        step = g.step(face)
        depth = _SLOPE_DEPTH[step] if slopes[2] is not slopes[1] else 2
        cur = int(np.floor(face.eave + 1e-4)) + row * step * face.sign
        anchor = cur if face.sign > 0 else cur - (depth - 1)
        return anchor, anchor + depth

    # Rows stop where the two faces would run into each other; what is
    # left between their top studs is the ridge.
    n_rows = g.n_rows
    if b is not None:
        for row in range(g.n_rows):
            fa, fb = footprint(a, row), footprint(b, row)
            lo_face, hi_face = (fb, fa) if a.sign < 0 else (fa, fb)
            if lo_face[1] > hi_face[0]:
                n_rows = row
                break
    n_rows = max(1, n_rows)

    def column_supported(p: int, r: int) -> bool:
        """Something BUILT right under the roof base in this column — a
        wall top, a pier, a floor.  (Solid mesh under it proves nothing:
        the overhang's own underside is solid in the volume.)"""
        if claimed is None:
            return True
        x, z = (p, r) if axis == "x" else (r, p)
        y = g.y_bottom - 1
        if not (0 <= x < claimed.shape[1] and 0 <= z < claimed.shape[2] and 0 <= y < claimed.shape[0]):
            return False
        return bool(claimed[max(0, y - COURSE + 1):y + 1, x, z].any())

    # The columns (along the ridge) that stand on the building: the gable
    # ends go on the first of them in from each end of the span.
    lo_p = min(min(f.eave, f.ridge) for f in faces)
    hi_p = max(max(f.eave, f.ridge) for f in faces)
    pitch_cols = range(int(np.floor(lo_p)), int(np.ceil(hi_p)))
    standing = [any(column_supported(p, r) for p in pitch_cols) for r in range(r0, r1)]
    first = next((i for i, s in enumerate(standing) if s), 0)
    last = next((i for i in range(len(standing) - 1, -1, -1) if standing[i]), len(standing) - 1)
    end_a, end_b = r0 + first, r0 + last
    # Interior cross-walls (bricks along the pitch) at both ends of the
    # standing span and every 8 studs between: the plates the next row
    # rests on bridge between them, so nothing spans more than 8 studs.
    supports = sorted({end_a, end_b} | set(range(end_a + 8, end_b, 8)))

    # Hipped ends: the span steps in `hstep` studs per row at that end,
    # and slope parts `hdepth` deep face out along the ridge there.
    hstep = g.step(a)
    hdepth = _SLOPE_DEPTH[hstep] if slopes[2] is not slopes[1] else 2
    ridge_axis = "z" if axis == "x" else "x"

    gap = (0, 0)
    for row in range(n_rows):
        y = g.y_bottom + row * COURSE
        rs0 = r0 + row * hstep if g.hip_lo else r0
        rs1 = r1 - row * hstep if g.hip_hi else r1
        if rs1 - rs0 < 1:
            n_rows = row
            break
        fps = []
        for face in faces:
            step = g.step(face)
            table = _in_colour(slopes[step], face.colour)
            lo_p, hi_p = footprint(face, row)
            depth = hi_p - lo_p
            rot = _slope_rotation(axis, face.sign)
            free = np.array([all(cell_free(y, lo_p + dp, r, COURSE) for dp in range(depth))
                             for r in range(rs0, rs1)], dtype=bool)
            # Row 0 stands on the building (or doesn't); higher rows stand
            # on the row below, which exists wherever row 0 did.
            unsup_all = np.array([not any(column_supported(lo_p + dp, r) for dp in range(depth))
                                  for r in range(rs0, rs1)], dtype=bool)
            for start, run in _runs(free):
                packed = _pack_run_supported(run, table, unsup_all[start:start + run], role="roof")
                for off, w in packed:
                    part = next(pt for L, pt in table if L == w)
                    put(part, lo_p, rs0 + start + off, y, depth, w, rot, face.colour)
            fps.append((lo_p, hi_p))
        if b is None:
            continue
        (la, ha), (lb, hb) = fps
        lo, hi = (ha, lb) if a.sign > 0 else (hb, la)      # the gap between the faces
        gap = (lo, hi)
        if hi - lo < 1:
            continue
        # Hipped ends: slopes across the gap facing out along the ridge,
        # then a fill band inside them (as under the long faces).
        hip_table = _in_colour(slopes[hstep], g.colour)
        row_supports = set(supports)
        for hipped, r_edge, sign in ((g.hip_lo, rs0, +1), (g.hip_hi, rs1 - hdepth, -1)):
            if not hipped:
                continue
            rot = _slope_rotation(ridge_axis, sign)
            free = np.array([all(cell_free(y, p, r_edge + dr, COURSE) for dr in range(hdepth))
                             for p in range(lo, hi)], dtype=bool)
            for start, run in _runs(free):
                for off, w in _pack_run(run, hip_table, role="roof"):
                    part = next(pt for L, pt in hip_table if L == w)
                    put(part, lo + start + off, r_edge, y, w, hdepth, rot, g.colour)
            band = (range(r_edge + hdepth, min(rs1, r_edge + hdepth + hstep + 2)) if sign > 0
                    else range(max(rs0, r_edge - hstep - 2), r_edge))
            row_supports.update(band)
            row_supports.discard(end_a if sign > 0 else end_b)
        if gable_bricks:
            for r in sorted(row_supports):
                if not (rs0 <= r < rs1):
                    continue
                # The two gable ends are faces of the model (the surface
                # above them is the roof's ridge, not something covering
                # them); only the cross-walls between may be hidden.
                lay_along_pitch(y, r, lo, hi, gable_bricks, g.gable_colour,
                                may_hide=r not in (end_a, end_b))
        # Brick fill along the inner edge of each face, step+2 studs wide,
        # laid ACROSS the pitch (one 1xN brick per ridge position): the
        # next row's slopes rest on it (two studs' depth, not just the one
        # top stud of the row below), and each brick overlaps the fill of
        # the course under it by two studs even though the edge steps in,
        # so the fill stands on the ceiling.  The middle stays hollow.
        wa, wb = g.step(a) + 2, (g.step(b) + 2 if b is not None else 0)
        w_lo, w_hi = (wb, wa) if a.sign > 0 else (wa, wb)
        if bricks:
            # only over the columns that stand on the building: the
            # overhang carries slopes, never fill
            for r in range(max(end_a, rs0), min(end_b, rs1 - 1) + 1):
                if r in row_supports:
                    continue
                lay_along_pitch(y, r, lo, min(hi, lo + w_lo), bricks, g.colour)
                if hi - w_hi > lo + w_lo:
                    lay_along_pitch(y, r, max(lo, hi - w_hi), hi, bricks, g.colour)
    # Ridge: tiles across the two faces' top studs and whatever gap remains,
    # over the part of the span the last row still covered.
    if b is not None and tiles and n_rows >= 1:
        ridge_y = g.y_bottom + n_rows * COURSE
        fa, fb = footprint(a, n_rows - 1), footprint(b, n_rows - 1)
        top_a = fa[0] if a.sign < 0 else fa[1] - 1
        top_b = fb[0] if b.sign < 0 else fb[1] - 1
        last = n_rows - 1
        rr0 = r0 + last * hstep if g.hip_lo else r0
        rr1 = r1 - last * hstep if g.hip_hi else r1
        saved = (r0, r1)
        r0, r1 = rr0, rr1
        for p in range(min(top_a, top_b), max(top_a, top_b) + 1):
            lay_along_ridge(ridge_y, p, tiles, g.colour)
        r0, r1 = saved
    return out


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

@dataclass
class BaseSpec:
    """The ground slab: a two-layer plate deck whose top is on the course
    line the walls start from, so the walls stand on it.  Whatever is
    below the deck is the interior filler's solid foundation."""
    region: PlanarRegion
    top: int                  # plates; walls start here

    @property
    def layers(self) -> list[int]:
        return [y for y in (self.top - 1, self.top - 2) if y >= 0]


def base_spec(region: PlanarRegion, target: MeshTarget, *, ground_y: int) -> Optional[BaseSpec]:
    mn, mx = (target.mm_point_to_voxel(b) for b in region.bounds)
    top_mm = max(float(mn[1]), float(mx[1]))
    top = ground_y + int(round((top_mm - ground_y) / COURSE)) * COURSE
    return BaseSpec(region=region, top=max(ground_y + 1, top))


def build_base(spec: BaseSpec, target: MeshTarget) -> list[Placement]:
    from .builders import BuilderConfig, FoundationBuilder, _pack_horizontal_slab
    plates = FoundationBuilder()._plate_parts()
    cfg = BuilderConfig()
    out: list[Placement] = []
    for i, y in enumerate(spec.layers):
        out += _pack_horizontal_slab(spec.region, target=target, part_family=plates,
                                     y_plate=y, config=cfg, mirror=(i == 1))
    return out


# ---------------------------------------------------------------------------
# Planning + composition
# ---------------------------------------------------------------------------

# Below these a region is detail, not architecture: the recipes leave it to
# the ordinary builders.  Walls shorter than a 1x4, pitches that would be a
# single slope part or two.
MIN_WALL_STUDS = 4
MIN_WALL_FILL = 0.5       # a wall's frame is mostly wall, else it's a pilaster or an edge
MIN_PITCH_SPAN = 4
MIN_PITCH_RUN = 2.0


@dataclass
class SetPlan:
    walls: list[WallSpec]
    gables: list[GableSpec]
    consumed: set[int]        # id() of regions the recipes took over
    bases: list[BaseSpec] = field(default_factory=list)
    decks: list = field(default_factory=list)       # massing.DeckSpec
    # Top solid plate per (x, z) when the plan came from the volume: roofs
    # skip the parts another roof or a tower hides.
    surface: Optional[np.ndarray] = None
    solid: Optional[np.ndarray] = None               # the filled volume itself
    # Cells the recipes own beyond their placements' columns — the shell
    # jitter a wall absorbed, which nobody else may fill with 1x1 stacks.
    owned_extra: Optional[np.ndarray] = None
    piers: list = field(default_factory=list)       # massing.PierSpec
    floors: list = field(default_factory=list)      # massing.FloorSpec
    pruned: int = 0                                 # design cells removed for standing in the air
    roof_avoid: Optional[np.ndarray] = None         # cells a roof part stood in with nothing under
    facade_fit: object = None                       # facade.FacadeFit when a measured front was used
    facade_changed: int = 0
    # A shape that is not a building (walls cover little of its outline)
    # is sculpted course by course instead; the solid to sculpt.
    sculpt: Optional[np.ndarray] = None
    palette_idx: Optional[np.ndarray] = None
    ground_y: int = 0
    stand: Optional[np.ndarray] = None              # pedestal cells under a figure
    stand_colour: int = 0

    def owned(self, shape: tuple[int, int, int]) -> np.ndarray:
        """Cells the recipes own outright, built or not: a roof's whole
        prism from its base up (its rows are a quantised DESIGN and may
        stop a brick short of the mesh ridge — nobody else fills that
        gap), and each wall's columns over its height."""
        H, X, Z = shape
        m = np.zeros(shape, dtype=bool)
        for g in self.gables:
            faces = [g.a] + ([g.b] if g.b is not None else [])
            lo = int(np.floor(min(min(f.eave, f.ridge) for f in faces)))
            hi = int(np.ceil(max(max(f.eave, f.ridge) for f in faces))) + 1
            r0 = min(f.span[0] for f in faces)
            r1 = max(f.span[1] for f in faces)
            y0 = max(0, g.y_bottom)
            if g.a.axis == "x":
                m[y0:, max(0, lo):min(X, hi), max(0, r0):min(Z, r1)] = True
            else:
                m[y0:, max(0, r0):min(X, r1), max(0, lo):min(Z, hi)] = True
        for w in self.walls:
            f = w.frame
            y0, y1 = max(0, f.y0), min(H, w.top)
            if f.axis_x:
                m[y0:y1, max(0, f.origin):min(X, f.origin + f.length), f.other] = True
            elif 0 <= f.other < X:
                m[y0:y1, f.other, max(0, f.origin):min(Z, f.origin + f.length)] = True
        for d in self.decks:
            for y in d.layers:
                if 0 <= y < H:
                    m[y, d.cells[:, 0], d.cells[:, 1]] = True
        if self.owned_extra is not None:
            m |= self.owned_extra
        if self.sculpt is not None:
            m |= self.sculpt                # a sculpted shape is the recipe's, core and all
        if self.stand is not None:
            m |= self.stand
        return m


def plan_set(regions: list[Region], target: MeshTarget, *,
             detect_openings: bool = True, opening_delta_e: float = 22.0,
             source: str = "massing", facade=None, stand: bool = True,
             stand_colour: Optional[int] = None) -> SetPlan:
    """`source` is where the specs come from: "massing" reads walls, roofs
    and decks off the solid voxel volume (robust to segmentation noise;
    every region is consumed), "regions" snaps the detector's planar
    regions (keeps the region detail for the builders).  `facade` (a
    facade.Facade) is a measured front that overrides the model's."""
    occ = target.voxel_grid.occupied()
    rows = np.flatnonzero(occ.any(axis=(1, 2)))
    ground_y = int(rows.min()) if len(rows) else 0
    if source == "massing":
        return _settle_colours(_plan_from_massing(regions, target, occ, ground_y,
                                  detect_openings=detect_openings, opening_delta_e=opening_delta_e,
                                  facade=facade, stand=stand, stand_colour=stand_colour))
    walls: list[WallSpec] = []
    pitches: list[PitchSpec] = []
    bases: list[BaseSpec] = []
    consumed: set[int] = set()
    for reg in regions:
        if not isinstance(reg, PlanarRegion):
            continue
        if isinstance(reg, FoundationRegion):
            spec = base_spec(reg, target, ground_y=ground_y)
            if spec is not None:
                bases.append(spec)
                consumed.add(id(reg))
            continue
        if reg.is_vertical:
            spec = wall_spec(reg, target, ground_y=ground_y,
                             detect_openings=detect_openings, opening_delta_e=opening_delta_e)
            if spec is not None and spec.frame.length >= MIN_WALL_STUDS \
                    and spec.mask.any(axis=1).sum() >= MIN_WALL_STUDS \
                    and spec.mask.mean() >= MIN_WALL_FILL:
                walls.append(spec)
                consumed.add(id(reg))
        elif reg.is_tilted_up and reg.label != RegionLabel.FOUNDATION:
            spec = pitch_spec(reg, target)
            if spec is not None and spec.span[1] - spec.span[0] >= MIN_PITCH_SPAN \
                    and spec.run >= MIN_PITCH_RUN:
                pitches.append(spec)
                consumed.add(id(reg))
    wall_colour = None
    if walls:
        vals, counts = np.unique([w.colour for w in walls], return_counts=True)
        wall_colour = int(vals[int(np.argmax(counts))])
    pitches = merge_pitches(pitches)
    seat_pitches_on_walls(pitches, walls, ground_y=ground_y)
    gables = pair_pitches(pitches, wall_colour=wall_colour)
    clip_walls_under_roofs(walls, gables)
    walls = [w for w in walls if w.frame.height >= COURSE]
    # Taller walls first: a wall that reaches higher is more likely a main
    # elevation; it claims shared cells and the others yield — except at
    # corners, which alternate.
    walls.sort(key=lambda w: (-(w.frame.height), -(w.frame.length)))
    interlock_corners(walls)
    plan = SetPlan(walls=walls, gables=gables, consumed=consumed, bases=bases)
    # Whatever the segmenter found INSIDE a recipe's volume (a ceiling under
    # the roof, a facet of the wall's own thickness) is the recipe's to
    # decide; building it separately would put a deck through the slopes.
    owned = plan.owned(occ.shape)
    for reg in regions:
        if id(reg) in consumed or isinstance(reg, FoundationRegion) or len(reg.points) == 0:
            continue
        pts = np.array([target.mm_point_to_voxel(p) for p in reg.points])
        idx = np.clip(np.floor(pts).astype(int), 0, np.array(occ.shape)[[1, 0, 2]] - 1)
        inside = owned[idx[:, 1], idx[:, 0], idx[:, 2]].mean()
        if inside >= 0.6:
            consumed.add(id(reg))
    return _settle_colours(plan)


def _wall_face_cells(w: WallSpec, solid: np.ndarray) -> np.ndarray:
    """Grid (x, y, z) points on a wall's outward face, one per (column,
    plate) it builds, on the side of the line that is open air."""
    f = w.frame
    H, X, Z = solid.shape
    mask = w.mask if w.mask is not None else np.ones((f.length, f.height // COURSE), bool)
    us, cs = np.nonzero(mask)
    if len(us) == 0:
        return np.zeros((0, 3))
    out = []
    for u, c in zip(f.origin + us, cs):
        for dy in range(COURSE):
            y = int(min(H - 1, f.y0 + c * COURSE + dy))
            if f.axis_x:
                x, z = int(u), int(f.other)
                plus = z + 1 >= Z or not solid[y, x, z + 1]
                out.append((x + 0.5, y + 0.5, z + (0.9 if plus else 0.1)))
            else:
                x, z = int(f.other), int(u)
                plus = x + 1 >= X or not solid[y, x + 1, z]
                out.append((x + (0.9 if plus else 0.1), y + 0.5, z + 0.5))
    return np.array(out, np.float64)


def _openings_from_photo(walls: list[WallSpec], solid: np.ndarray, facade, fit, *,
                         min_seen: float = 0.3, opening_delta_e: float = 22.0) -> int:
    """Where the photo shows a wall, its windows and doors are the photo's:
    the photo's colour of every (stud, plate) cell of the wall's face, and
    the off-colour rectangles in it (`openings_from_grid`).  The mesh's
    colours are the photo's after a generator, a decimation and a
    quantisation, and lose all but the boldest openings.  Visibility is the
    model's own (`model_visible`): the walls were straightened after the
    photo was matched to the model, so the photo's depths no longer sit on
    them.  Walls seen less than `min_seen` keep what the mesh gave.
    Returns the walls whose openings came from the photo."""
    from ..colors import _rgb_to_lab
    from ..facade import PLATE_MM, STUD_MM, _blurred_photo, camera_intrinsics, model_visible
    from .openings import openings_from_grid, reserved_cells
    K = camera_intrinsics(facade)
    if K is None or facade.rgb is None:
        return 0
    photo = _blurred_photo(facade, fit, K)
    H, X, Z = solid.shape
    camera = fit.T[:3, 3]                               # camera centre, model mm
    scale = np.array([STUD_MM, PLATE_MM, STUD_MM])
    # A cell's colour is the photo averaged over its footprint; within that
    # footprint of the matte's edge the average takes in the background
    # (a bright band under every eave), so only cells inside it count.
    from scipy.ndimage import binary_erosion
    s_mm_per_m = float(np.cbrt(abs(np.linalg.det(fit.T[:3, :3]))))
    zs = facade.pts[..., 2][facade.valid]
    cell_px = K[0] * STUD_MM / s_mm_per_m / max(float(np.nanmedian(zs)), 1e-6)
    inner = binary_erosion(facade.mask, iterations=max(1, int(np.ceil(cell_px))))
    done = 0
    for w in walls:
        f = w.frame
        mask = w.mask if w.mask is not None else np.ones((f.length, f.height // COURSE), bool)
        idx, pts, normals = [], [], []
        for u in range(f.length):
            for yp in range(f.height):
                if not mask[u, min(mask.shape[1] - 1, yp // COURSE)]:
                    continue
                y = int(min(H - 1, f.y0 + yp))
                if f.axis_x:
                    x, z = f.origin + u, f.other
                    if not (0 <= x < X and 0 <= z < Z):
                        continue
                    plus = z + 1 >= Z or not solid[y, x, z + 1]
                    pts.append((x + 0.5, y + 0.5, z + (0.9 if plus else 0.1)))
                    normals.append((0.0, 0.0, 1.0 if plus else -1.0))
                else:
                    x, z = f.other, f.origin + u
                    if not (0 <= x < X and 0 <= z < Z):
                        continue
                    plus = x + 1 >= X or not solid[y, x + 1, z]
                    pts.append((x + (0.9 if plus else 0.1), y + 0.5, z + 0.5))
                    normals.append((1.0 if plus else -1.0, 0.0, 0.0))
                idx.append((u, yp))
        if not pts:
            continue
        pts_a = np.array(pts, np.float64)
        seen, pu, pv = model_visible(pts_a, solid, facade, fit, K)
        # and only a face turned toward the camera (a wall's back shows
        # nothing of it, whatever slips past the z-buffer at the outline)
        seen &= ((camera - pts_a * scale) * np.array(normals)).sum(axis=1) > 0
        seen[seen] &= inner[pv[seen], pu[seen]]
        if seen.mean() < min_seen:
            continue
        rgb = np.clip(photo[pv[seen], pu[seen]], 0, 255).astype(np.uint8)
        idx = np.array(idx)[seen]
        mean = np.zeros((f.length, f.height, 3))
        has = np.zeros((f.length, f.height), bool)
        mean[idx[:, 0], idx[:, 1]] = _rgb_to_lab(rgb.astype(np.float32) / 255.0)
        has[idx[:, 0], idx[:, 1]] = True
        w.openings = openings_from_grid(f, mean, has, min_delta_e=opening_delta_e, adaptive=True)
        w.reserved = reserved_cells(w.openings, f)
        done += 1
    return done


def _colour_walls_from_photo(walls: list[WallSpec], solid: np.ndarray, facade, fit,
                             *, seen_share: float = 0.3) -> None:
    """Walls take their colour from the photo where it shows them, as the
    dominant material of what it shows (not a vote of snapped cells).  A
    single photo shows one or two sides; the generator inpainted the
    rest, usually from whatever was nearest — roof slate, shadow.  A set
    designer carries the photographed wall material around the building:
    the walls the photo does not show take the dominant material of all
    the walls it does."""
    from ..colors import dominant_colour
    from ..facade import _blurred_photo, camera_intrinsics, seen_rgb
    K = camera_intrinsics(facade)
    if K is None or facade.rgb is None or not walls:
        return
    photo = _blurred_photo(facade, fit, K)
    views = []
    for w in walls:
        cells = _wall_face_cells(w, solid)
        if len(cells) == 0:
            views.append((w, 0.0, np.zeros((0, 3), np.uint8)))
            continue
        seen, rgb = seen_rgb(cells, facade, fit, K, photo=photo)
        views.append((w, float(seen.mean()), rgb))
    # The photo shows the building, so it shows some of its walls: those
    # seen over `seen_share` of their face — or, when regularising the
    # walls moved them off the depths the photo measured, the walls seen
    # most (at least half the best wall's share).
    best = max((v for _, v, rgb in views if len(rgb) >= 6), default=0.0)
    if best == 0.0:
        return
    bar = min(seen_share, 0.5 * best)
    samples: list[np.ndarray] = []
    unseen: list[WallSpec] = []
    for w, share, rgb in views:
        if share >= bar and len(rgb) >= 6:
            w.colour = dominant_colour(rgb)
            samples.append(rgb)
        else:
            unseen.append(w)
    body = dominant_colour(np.vstack(samples))
    for w in unseen:
        w.colour = body


def _seen_roof_colour(pitches: list[PitchSpec], solid: np.ndarray, facade, fit) -> Optional[int]:
    """The dominant material of the roof cells the photo shows."""
    from ..colors import dominant_colour
    from ..facade import camera_intrinsics, seen_rgb
    from .massing import height_map
    K = camera_intrinsics(facade)
    if K is None or facade.rgb is None:
        return None
    top = height_map(solid)
    cells = []
    for p in pitches:
        if p.cells is None or len(p.cells) == 0:
            continue
        xs, zs = p.cells[:, 0], p.cells[:, 1]
        ys = top[xs, zs]
        ok = ys >= 0
        cells.append(np.c_[xs[ok] + 0.5, ys[ok] + 0.9, zs[ok] + 0.5])
    if not cells:
        return None
    _, rgb = seen_rgb(np.vstack(cells), facade, fit, K, tol_studs=1.5)
    return dominant_colour(rgb, ok=_roof_buildable) if len(rgb) >= 6 else None


def _roof_buildable(code: int) -> bool:
    """Whether a roof can be built in LDraw colour `code`: both slope
    families and the ridge tiles come in it at the run sizes (1, 2, 4)."""
    need = [p for t in (_parts(_SLOPES_45), _parts(_SLOPES_33), _parts(_TILES))
            for L, p in t if L in (1, 2, 4)]
    return all(made_in(p, code) for p in need)


# A facade registration is used only when the model, seen through the
# registered camera, overlaps the photo's matte this well and this many
# measured points lie on the model's skin.
FACADE_MIN_IOU = 0.6
FACADE_MIN_INLIERS = 0.7


def _facade_trusted(facade, fit, solid: np.ndarray) -> bool:
    from ..facade import silhouette_iou
    return fit.inliers >= FACADE_MIN_INLIERS and silhouette_iou(solid, facade, fit) >= FACADE_MIN_IOU


def regularise_roofs(solid: np.ndarray, gables: list[GableSpec], *,
                     keep_above: int = 3 * COURSE) -> tuple[np.ndarray, int]:
    """Rebuild each paired gable's roof volume as the stepped prism its
    recipe lays: row i (one course) stands `step` studs in from each eave
    — and from each hipped end — up to the gable's rows.  A reconstructed
    roof is rarely that (a ridge a few degrees off its walls, terraced
    faces, an eave hanging far past the walls) and every terrace step
    would read as a wall.  The eaves are seated on the walls first (a
    stud of overhang at most: the walls' typical extent across the ridge,
    read under the roof slab).  Only the gable's own cells are rebuilt (a
    crossing gable is not this prism's), and a column that rises more
    than `keep_above` plates over the prism — a tower, a chimney, a
    dormer — keeps its height.  A roof that already matches its prism
    within a course over three quarters of its cells is left alone.
    Returns the new solid and the cells changed."""
    from scipy.ndimage import binary_dilation
    from .massing import height_map
    out = solid.copy()
    H, X, Z = solid.shape
    top = height_map(solid)
    for g in gables:
        a, b = g.a, g.b
        lo_face, hi_face = (a, b) if a.sign > 0 else (b, a)
        e0_raw, e1_raw = lo_face.eave, hi_face.eave
        if not e0_raw < e1_raw:
            continue
        y0, n_rows = g.y_bottom, g.n_rows
        slab = max(0, y0 - 2 * COURSE)
        r0 = max(0, min(a.span[0], b.span[0]))
        r1 = min(max(a.span[1], b.span[1]), X if a.axis == "z" else Z)
        band = solid[max(0, y0 - 4 * COURSE):slab]
        per = (band[:, r0:r1, :].any(axis=0) if a.axis == "z" else band[:, :, r0:r1].any(axis=0).T)
        rows_with = [np.flatnonzero(row) for row in per if row.any()]
        if not rows_with:
            continue
        lo_wall = float(np.median([r.min() for r in rows_with]))
        hi_wall = float(np.median([r.max() for r in rows_with]))
        e0 = max(e0_raw, lo_wall - 1.0)
        e1 = min(e1_raw, hi_wall + 2.0)
        if not e0 < e1:
            continue
        # and the ends on the gable walls, a stud of overhang past each:
        # per[i] is the plan at r0 + i, so a column of it runs along the ridge
        cols_with = [np.flatnonzero(col) for col in per.T if col.any()]
        end_lo, end_hi = r0, r1 - 1
        if cols_with:
            end_lo = r0 + int(np.median([c.min() for c in cols_with]))
            end_hi = r0 + int(np.median([c.max() for c in cols_with]))
            r0, r1 = max(r0, end_lo - 1), min(r1, end_hi + 2)
        s_lo, s_hi = g.step(lo_face), g.step(hi_face)
        hstep = g.step(a)                                # as the recipe steps a hipped end

        def rows_at(d_lo: float, d_hi: float, step_lo: int, step_hi: int) -> int:
            if d_lo < 0 or d_hi < 0:
                return 0
            return int(min(n_rows, d_lo // step_lo + 1, d_hi // step_hi + 1))

        own = np.zeros((X, Z), dtype=bool)
        for f in (a, b):
            if f.cells is not None and len(f.cells):
                own[f.cells[:, 0], f.cells[:, 1]] = True
        if not own.any():
            continue
        own = binary_dilation(own, iterations=2)
        prism = np.full((X, Z), -1, dtype=int)           # top plate of the prism per column
        for x, z in np.argwhere(own):
            across = z if a.axis == "z" else x
            r = x if a.axis == "z" else z
            if not (r0 <= r < r1) or across + 0.5 < e0 or across + 0.5 > e1:
                continue
            n = rows_at(across - np.floor(e0), np.ceil(e1) - 1 - across, s_lo, s_hi)
            if g.hip_lo or g.hip_hi:
                n = min(n, rows_at(r - r0 if g.hip_lo else 1e9, (r1 - 1 - r) if g.hip_hi else 1e9,
                                   hstep, hstep))
            if n > 0:
                prism[x, z] = y0 + n * COURSE - 1
        on = own & (prism >= y0) & (top >= slab)
        if not on.any() or np.percentile(np.abs(top[on] - prism[on]), 75) <= COURSE:
            continue                                     # already its prism
        roof_top = y0 + n_rows * COURSE
        for x, z in np.argwhere(own):
            if top[x, z] > roof_top + keep_above:
                continue                                 # a tower, a chimney: its own mass
            across = z if a.axis == "z" else x
            along = x if a.axis == "z" else z
            if across + 0.5 < lo_wall or across + 0.5 > hi_wall + 1 \
                    or along < end_lo or along > end_hi:
                out[slab:y0, x, z] = False               # nothing stands out past the walls
            if prism[x, z] < y0:
                if top[x, z] >= slab:
                    out[max(slab, y0):, x, z] = False    # the generator's overhang, gone
                continue
            if top[x, z] < y0 and not solid[slab:y0, x, z].any():
                continue                                 # no building under it here
            out[y0:prism[x, z] + 1, x, z] = True
            out[prism[x, z] + 1:, x, z] = False
    return out, int((out != solid).sum())


def regularise_walls(solid: np.ndarray, gables: list[GableSpec], *, ground_y: int = 0,
                     keep_above: int = 3 * COURSE) -> tuple[np.ndarray, int]:
    """Under each paired gable, rebuild the walls as the box the roof sits
    on: two straight walls along the ridge, a gable end at each end, from
    the building's foot up to the roof.  A generated house often leans a
    wall a few degrees off its roof; on the stud grid that wall becomes a
    staircase of short walls with holes between them, and the structure
    pass props what hangs over the holes with grey columns.  A set's walls
    are straight planes under straight eaves.

    Each wall line is where the walls typically are — the median of the
    outline over the walls' whole height (as `regularise_roofs` reads the
    eaves' seat), so a leaning wall comes out at its middle.  Only the
    roof's own columns are rebuilt (an annex, a porch outside it, a
    crossing gable's walls are not this box), and a column rising more
    than `keep_above` over the roof — a tower — keeps its shape.  A
    building whose foot is within FOOT_COURSES of the ground stands on
    the base: a ragged foot (plants, a matte that bit into a dark wall) is
    the reconstruction's, not the building's.  Returns the new solid and
    the cells changed."""
    from .massing import FOOT_COURSES, height_map
    out = solid.copy()
    H, X, Z = solid.shape
    top = height_map(solid)
    for g in gables:
        a, b = g.a, g.b
        if b is None:
            continue
        y0 = g.y_bottom
        own = np.zeros((X, Z), dtype=bool)
        for f in (a, b):
            if f.cells is not None and len(f.cells):
                own[f.cells[:, 0], f.cells[:, 1]] = True
        if not own.any() or y0 < COURSE:
            continue
        # this roof's columns are the ones its faces cover — not a tower or
        # a neighbouring wing beside it (in a house of several roofs those
        # stand within a stud or two of every face)
        roof_top = y0 + g.n_rows * COURSE
        cols = own & (top <= roof_top + keep_above)
        # (across, along-ridge) coordinates of the plan
        plan = (lambda m: m) if a.axis == "z" else (lambda m: m.T)   # rows: along-ridge r, cols: across
        mins_a, maxs_a, mins_r, maxs_r, feet = [], [], [], [], []
        for y in range(0, y0 - COURSE + 1, COURSE):
            layer = plan(solid[y:y + COURSE].any(axis=0) & cols)
            if not layer.any():
                continue
            for row in layer:                            # each position along the ridge
                idx = np.flatnonzero(row)
                if len(idx):
                    mins_a.append(idx.min()); maxs_a.append(idx.max())
            for col in layer.T:                          # each position across
                idx = np.flatnonzero(col)
                if len(idx):
                    mins_r.append(idx.min()); maxs_r.append(idx.max())
        if not mins_a or not mins_r:
            continue
        lo_a, hi_a = int(np.median(mins_a)), int(np.median(maxs_a))
        lo_r, hi_r = int(np.median(mins_r)), int(np.median(maxs_r))
        if hi_a - lo_a < 2 or hi_r - lo_r < 2:
            continue
        box = np.zeros((X, Z), dtype=bool)
        if a.axis == "z":
            box[lo_r:hi_r + 1, lo_a:hi_a + 1] = True
        else:
            box[lo_a:hi_a + 1, lo_r:hi_r + 1] = True
        # the foot: where the building under this roof starts, typically
        bottoms = [int(np.flatnonzero(solid[:y0, x, z]).min())
                   for x, z in np.argwhere(box & cols) if solid[:y0, x, z].any()]
        if not bottoms:
            continue
        foot = int(np.median(bottoms))
        if foot - ground_y <= FOOT_COURSES * COURSE:
            foot = ground_y
        for x, z in np.argwhere(cols):
            out[foot:y0, x, z] = box[x, z]
    return out, int((out != solid).sum())


def _colour_new_cells(solid: np.ndarray, palette_idx: np.ndarray) -> np.ndarray:
    """Cells of the solid with no colour (filled by a carve or a
    regularisation) take the colour of the nearest coloured cell — the
    surface they replaced."""
    from scipy.ndimage import distance_transform_edt
    known = palette_idx >= 0
    if not known.any() or not (solid & ~known).any():
        return palette_idx
    _, (iy, ix, iz) = distance_transform_edt(~known, return_indices=True)
    palette_idx = palette_idx.copy()
    fill = solid & ~known
    palette_idx[fill] = palette_idx[iy[fill], ix[fill], iz[fill]]
    return palette_idx


def ground_walls(walls: list[WallSpec], *, ground_y: int, max_gap: Optional[int] = None) -> int:
    """Walls stand on the base plate.  A reconstruction's foot is ragged —
    plants, a shadow, a matte that bit into a dark wall — and the plan
    reads gaps there; a column of a wall that starts on the ground but
    whose bricks begin up to `max_gap` courses above it is continued down
    to the base (`max_gap` defaults to the building's foot, FOOT_COURSES;
    a taller gap is an opening).  Returns the courses added."""
    from .massing import FOOT_COURSES
    max_gap = FOOT_COURSES if max_gap is None else max_gap
    added = 0
    for w in walls:
        f = w.frame
        if w.mask is None or (f.y0 - ground_y) // COURSE > 0:
            continue                                     # not a ground-floor wall
        for u in range(w.mask.shape[0]):
            built = np.flatnonzero(w.mask[u])
            if len(built) == 0:
                continue
            b = int(built.min())
            if 0 < b <= max_gap:
                w.mask[u, :b] = True
                added += b
    return added


def _plan_from_massing(regions: list[Region], target: MeshTarget, occ: np.ndarray, ground_y: int,
                       *, detect_openings: bool, opening_delta_e: float, facade=None,
                       stand: bool = True, stand_colour: Optional[int] = None) -> SetPlan:
    from .massing import height_map, pitches_and_decks, solid_volume, walls_from_solid
    palette_idx = target.voxel_grid.palette_idx
    solid = solid_volume(occ)
    facade_fit, facade_changed, relief = None, 0, None
    # Building or sculpture is a property of the whole shape, read before
    # the photo corrects it: a carve seen from one side cuts view-shaped
    # (diagonal) wedges that would make a building's outline look carved.
    from .massing import SCULPT_CORNER_SHARE as SCULPT_CORNER_SHARE_, outline_corner_share
    coverage = outline_corner_share(solid, ground_y=ground_y)
    if facade is not None:
        from scipy.ndimage import binary_fill_holes
        from ..facade import carve_facade, measured_front, register, square_walls
        facade_fit = register(facade, solid)
    if facade is not None and not _facade_trusted(facade, facade_fit, solid):
        # a registration this poor would carve the wrong cells and paint
        # the wrong colours: the model stays as generated
        facade = None
    if facade is not None:
        if coverage <= SCULPT_CORNER_SHARE_:
            # a building's walls meet square; a photo's depth bends them
            facade, _ = square_walls(facade, facade_fit)
        solid, facade_changed = carve_facade(solid, facade, facade_fit)
        # cavities the carving opened are closed again; the leaky-shell
        # heuristics are NOT re-run, they would undo what the rays cut
        solid = binary_fill_holes(solid)
        # the shell the recipes read colours from follows the new front
        occ = occ & solid
        # cells the carve filled (a front brought forward) or exposed (the
        # inside of a cut tower) carry no colour: they take the colour of
        # the nearest coloured cell — the surface they replaced
        palette_idx = _colour_new_cells(solid, palette_idx)
        # and where the photo sees the model, the photo's own colours
        from ..facade import photo_colours
        palette_idx, _ = photo_colours(solid, palette_idx, facade, facade_fit)
        # the unsmoothed front: window reveals are a stud wide and a
        # few plates deep, below what the smoothing keeps
        raw_front, _ = measured_front(facade, facade_fit, solid.shape)
        relief = (facade_fit, raw_front)
    # Not a building?  (Its outlines are staircases, not straight runs.)
    # Then no walls, roofs or floors: the whole shape is sculpted in brick
    # courses, standing on a base plate over its footprint.
    from .massing import FloorSpec
    if coverage > SCULPT_CORNER_SHARE_:
        from .massing import stand_mask
        from ..colors import palette as _palette
        if stand_colour is None:
            pal = _palette()
            stand_colour = next((i for i, c in enumerate(pal) if c.code == 72), 0)   # dark bluish grey
        pedestal = stand_mask(solid, ground_y=ground_y) if stand else None
        if pedestal is not None and not pedestal.any():
            pedestal = None
        foot = solid.any(axis=0) | (pedestal.any(axis=0) if pedestal is not None else False)
        footprint = np.argwhere(foot)
        base = FloorSpec(cells=footprint, y=ground_y + COURSE - 1, colour=stand_colour, base=True)
        plan = SetPlan(walls=[], gables=[], consumed={id(r) for r in regions}, floors=[base],
                       sculpt=solid, palette_idx=palette_idx, ground_y=ground_y,
                       stand=pedestal, stand_colour=stand_colour)
        plan.surface = height_map(solid)
        plan.solid = solid
        plan.facade_fit, plan.facade_changed = facade_fit, facade_changed
        plan.coverage = coverage
        return plan
    # The roof first: its pitches, paired into gables, define an ideal
    # prism over the footprint.  A reconstructed roof is rarely that prism
    # — its ridge runs a few degrees off its walls, its faces terrace —
    # and every terrace step would read as a wall.  So the roof volume is
    # rebuilt as the prism before the walls are read off the solid.
    pitches, decks, skin = pitches_and_decks(solid, palette_idx, ground_y=ground_y)
    pitches = merge_pitches(pitches)
    prism_gables = [g for g in one_roof_per_block(absorb_hips(pair_pitches(pitches)), solid)
                    if g.b is not None]
    if prism_gables:
        # the walls under each roof as the box it sits on, then the roof
        # seated on those walls
        solid, walls_changed = regularise_walls(solid, prism_gables, ground_y=ground_y)
        solid, changed = regularise_roofs(solid, prism_gables)
        if walls_changed:
            palette_idx = _colour_new_cells(solid, palette_idx)
        if changed or walls_changed:
            pitches, decks, skin = pitches_and_decks(solid, palette_idx, ground_y=ground_y)
            pitches = merge_pitches(pitches)
    walls = walls_from_solid(solid, palette_idx, target, ground_y=ground_y, exclude=skin,
                             detect_openings=detect_openings, opening_delta_e=opening_delta_e,
                             relief=relief)
    ground_walls(walls, ground_y=ground_y)
    if facade is not None:
        _colour_walls_from_photo(walls, solid, facade, facade_fit)
        if detect_openings:
            _openings_from_photo(walls, solid, facade, facade_fit, opening_delta_e=opening_delta_e)
    wall_colour = None
    if walls:
        vals, counts = np.unique([w.colour for w in walls], return_counts=True)
        wall_colour = int(vals[int(np.argmax(counts))])
    # One roof colour for the building: the reconstruction's sampling
    # splits one slate roof between two palette entries patch by patch,
    # and no set mixes them face by face.
    if pitches:
        weights: dict[int, float] = {}
        for p in pitches:
            weights[p.colour] = weights.get(p.colour, 0.0) + p.run * (p.span[1] - p.span[0])
        roof_colour = max(weights, key=weights.get)
        if facade is not None:
            # what the photo shows of the roof decides its colour; the
            # rest of the reconstruction's roof was inpainted
            seen_roof = _seen_roof_colour(pitches, solid, facade, facade_fit)
            if seen_roof is not None:
                roof_colour = seen_roof
        for p in pitches:
            p.colour = roof_colour
    seat_pitches_on_walls(pitches, walls, ground_y=ground_y)
    gables = one_roof_per_block(absorb_hips(pair_pitches(pitches, wall_colour=wall_colour)), solid)
    clip_walls_under_roofs(walls, gables)
    walls = [w for w in walls if w.frame.height >= COURSE]
    walls.sort(key=lambda w: (-(w.frame.height), -(w.frame.length)))
    # A flat strip on a ridge is the roof's (its tiles close it); a flat
    # patch elsewhere — a tower top, a terrace, a step — is a deck.
    kept = []
    for d in decks:
        on_ridge = np.zeros(len(d.cells), dtype=bool)
        for g in gables:
            if abs(d.y - (g.y_bottom + g.n_rows * COURSE)) > COURSE and abs(d.y - g.a.y_top) > COURSE:
                continue
            faces = [g.a] + ([g.b] if g.b is not None else [])
            r0 = min(f.span[0] for f in faces)
            r1 = max(f.span[1] for f in faces)
            ridge = float(np.mean([f.ridge for f in faces]))
            along = d.cells[:, 0] if g.a.axis == "x" else d.cells[:, 1]
            across = d.cells[:, 1] if g.a.axis == "x" else d.cells[:, 0]
            on_ridge |= (np.abs(along + 0.5 - ridge) <= 2.0) & (across >= r0) & (across < r1)
        if (~on_ridge).sum() >= 4:
            d.cells = d.cells[~on_ridge]
            kept.append(d)
    plan = SetPlan(walls=walls, gables=gables, consumed={id(r) for r in regions},
                   bases=[], decks=kept)
    plan.surface = height_map(solid)
    plan.solid = solid
    # A wall stands for the facade's whole wobble: the shell cells within
    # WALL_SNAP_STUDS either side of its line, over its courses, are its
    # (built as the wall, not again as plate stacks by the filler).
    from .massing import WALL_SNAP_STUDS
    H, X, Z = occ.shape
    extra = np.zeros_like(occ)
    for w in walls:
        f = w.frame
        y0, y1 = max(0, f.y0), min(H, w.top)
        lo = max(0, f.other - WALL_SNAP_STUDS)
        hi = min((Z if f.axis_x else X), f.other + WALL_SNAP_STUDS + 1)
        u0, u1 = max(0, f.origin), min((X if f.axis_x else Z), f.origin + f.length)
        if f.axis_x:
            extra[y0:y1, u0:u1, lo:hi] = True
        else:
            extra[y0:y1, lo:hi, u0:u1] = True
    plan.owned_extra = extra & occ
    # Small solid blocks the walls and roofs don't account for are built
    # solid — a chimney, a pillar, a plinth — so what stands on them is held.
    from .massing import drop_wall_supports, floors_and_spines, piers_from_solid
    # Floors: a plate sheet at every storey, under every roof and under
    # every raised wall, covering wall tops and room alike so it ties the
    # walls; hidden spine walls under it.  Then raised walls drop columns
    # down to the floor that carries them, and the interlock sees the
    # final masks.
    # A flat top at ground level — a tread of the steps, a plinth, a low
    # terrace — is not a floor over a room; the solid under it is built
    # solid by the pier recipe (PLINTH_COURSES), so only higher tops
    # become floors.
    from .massing import PLINTH_COURSES
    high_decks = [d for d in plan.decks
                  if d.y <= ground_y + COURSE                      # the base plate itself stays a sheet
                  or d.y > ground_y + PLINTH_COURSES * COURSE]
    plan.floors, spines, floor_cells = floors_and_spines(
        solid, walls, gables, ground_y=ground_y, structural_colour=wall_colour or 0,
        decks=high_decks)
    plan.decks = []                    # flat tops are floors (or plinths) now
    drop_wall_supports(walls, solid, ground_y, floor_cells)
    walls.extend(spines)
    plan.walls = walls
    interlock_corners(walls)
    # Piers: solid bits nobody else accounts for, outside every roof's
    # prism (the roof recipe owns its attic, hollow or not).
    taken = plan.owned(occ.shape) | skin
    H, X, Z = occ.shape
    for g in gables:
        faces = [g.a] + ([g.b] if g.b is not None else [])
        lo = int(np.floor(min(min(f.eave, f.ridge) for f in faces)))
        hi = int(np.ceil(max(max(f.eave, f.ridge) for f in faces)))
        r0 = min(f.span[0] for f in faces)
        r1 = max(f.span[1] for f in faces)
        y0 = max(0, g.y_bottom - COURSE)
        if g.a.axis == "x":
            taken[y0:, max(0, lo):hi, max(0, r0):r1] = True
        else:
            taken[y0:, max(0, r0):r1, max(0, lo):hi] = True
    for fl in plan.floors:
        taken[fl.y, fl.cells[:, 0], fl.cells[:, 1]] = True
    plan.piers = piers_from_solid(solid, palette_idx, ground_y=ground_y, exclude=taken,
                                  floor_cells=floor_cells)
    plan.pruned = prune_unsupported(plan, occ.shape)
    plan.facade_fit, plan.facade_changed = facade_fit, facade_changed
    return plan


def prune_unsupported(plan: SetPlan, grid_shape: tuple[int, int, int], *, rounds: int = 5) -> int:
    """A set designer never leaves a brick in the air; when a cell of the
    design has nothing under it, the design changes, not the physics.
    Dry-build the plan, find every wall, floor or pier part with nothing
    at all beneath it, and take those cells out of their specs; repeat,
    since removing a plate can expose the plate that stood on it.  Roof
    parts are left to the structure pass (a slope's support is the row
    below, which the recipe already arranges).  Returns cells removed."""
    from ..stability import check as _check
    from .massing import build_floor, build_pier
    removed = 0
    for _ in range(rounds):
        claimed = np.zeros(grid_shape, dtype=bool)
        owners: list[tuple[Placement, str, int]] = []

        def claim(ps):
            for p in ps:
                claimed[p.y_plate:p.y_plate + p.part.height_units,
                        p.x_stud:p.x_stud + p.w_world, p.z_stud:p.z_stud + p.d_world] = True

        for i, w in enumerate(plan.walls):
            ps = build_wall(w, claimed)
            claim(ps)
            owners += [(p, "wall", i) for p in ps]
        for i, pr in enumerate(plan.piers):
            ps = build_pier(pr, claimed, grid_shape)
            claim(ps)
            owners += [(p, "pier", i) for p in ps]
        for i, fl in enumerate(plan.floors):
            ps = build_floor(fl, claimed, grid_shape)
            claim(ps)
            owners += [(p, "floor", i) for p in ps]
        if plan.roof_avoid is None:
            plan.roof_avoid = np.zeros(grid_shape, dtype=bool)
        for g in plan.gables:
            ps = build_gable(g, claimed, grid_shape, surface=plan.surface, solid=plan.solid,
                             avoid=plan.roof_avoid)
            claim(ps)
            owners += [(p, "roof", -1) for p in ps]
        placements = [p for p, _, _ in owners]
        rep = _check(placements)
        occ = np.zeros(grid_shape, dtype=bool)
        for p in placements:
            occ[p.y_plate:p.y_plate + p.part.height_units,
                p.x_stud:p.x_stud + p.w_world, p.z_stud:p.z_stud + p.d_world] = True
        floating = {id(p) for p in rep.floating}
        hit = 0
        for p, kind, i in owners:
            if id(p) not in floating or p.y_plate == 0:
                continue
            under = occ[p.y_plate - 1, p.x_stud:p.x_stud + p.w_world, p.z_stud:p.z_stud + p.d_world]
            if under.any():
                continue
            if kind == "roof":
                # a gable-end brick or an eave part over the void: the
                # roof is rebuilt without that cell (its end support then
                # moves to the first column that stands on the building)
                plan.roof_avoid[p.y_plate:p.y_plate + p.part.height_units,
                                p.x_stud:p.x_stud + p.w_world, p.z_stud:p.z_stud + p.d_world] = True
                hit += 1
            elif kind == "wall":
                w = plan.walls[i]
                f = w.frame
                c = (p.y_plate - f.y0) // COURSE
                if not (0 <= c < w.mask.shape[1]):
                    continue
                for dx in range(p.w_world):
                    for dz in range(p.d_world):
                        u = (p.x_stud + dx - f.origin) if f.axis_x else (p.z_stud + dz - f.origin)
                        if 0 <= u < f.length and w.mask[u, c]:
                            w.mask[u, c] = False
                            hit += 1
            elif kind == "floor":
                fl = plan.floors[i]
                cells = fl.cells
                keep = ~((cells[:, 0] >= p.x_stud) & (cells[:, 0] < p.x_stud + p.w_world)
                         & (cells[:, 1] >= p.z_stud) & (cells[:, 1] < p.z_stud + p.d_world))
                hit += int((~keep).sum())
                fl.cells = cells[keep]
            elif kind == "pier":
                pr = plan.piers[i]
                c = (p.y_plate - pr.y0) // COURSE
                if c == 0:
                    pr.n_courses = 0
                    hit += 1
        removed += hit
        plan.floors = [fl for fl in plan.floors if len(fl.cells) >= 2]
        plan.piers = [pr for pr in plan.piers if pr.n_courses > 0]
        if hit == 0:
            break
    return removed


def build_set(plan: SetPlan, claimed: np.ndarray, target: Optional[MeshTarget] = None
              ) -> tuple[list[Placement], dict[str, int]]:
    """Bases and the base plate, then walls, then roofs, each marking the
    shared `claimed` mask so the later ones build around (and on) the
    earlier."""
    out: list[Placement] = []
    counts: dict[str, int] = {}

    def claim(ps: list[Placement]) -> None:
        H, X, Z = claimed.shape
        for p in ps:
            y1 = min(H, p.y_plate + p.part.height_units)
            x1 = min(X, p.x_stud + p.w_world)
            z1 = min(Z, p.z_stud + p.d_world)
            if p.y_plate < y1 and p.x_stud < x1 and p.z_stud < z1:
                claimed[p.y_plate:y1, p.x_stud:x1, p.z_stud:z1] = True

    if target is not None:
        for b in plan.bases:
            ps = build_base(b, target)
            claim(ps)
            out.extend(ps)
            counts["foundation:BaseRecipe"] = counts.get("foundation:BaseRecipe", 0) + len(ps)
    # The base plate before anything stands on it: the walls, piers and
    # steps then build from its top (they skip the cells it claimed), so
    # everything on the ground is one assembly through it, as on a set's
    # baseplate.
    from .massing import build_floor
    for fl in plan.floors:
        if fl.base:
            ps = build_floor(fl, claimed, claimed.shape)
            claim(ps)
            out.extend(ps)
            counts["floor:FloorRecipe"] = counts.get("floor:FloorRecipe", 0) + len(ps)
    for w in plan.walls:
        ps = build_wall(w, claimed)
        claim(ps)
        out.extend(ps)
        counts["planar_wall:WallRecipe"] = counts.get("planar_wall:WallRecipe", 0) + len(ps)
    if plan.piers:
        from .massing import build_pier
        for pier in plan.piers:
            ps = build_pier(pier, claimed, claimed.shape)
            claim(ps)
            out.extend(ps)
            counts["pier:PierRecipe"] = counts.get("pier:PierRecipe", 0) + len(ps)
    if plan.floors:
        for fl in plan.floors:
            if fl.base:
                continue                                     # built first
            ps = build_floor(fl, claimed, claimed.shape)
            claim(ps)
            out.extend(ps)
            counts["floor:FloorRecipe"] = counts.get("floor:FloorRecipe", 0) + len(ps)
    if plan.decks:
        from .massing import build_deck
        for d in plan.decks:
            ps = build_deck(d, claimed, claimed.shape)
            claim(ps)
            out.extend(ps)
            counts["deck:DeckRecipe"] = counts.get("deck:DeckRecipe", 0) + len(ps)
    def footprint_mask(g: GableSpec) -> np.ndarray:
        H, X, Z = claimed.shape
        m = np.zeros((X, Z), dtype=bool)
        faces = [g.a] + ([g.b] if g.b is not None else [])
        lo = int(np.floor(min(min(f.eave, f.ridge) for f in faces)))
        hi = int(np.ceil(max(max(f.eave, f.ridge) for f in faces)))
        r0 = min(f.span[0] for f in faces)
        r1 = max(f.span[1] for f in faces)
        if g.a.axis == "x":
            m[max(0, lo):hi, max(0, r0):r1] = True
        else:
            m[max(0, r0):r1, max(0, lo):hi] = True
        return m

    if plan.sculpt is not None:
        from .massing import build_stand, sculpt_solid
        if plan.stand is not None:
            ps = build_stand(plan.stand, claimed, plan.stand_colour, ground_y=plan.ground_y)
            out.extend(ps)
            counts["stand:StandRecipe"] = len(ps)
        ps = sculpt_solid(plan.sculpt, claimed, plan.palette_idx, ground_y=plan.ground_y)
        out.extend(ps)
        counts["sculpt:SculptRecipe"] = len(ps)
    for g in plan.gables:
        others = None
        if plan.surface is not None and len(plan.gables) > 1:
            others = np.zeros(claimed.shape[1:], dtype=bool)
            for o in plan.gables:
                if o is not g:
                    others |= footprint_mask(o)
        ps = build_gable(g, claimed, claimed.shape, surface=plan.surface, solid=plan.solid,
                         others=others, avoid=plan.roof_avoid)
        claim(ps)
        out.extend(ps)
        key = "planar_roof:GableRecipe" if g.b is not None else "planar_roof:LeanToRecipe"
        counts[key] = counts.get(key, 0) + len(ps)
    return out, counts
