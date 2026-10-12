"""Windows and doors found on the mesh before the walls are packed.

A photo of a house keeps its windows as darker, bluer or otherwise
off-colour rectangles on the walls.  Instead of hoping the packed bricks
happen to form a frame-sized block of one colour, each vertical wall region
is rasterised at stud-by-plate resolution from the mesh's own surface
samples, cells whose colour differs clearly from the wall's are grouped
into rectangles, and each rectangle is snapped to the nearest real window
frame or door.  The wall is then packed around the reserved cells.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import label as _label

from ..bricks import Part, full_catalog
from ..colors import _rgb_to_lab, palette
from ..pack import Placement
from .mesh_target import MeshTarget
from .regions import PlanarRegion

# Window frames and doors by (studs wide, plates tall).
FRAME_SIZES: dict[tuple[int, int], str] = {
    (2, 6): "60592.dat", (2, 9): "60800.dat", (4, 9): "60594.dat", (6, 9): "3853.dat"}
DOOR_SIZES: dict[tuple[int, int], str] = {(4, 15): "60623.dat", (4, 18): "60614.dat"}
COURSE = 3   # plates per brick course; frames and doors are whole courses tall


@dataclass(frozen=True)
class WallFrame:
    """The stud-by-plate grid of one vertical wall, as the wall builder lays
    it out: `axis_x` says whether the wall runs along x, `origin` is the
    first stud along it, `other` the stud across it, `y0` its bottom plate."""
    axis_x: bool
    origin: int
    other: int
    y0: int
    length: int
    height: int

    def placement(self, part: Part, color_idx: int, u: int, y: int, width: int) -> Placement:
        if self.axis_x:
            return Placement(part, color_idx, self.origin + u, self.other, self.y0 + y,
                             width, 1, 0)
        return Placement(part, color_idx, self.other, self.origin + u, self.y0 + y,
                         1, width, 90)


def wall_frame(region: PlanarRegion, target: MeshTarget) -> WallFrame:
    """Where the wall sits on the stud grid; the wall builder and the
    opening detector both use this, so they agree on every cell.

    The stud columns run from the first to the last stud the wall's bounds
    touch (a bound on a stud boundary still touches that stud, so a corner
    column belongs to both walls that meet there), and the plates from the
    lowest bound to the highest occupied plate in those columns: the mesh's
    top maps to a fraction of a plate, the voxel grid to a whole one.
    """
    mn = target.mm_point_to_voxel(region.bounds[0])
    mx = target.mm_point_to_voxel(region.bounds[1])
    span_x = float(abs(mx[0] - mn[0]))
    span_z = float(abs(mx[2] - mn[2]))
    axis_x = span_x >= span_z
    occupied = target.voxel_grid.occupied()
    H, X, Z = occupied.shape
    lo_u, hi_u = ((min(mn[0], mx[0]), max(mn[0], mx[0])) if axis_x
                  else (min(mn[2], mx[2]), max(mn[2], mx[2])))
    eps = 1e-4   # a bound meant for a stud boundary may come back a hair short
    origin = int(np.floor(lo_u + eps))
    last = min((X if axis_x else Z) - 1, int(np.floor(hi_u + eps)))
    length = max(1, last - origin + 1)
    other = min((Z if axis_x else X) - 1, int(np.floor((mn[2] if axis_x else mn[0]) + eps)))
    y0 = max(0, int(np.floor(min(mn[1], mx[1]) + eps)))
    columns = occupied[:, origin:origin + length, other] if axis_x \
        else occupied[:, other, origin:origin + length]
    rows = np.flatnonzero(columns[y0:].any(axis=1))
    height = int(rows.max()) + 1 if len(rows) else max(1, int(np.ceil(max(mn[1], mx[1]))) - y0)
    return WallFrame(axis_x, origin, other, y0, length, height)


def wall_colour_grid(region: PlanarRegion, target: MeshTarget, frame: WallFrame,
                     *, plane_tolerance_mm: float | None = None
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Mean LAB colour of the mesh surface in every (stud, plate) cell of the
    wall, and a mask of the cells that have samples.  Samples are the
    target's surface points within a little of the wall's plane."""
    pts = target.surface_points
    n = region.plane_normal / (np.linalg.norm(region.plane_normal) + 1e-9)
    dist = np.abs((pts - region.plane_origin) @ n)
    if plane_tolerance_mm is None:
        # Half a stud, in mesh millimetres.
        plane_tolerance_mm = 0.5 / max(1e-6, float(target.mesh_to_voxel[0, 0]))
    lo, hi = region.bounds
    pad = plane_tolerance_mm
    inside = ((dist <= plane_tolerance_mm)
              & (pts >= lo - pad).all(axis=1) & (pts <= hi + pad).all(axis=1))
    idx = np.flatnonzero(inside)
    lab_sum = np.zeros((frame.length, frame.height, 3), dtype=np.float64)
    count = np.zeros((frame.length, frame.height), dtype=np.int64)
    if len(idx):
        vox = np.hstack([pts[idx], np.ones((len(idx), 1), np.float32)]) @ target.mesh_to_voxel.T
        u = np.floor(vox[:, 0] if frame.axis_x else vox[:, 2]).astype(int) - frame.origin
        y = np.floor(vox[:, 1]).astype(int) - frame.y0
        ok = (u >= 0) & (u < frame.length) & (y >= 0) & (y < frame.height)
        lab = _rgb_to_lab(target.surface_colors[idx[ok]].astype(np.float32) / 255.0)
        np.add.at(lab_sum, (u[ok], y[ok]), lab)
        np.add.at(count, (u[ok], y[ok]), 1)
    has = count > 0
    mean = np.zeros_like(lab_sum)
    mean[has] = lab_sum[has] / count[has][:, None]
    return mean, has


def _snap(width: int, height: int, bottom: int, wall_height: int
          ) -> tuple[tuple[int, int], str] | None:
    """The frame or door nearest to a patch `width` studs by `height` plates
    whose bottom is `bottom` plates above the wall's foot."""
    near_ground = bottom <= COURSE
    sizes = DOOR_SIZES if near_ground and height >= 12 else FRAME_SIZES
    best = None
    for (w, h), ldraw in sizes.items():
        if w > width + 1 or h > height + COURSE or h > wall_height:
            continue
        cost = abs(w - width) + abs(h - height) / COURSE
        if best is None or cost < best[0]:
            best = (cost, (w, h), ldraw)
    return (best[1], best[2]) if best else None


def largest_rectangle(mask: np.ndarray) -> tuple[int, int, int, int, int]:
    """The largest all-True axis-aligned rectangle in a 2-D mask, as
    (area, u0, u1, y0, y1); the histogram-and-stack method, one pass per
    row of the first axis."""
    L, H = mask.shape
    heights = np.zeros(H, dtype=int)
    best = (0, 0, 0, 0, 0)
    for u in range(L):
        heights = np.where(mask[u], heights + 1, 0)
        stack: list[int] = []
        for y in range(H + 1):
            h = heights[y] if y < H else 0
            start = y
            while stack and heights[stack[-1]] >= h:
                top = stack.pop()
                start = top
                width = y - top
                area = heights[top] * width
                if area > best[0]:
                    best = (area, u - heights[top] + 1, u + 1, top, y)
            stack.append(start)
            heights[start] = h
    return best


def rectangles(cells: np.ndarray, min_width: int = 2, min_height: int = 2 * COURSE
               ) -> list[tuple[int, int, int, int]]:
    """Split one blob of off-colour cells into the rectangles it is made
    of.  Two openings whose colours bleed into the gap between them join
    into one blob; peeling off the largest filled rectangle, then the
    next, recovers each opening and leaves the thin bleed slivers, which
    are below `min_width` by `min_height` and dropped.  Returns
    (u0, u1, y0, y1) boxes."""
    rest = cells.copy()
    out = []
    while True:
        area, u0, u1, y0, y1 = largest_rectangle(rest)
        if area == 0 or (u1 - u0 < min_width and y1 - y0 < min_height):
            break
        u0, u1, y0, y1 = _grow(rest, u0, u1, y0, y1)
        rest[u0:u1, y0:y1] = False
        if u1 - u0 >= min_width and y1 - y0 >= min_height:
            out.append((u0, u1, y0, y1))
    return out


def _grow(cells: np.ndarray, u0: int, u1: int, y0: int, y1: int,
          fill: float = 0.5) -> tuple[int, int, int, int]:
    """Extend a core rectangle over its ragged edges: a neighbouring column
    or row joins while at least `fill` of it, across the rectangle's span,
    is off-colour.  Sampling leaves an opening's edge cells half coloured,
    and without this the same window comes out a stud narrower on one run
    than on the next."""
    L, H = cells.shape
    changed = True
    while changed:
        changed = False
        if u0 > 0 and cells[u0 - 1, y0:y1].mean() >= fill:
            u0 -= 1; changed = True
        if u1 < L and cells[u1, y0:y1].mean() >= fill:
            u1 += 1; changed = True
        if y0 > 0 and cells[u0:u1, y0 - 1].mean() >= fill:
            y0 -= 1; changed = True
        if y1 < H and cells[u0:u1, y1].mean() >= fill:
            y1 += 1; changed = True
    return u0, u1, y0, y1


def find_openings(region: PlanarRegion, target: MeshTarget, *,
                  min_delta_e: float = 22.0, min_fill: float = 0.55,
                  frame: WallFrame | None = None) -> list[Placement]:
    """Window frames and doors for the off-colour rectangles on a wall.

    Each rectangle of cells whose colour is at least `min_delta_e` (CIE76)
    from the wall's median colour (blobs peeled into their rectangles,
    see `rectangles`), at least 2 studs by 2 courses and filled to `min_fill`,
    becomes the frame or door nearest its size, centred on it and aligned
    to the wall's brick courses.
    """
    frame = frame or wall_frame(region, target)
    if frame.length < 2 or frame.height < 2 * COURSE:
        return []
    mean, has = wall_colour_grid(region, target, frame)
    if has.sum() < 4:
        return []
    wall_lab = np.median(mean[has], axis=0)
    delta = np.linalg.norm(mean - wall_lab, axis=2)
    off = has & (delta >= min_delta_e)
    labels, n = _label(off)
    parts = {p.ldraw: p for p in full_catalog()}
    pal_lab = _rgb_to_lab(np.array([c.rgb for c in palette()], np.float32) / 255.0)
    out: list[Placement] = []
    taken = np.zeros_like(off)
    patches = []
    for u0, u1, y0, y1 in (box for k in range(1, n + 1) for box in rectangles(labels == k)):
        cells = np.zeros_like(off)
        cells[u0:u1, y0:y1] = off[u0:u1, y0:y1]
        width, height = u1 - u0, y1 - y0
        if width < 2 or height < 2 * COURSE or cells.sum() < min_fill * width * height:
            continue
        colour = int(np.argmin(np.linalg.norm(pal_lab - mean[cells].mean(axis=0), axis=1)))
        patches.append(((u0 + u1) / 2, (y0 + y1) / 2, width, height, y0, colour))
    for cu, cy, width, height, y0, colour in equalise(patches, length=frame.length):
        snapped = _snap(width, height, y0, frame.height)
        if snapped is None:
            continue
        (w, h), ldraw = snapped
        u = max(0, min(frame.length - w, int(round(cu - w / 2))))
        y = int(round((cy - h / 2) / COURSE)) * COURSE
        y = max(0, min(frame.height - h, y))
        if taken[u:u + w, y:y + h].any():
            continue
        taken[u:u + w, y:y + h] = True
        out.append(frame.placement(parts[ldraw], colour, u, y, w))
    return out


Patch = tuple[float, float, int, int, int, int]   # centre u, centre y, width, height, bottom, colour


def equalise(patches: list[Patch], *, row_tolerance: float = COURSE,
             size_tolerance: float = 2.0, length: float | None = None) -> list[Patch]:
    """Make repeated openings identical.  Patches whose centres sit on the
    same row and whose sizes are within `size_tolerance` are one kind of
    window: each takes the kind's median width, height and row, so the
    sampling noise that makes one window a stud wider than its twin is
    gone before any frame is chosen.  Patches that mirror each other
    about the wall's centre keep their own columns."""
    groups: list[list[Patch]] = []
    for p in sorted(patches, key=lambda q: (q[1], q[0])):
        for g in groups:
            ref = g[0]
            if (abs(p[1] - ref[1]) <= row_tolerance and abs(p[2] - ref[2]) <= size_tolerance
                    and abs(p[3] - ref[3]) <= size_tolerance * COURSE):
                g.append(p)
                break
        else:
            groups.append([p])
    out: list[Patch] = []
    for g in groups:
        width = int(round(float(np.median([q[2] for q in g]))))
        height = int(round(float(np.median([q[3] for q in g]))))
        cy = float(np.median([q[1] for q in g]))
        y0 = int(round(float(np.median([q[4] for q in g]))))
        centres = [q[0] for q in g]
        if length is not None:
            centres = _mirror(centres, length)
        out.extend((cu, cy, width, height, y0, q[5]) for cu, q in zip(centres, g))
    return out


def _mirror(centres: list[float], length: float, tolerance: float = 1.0) -> list[float]:
    """Openings that face each other across the wall's middle are placed
    exactly symmetric; the wall's own centre line is the mirror."""
    mid = length / 2
    out = list(centres)
    used = set()
    for i, a in enumerate(centres):
        if i in used:
            continue
        for j in range(i + 1, len(centres)):
            if j in used:
                continue
            if abs((a - mid) + (centres[j] - mid)) <= tolerance:
                half = (abs(a - mid) + abs(centres[j] - mid)) / 2
                out[i] = mid - half if a < mid else mid + half
                out[j] = mid + half if a < mid else mid - half
                used.update((i, j))
                break
    return out


def reserved_cells(openings: list[Placement], frame: WallFrame) -> np.ndarray:
    """(length, height) mask of the wall cells the openings occupy."""
    mask = np.zeros((frame.length, frame.height), dtype=bool)
    for p in openings:
        u = (p.x_stud if frame.axis_x else p.z_stud) - frame.origin
        w = p.w_world if frame.axis_x else p.d_world
        y = p.y_plate - frame.y0
        mask[u:u + w, y:y + p.part.height_units] = True
    return mask


def forced_openings(region: PlanarRegion, target: MeshTarget, frame: WallFrame,
                    boxes: list[dict]) -> list[Placement]:
    """Openings a part segmentation named on this wall, as frame and door
    placements: each box (mesh millimetres, with the part's normal) that
    lies in the wall's plane is projected onto the wall's stud-by-plate
    grid and snapped to the nearest frame or door like a detected one."""
    n = region.plane_normal / (np.linalg.norm(region.plane_normal) + 1e-9)
    parts = {p.ldraw: p for p in full_catalog()}
    pal_lab = _rgb_to_lab(np.array([c.rgb for c in palette()], np.float32) / 255.0)
    out: list[Placement] = []
    taken = np.zeros((frame.length, frame.height), dtype=bool)
    tolerance = 0.75 / max(1e-6, float(target.mesh_to_voxel[0, 0]))   # 3/4 stud, in mm
    for box in boxes:
        bn = np.asarray(box["normal"], dtype=np.float32)
        if abs(float(np.dot(bn, n))) < 0.9:
            continue
        mn, mx = np.asarray(box["min"], np.float32), np.asarray(box["max"], np.float32)
        centre = (mn + mx) / 2
        if abs(float(np.dot(centre - region.plane_origin, n))) > tolerance:
            continue
        lo, hi = target.mm_point_to_voxel(mn), target.mm_point_to_voxel(mx)
        u0 = int(np.floor((min(lo[0], hi[0]) if frame.axis_x else min(lo[2], hi[2])) + 1e-4)) - frame.origin
        u1 = int(np.ceil((max(lo[0], hi[0]) if frame.axis_x else max(lo[2], hi[2])) - 1e-4)) - frame.origin
        y0 = int(np.floor(min(lo[1], hi[1]) + 1e-4)) - frame.y0
        y1 = int(np.ceil(max(lo[1], hi[1]) - 1e-4)) - frame.y0
        u0, u1 = max(0, u0), min(frame.length, u1)
        y0, y1 = max(0, y0), min(frame.height, y1)
        if u1 - u0 < 1 or y1 - y0 < 1:
            continue
        snapped = _snap(u1 - u0, y1 - y0, y0, frame.height)
        if snapped is None:
            continue
        (w, h), ldraw = snapped
        if box.get("name") == "door":
            door = _snap(u1 - u0, max(y1 - y0, 15), 0, frame.height)
            if door is not None:
                (w, h), ldraw = door
                y0 = 0
        u = max(0, min(frame.length - w, int(round((u0 + u1) / 2 - w / 2))))
        y = int(round(((y0 + y1) / 2 - h / 2) / COURSE)) * COURSE
        y = max(0, min(frame.height - h, y))
        if taken[u:u + w, y:y + h].any():
            continue
        taken[u:u + w, y:y + h] = True
        rgb = box.get("colour") or (60, 60, 60)
        colour = int(np.argmin(np.linalg.norm(pal_lab - _rgb_to_lab(np.array(rgb, np.float32)[None] / 255.0)[0], axis=1)))
        out.append(frame.placement(parts[ldraw], colour, u, y, w))
    return out
