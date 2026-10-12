"""Bonding: break running seams by re-merging the parts on either side.

A wall whose courses all split at the same stud is a stack of independent
columns; in the hand it folds along that line.  The packer stays within a
course, so it cannot see this.  `bond` runs after packing and support
repair: wherever `stability.running_seams` reports a joint that continues
through several courses, it rewrites one course in every three along it so
that a part bridges the joint.  Two side-by-side plain parts of one
colour and one height become

- the single longer brick or plate of their combined footprint when the
  catalog has it (1x2 + 1x2 -> 1x4);
- otherwise two or three parts with the split moved off the line
  (1x4 + 1x4 -> 1x2 + 1x4 + 1x2), so the joint no longer continues into
  the courses above and below;
- for tall parts (panels, 1x2x5 bricks) that neither rule can bridge,
  the same box rebuilt from plain brick courses whose splits alternate
  (two 1x4x6 panels -> six 1x8 bricks).

Every cell is covered by exactly one part before and after, and no colour
changes; only the subdivision of the same solid differs.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Callable

from . import stability
from .bricks import Part, catalog
from .pack import Placement

_PLAIN = re.compile(r"^(Brick|Plate|Panel) \d+ x \d+(?: x \d+)?$")
_BRICK, _PLATE = 3, 1


@lru_cache(maxsize=1)
def _plain_parts() -> dict[tuple[str, int, int, int], Part]:
    """The one plain box part per (shape, height, w, d) footprint, w >= d:
    the curated entry wins, decorated variants (profiles, headlights,
    tubeless bricks, fences) are never produced or rewritten."""
    out: dict[tuple[str, int, int, int], Part] = {}
    for p in catalog():
        if p.shape in ("brick", "plate", "brick_tall", "panel") and _PLAIN.match(p.name):
            out.setdefault((p.shape, p.height_units, max(p.w, p.d), min(p.w, p.d)), p)
    return out


@lru_cache(maxsize=1)
def _plain_ids() -> frozenset[str]:
    return frozenset(p.ldraw for p in _plain_parts().values())


def plain_part(height: int, w: int, d: int) -> Part | None:
    """The plain brick (height 3) or plate (height 1) covering a `w` by
    `d` world footprint; None for any other height."""
    shape = {_BRICK: "brick", _PLATE: "plate"}.get(height)
    if shape is None:
        return None
    return _plain_parts().get((shape, height, max(w, d), min(w, d)))


def _is_plain(p: Placement) -> bool:
    return p.part.ldraw in _plain_ids()


def _make(height: int, color: int, x: int, z: int, y: int, w: int, d: int) -> Placement | None:
    part = plain_part(height, w, d)
    if part is None:
        return None
    rot = 0 if (w, d) == part.footprint else 90
    return Placement(part, color, x, z, y, w, d, rot)


def _mergeable(a: Placement, b: Placement, axis: str) -> bool:
    """Whether `a` and `b` sit side by side across the `axis` joint, with
    the same colour, height and extent along the joint, so that one box
    covers them both exactly.  Bricks and plates only pair with their own
    shape; tall parts pair with any plain part of their height."""
    if not (_is_plain(a) and _is_plain(b)):
        return False
    if a.color_idx != b.color_idx or a.y_plate != b.y_plate:
        return False
    if a.part.height_units != b.part.height_units:
        return False
    if a.part.height_units in (_BRICK, _PLATE) and a.part.shape != b.part.shape:
        return False
    if axis == "x":
        return (a.z_stud == b.z_stud and a.d_world == b.d_world
                and a.x_stud + a.w_world == b.x_stud)
    return (a.x_stud == b.x_stud and a.w_world == b.w_world
            and a.z_stud + a.d_world == b.z_stud)


def _rewrite(a: Placement, b: Placement, axis: str,
             blocked: Callable[[int, int], bool]) -> list[Placement] | None:
    """Replace the pair `a` (lower coordinate) and `b` with parts that
    bridge the joint between them.  `blocked(coord, side)` says whether a
    part boundary lies on the plane whose high-side stud is `coord` in
    the layer just below (`side` = -1) or above (`side` = +1) the pair,
    so a moved split does not line up with a neighbouring joint."""
    color, y, height = a.color_idx, a.y_plate, a.part.height_units
    if axis == "x":
        x, z, across, length = a.x_stud, a.z_stud, a.d_world, a.w_world + b.w_world
    else:
        x, z, across, length = a.x_stud, a.z_stud, a.w_world, a.d_world + b.d_world
    start = x if axis == "x" else z
    old_split = a.w_world if axis == "x" else a.d_world

    def piece(h: int, yy: int, offset: int, size: int) -> Placement | None:
        if axis == "x":
            return _make(h, color, x + offset, z, yy, size, across)
        return _make(h, color, x, z + offset, yy, across, size)

    def splits(h: int) -> list[int]:
        """Offsets where two plain parts of height `h` meet to cover the
        whole length, nearest the middle first."""
        ks = [k for k in range(1, length)
              if piece(h, y, 0, k) is not None and piece(h, y, k, length - k) is not None]
        return sorted(ks, key=lambda k: (abs(k - length / 2), k))

    if height in (_BRICK, _PLATE):
        whole = piece(height, y, 0, length)
        if whole is not None:
            return [whole]
        below = {k for k in range(1, length) if blocked(start + k, -1)}
        above = {k for k in range(1, length) if blocked(start + k, +1)}
        best: tuple[tuple[int, float], list[Placement]] | None = None
        for k in splits(height):
            if k == old_split:
                continue
            key = (int(k in below or k in above), abs(k - length / 2))
            if best is None or key < best[0]:
                best = (key, [piece(height, y, 0, k), piece(height, y, k, length - k)])
        if best is None or best[0][0]:
            # Three pieces with a bridging middle over the old split.
            for k in range(1, old_split):
                for n in range(1, length - old_split):
                    parts = [piece(height, y, 0, k), piece(height, y, k, old_split - k + n),
                             piece(height, y, old_split + n, length - old_split - n)]
                    if any(p is None for p in parts):
                        continue
                    k2 = old_split + n
                    key = (int(k in below or k in above or k2 in below or k2 in above),
                           abs(k - (length - k2)))
                    if best is None or key < best[0]:
                        best = (key, parts)
        if best is None or best[0][0]:
            return None          # every split lines up with a neighbouring joint
        return best[1]

    # A tall pair: rebuild the box as brick courses (plates on top for any
    # remainder) whose splits alternate, none continuing the joints that
    # touch the box from below or above.
    layers: list[tuple[int, int]] = []
    yy, left = y, height
    while left >= _BRICK:
        layers.append((_BRICK, yy))
        yy, left = yy + _BRICK, left - _BRICK
    while left > 0:
        layers.append((_PLATE, yy))
        yy, left = yy + _PLATE, left - _PLATE
    out: list[Placement] = []
    prev: int | None = None
    for i, (h, yy) in enumerate(layers):
        whole = piece(h, yy, 0, length)
        if whole is not None:
            out.append(whole)
            prev = None
            continue
        options = [k for k in splits(h) if k != prev]
        if i == 0:
            options = [k for k in options if not blocked(start + k, -1)] or options
        if i == len(layers) - 1:
            options = [k for k in options if not blocked(start + k, +1)] or options
        if not options:
            return None
        k = options[0]
        out.extend([piece(h, yy, 0, k), piece(h, yy, k, length - k)])
        prev = k
    return out


def seam_layers(seams: list[stability.Seam]) -> int:
    """The total face area of seams, in stud-layers: the measure that keeps
    falling as seams are broken, where the count can rise when one long
    seam is cut into two shorter ones."""
    return sum(s.span * (s.y_hi - s.y_lo + 1) for s in seams)


def bond(placements: list[Placement], *, max_rounds: int = 12) -> tuple[list[Placement], dict]:
    """Break every running seam that plain parts can bridge.  Returns the
    new placements and stats: seams (count and stud-layer area) before and
    after, parts merged into one longer part, splits moved, tall pairs
    rebuilt as courses, rounds run.  Order of untouched parts is kept;
    rewritten parts take the place of the first part they replace."""
    placements = list(placements)
    seams = stability.running_seams(placements)
    stats = {"running_seams_before": len(seams), "running_seam_layers_before": seam_layers(seams),
             "merged": 0, "shifted": 0, "rebuilt": 0, "rounds": 0}
    for _ in range(max_rounds):
        if not seams:
            break
        stats["rounds"] += 1
        owner, (oy, ox, oz) = stability.owner_grid(placements)
        Y, X, Z = owner.shape

        def joint_at(axis: str, coord: int, along: tuple[int, int], y: int) -> bool:
            """Whether some part boundary lies on the plane between studs
            `coord - 1` and `coord` at layer `y`, within the along range."""
            if not 0 <= y < Y:
                return False
            if axis == "x":
                c = coord - ox
                if not 1 <= c < X:
                    return False
                lo, hi = max(0, along[0] - oz), min(Z, along[1] - oz + 1)
                left, right = owner[y, c - 1, lo:hi], owner[y, c, lo:hi]
            else:
                c = coord - oz
                if not 1 <= c < Z:
                    return False
                lo, hi = max(0, along[0] - ox), min(X, along[1] - ox + 1)
                left, right = owner[y, lo:hi, c - 1], owner[y, lo:hi, c]
            return bool(((left >= 0) & (right >= 0) & (left != right)).any())

        replacements: dict[int, list[Placement]] = {}
        for seam in seams:
            # The (a, b) pairs facing each other across the seam, keyed by
            # the first layer they appear in; pairs starting at the same
            # layer share a course.
            first: dict[tuple[int, int], int] = {}
            for y in range(seam.y_lo - oy, seam.y_hi - oy + 1):
                for k in range(seam.span):
                    if seam.axis == "x":
                        xi, zi = seam.x - ox, seam.z - oz + k
                        pair = (int(owner[y, xi, zi]), int(owner[y, xi + 1, zi]))
                    else:
                        xi, zi = seam.x - ox + k, seam.z - oz
                        pair = (int(owner[y, xi, zi]), int(owner[y, xi, zi + 1]))
                    first.setdefault(pair, y)
            levels = sorted(set(first.values()))
            course_of = {pair: levels.index(y) for pair, y in first.items()}
            # Rewriting one course in three leaves runs of at most two.
            # Try the second course first so the ends keep their parts;
            # when a course cannot be rewritten, the next one is tried.
            order = sorted(first, key=lambda pr: ((course_of[pr] % 3 != 1), course_of[pr]))
            fixed: list[int] = []
            for pair in order:
                ci = course_of[pair]
                if any(0 < abs(ci - f) < 3 for f in fixed):
                    continue
                ia, ib = pair
                if ia in replacements or ib in replacements:
                    continue
                a, b = placements[ia], placements[ib]
                if not _mergeable(a, b, seam.axis):
                    continue
                y_lo, y_hi = a.y_plate - oy, a.y_plate - oy + a.part.height_units - 1
                if seam.axis == "x":
                    along = (a.z_stud, a.z_stud + a.d_world - 1)
                else:
                    along = (a.x_stud, a.x_stud + a.w_world - 1)

                def blocked(coord: int, side: int, along=along, y_lo=y_lo, y_hi=y_hi,
                            axis=seam.axis) -> bool:
                    return joint_at(axis, coord, along, y_lo - 1 if side < 0 else y_hi + 1)

                new = _rewrite(a, b, seam.axis, blocked)
                if new is None:
                    continue
                replacements[ia] = new
                replacements[ib] = []
                fixed.append(ci)
                if len(new) == 1:
                    stats["merged"] += 1
                elif a.part.height_units in (_BRICK, _PLATE):
                    stats["shifted"] += 1
                else:
                    stats["rebuilt"] += 1
        if not replacements:
            break
        out: list[Placement] = []
        for i, p in enumerate(placements):
            out.extend(replacements.get(i, [p]))
        placements = out
        seams = stability.running_seams(placements)
    stats["running_seams_after"] = len(seams)
    stats["running_seam_layers_after"] = seam_layers(seams)
    return placements, stats
