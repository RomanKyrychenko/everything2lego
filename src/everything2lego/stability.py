"""Geometric stud-connection checks, not a simulation of strength or tipping.

`check` answers two kinds of question.  The hard ones decide `ok`: is every
part held, does nothing overlap, is the model one assembly.  The advisory
ones catch the ways a geometrically connected model still fails in the
hand: a part whose weight hangs far outside the studs that hold it
(`cantilevers`), a vertical joint that runs straight through several
courses with nothing bridging it (`running_seams`), and a group of parts
that the rest of the model holds through one stud (`lift_groups`).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from .pack import Placement, slope_world_low

# A part's centre may sit this far (in studs, horizontally) outside the box of
# the studs holding it before it counts as a cantilever.
CANTILEVER_STUDS = 2.0
# A seam that runs through this many courses without a part bridging it.
RUNNING_SEAM_COURSES = 3
PLATES_PER_COURSE = 3


@dataclass(frozen=True)
class Seam:
    """A vertical joint between side-by-side parts that continues through
    several courses.  `axis` is "x" when the joint separates studs x and
    x+1 (a plane of constant x, running along z) and "z" for the other
    direction; the plane starts at stud (x, z) and runs `span` studs along
    the other axis; `y_lo`..`y_hi` are the plate layers it covers
    (inclusive), `courses` how many distinct part pairs stack along it,
    and `parts` the parts on both sides."""
    axis: str
    x: int
    z: int
    y_lo: int
    y_hi: int
    courses: int
    span: int = 1
    parts: tuple[Placement, ...] = ()

    @property
    def line(self) -> str:
        """The joint as text: "x=3|4 z=0..5 y=2..10" for the plane between
        studs x=3 and x=4, from z=0 to z=5."""
        if self.axis == "x":
            return f"x={self.x}|{self.x + 1} z={self.z}..{self.z + self.span - 1} y={self.y_lo}..{self.y_hi}"
        return f"z={self.z}|{self.z + 1} x={self.x}..{self.x + self.span - 1} y={self.y_lo}..{self.y_hi}"


@dataclass
class StabilityReport:
    total: int
    supported: int
    floating: list[Placement]
    components: int = 0
    overlaps: list[tuple[int, int]] = field(default_factory=list)
    weak_connections: int = 0
    weak: list[Placement] = field(default_factory=list)   # held by a single stud
    # Advisory rules: reported, never a reason for `ok` to fail.
    cantilevers: list[Placement] = field(default_factory=list)
    cantilever_count: int = 0
    running_seams: list[Seam] = field(default_factory=list)
    running_seam_count: int = 0
    lift_groups: list[list[Placement]] = field(default_factory=list)  # hang by one stud
    lift_group_count: int = 0

    @property
    def ok(self) -> bool:
        return self.total > 0 and not self.floating and not self.overlaps and self.components == 1

    @property
    def warnings(self) -> dict[str, int]:
        """Advisory counts: what to review even when `ok` holds."""
        return {
            "single_stud_overhangs": self.weak_connections,
            "cantilevers": self.cantilever_count,
            "running_seams": self.running_seam_count,
            "single_stud_groups": self.lift_group_count,
        }


def _footprint_studs(p: Placement) -> set[tuple[int, int]]:
    return {(p.x_stud + dx, p.z_stud + dz)
            for dx in range(p.w_world) for dz in range(p.d_world)}


def _high_side(p: Placement) -> set[tuple[int, int]]:
    """The one-stud-deep strip of a slope's footprint opposite its low side
    (the corner cell for a corner slope): the only part of a sloped brick
    that is full height."""
    low = slope_world_low(p) or "+x"
    cells = _footprint_studs(p)
    x0, x1 = p.x_stud, p.x_stud + p.w_world - 1
    z0, z1 = p.z_stud, p.z_stud + p.d_world - 1
    if "+x" in low:
        cells = {c for c in cells if c[0] == x0}
    elif "-x" in low:
        cells = {c for c in cells if c[0] == x1}
    if "+z" in low:
        cells = {c for c in cells if c[1] == z0}
    elif "-z" in low:
        cells = {c for c in cells if c[1] == z1}
    return cells


def top_studs(p: Placement) -> set[tuple[int, int]]:
    """Cells on which this part offers studs to the part above: none on a
    tile or a cheese slope, only the high strip of a slope, the whole top
    of everything else (inverted slopes, windows, doors and panels have
    studded tops)."""
    shape = p.part.shape
    if shape == "tile":
        return set()
    if shape in ("slope", "slope_corner"):
        if p.part.height_units < 3 and p.w_world * p.d_world == 1:
            return set()
        return _high_side(p)
    return _footprint_studs(p)


def bottom_cells(p: Placement) -> set[tuple[int, int]]:
    """Cells through which this part can grip studs below it: an inverted
    slope only has tubes under its full-height strip."""
    if p.part.shape == "slope_inv":
        return _high_side(p)
    return _footprint_studs(p)


def cantilever_reach(p: Placement, support: set[tuple[int, int]]) -> float:
    """How far (studs, horizontally) the part's centre lies outside the
    bounding box of the studs it is held on; 0 when the centre is over
    them."""
    if not support:
        return 0.0
    cx = p.x_stud + (p.w_world - 1) / 2
    cz = p.z_stud + (p.d_world - 1) / 2
    xs = [s[0] for s in support]
    zs = [s[1] for s in support]
    dx = max(0.0, min(xs) - cx, cx - max(xs))
    dz = max(0.0, min(zs) - cz, cz - max(zs))
    return float(np.hypot(dx, dz))


def owner_grid(placements: list[Placement]) -> tuple[np.ndarray, tuple[int, int, int]]:
    """(Y, X, Z) array of the index of the part filling each cell, -1 where
    empty, plus the (y, x, z) offset that was subtracted so every
    coordinate is non-negative.  Later parts win overlapping cells."""
    if not placements:
        return np.full((0, 0, 0), -1, dtype=np.int32), (0, 0, 0)
    y0 = min(p.y_plate for p in placements)
    x0 = min(p.x_stud for p in placements)
    z0 = min(p.z_stud for p in placements)
    Y = max(p.y_plate + p.part.height_units for p in placements) - y0
    X = max(p.x_stud + p.w_world for p in placements) - x0
    Z = max(p.z_stud + p.d_world for p in placements) - z0
    owner = np.full((Y, X, Z), -1, dtype=np.int32)
    for i, p in enumerate(placements):
        owner[p.y_plate - y0:p.y_plate - y0 + p.part.height_units,
              p.x_stud - x0:p.x_stud - x0 + p.w_world,
              p.z_stud - z0:p.z_stud - z0 + p.d_world] = i
    return owner, (y0, x0, z0)


def running_seams(placements: list[Placement], *, min_courses: int = RUNNING_SEAM_COURSES
                  ) -> list[Seam]:
    """Vertical joints between side-by-side parts that continue, layer after
    layer, through `min_courses` or more courses with no part bridging
    them.  A course is a stretch of layers where the same two parts face
    each other across the joint; a joint between two tall parts counts a
    course per three plates as well, so two 1x2x5 bricks side by side are
    a running seam too.  Neighbouring stud faces on the same plane with
    the same layer range are one seam."""
    owner, (y0, x0, z0) = owner_grid(placements)
    seams: list[Seam] = []
    if owner.size == 0:
        return seams
    for axis in ("x", "z"):
        if axis == "x":
            a, b = owner[:, :-1, :], owner[:, 1:, :]
        else:
            a, b = owner[:, :, :-1], owner[:, :, 1:]
        joint = (a >= 0) & (b >= 0) & (a != b)       # (Y, X', Z')
        if not joint.any():
            continue
        # Every (x, z) column of faces with a joint somewhere; then the runs
        # of consecutive layers along it.  Keyed by (plane, layer range) so
        # the faces along one plane with the same run merge into one seam.
        runs: dict[tuple[int, int, int], list[tuple[int, list[tuple[int, int]]]]] = {}
        for xi, zi in zip(*np.nonzero(joint.any(axis=0))):
            col = joint[:, xi, zi]
            ys = np.flatnonzero(col)
            breaks = np.flatnonzero(np.diff(ys) > 1)
            starts = np.concatenate(([0], breaks + 1))
            ends = np.concatenate((breaks, [len(ys) - 1]))
            for s, e in zip(starts, ends):
                lo, hi = int(ys[s]), int(ys[e])
                pairs = list(zip(a[lo:hi + 1, xi, zi].tolist(), b[lo:hi + 1, xi, zi].tolist()))
                courses = 1 + sum(1 for k in range(1, len(pairs)) if pairs[k] != pairs[k - 1])
                courses = max(courses, (hi - lo + 1) // PLATES_PER_COURSE)
                if courses < min_courses:
                    continue
                plane, along = (int(xi), int(zi)) if axis == "x" else (int(zi), int(xi))
                runs.setdefault((plane, lo, hi), []).append((along, pairs))
        for (plane, lo, hi), faces in runs.items():
            faces.sort()
            group: list[tuple[int, list[tuple[int, int]]]] = []
            for face in faces + [(None, [])]:
                if group and (face[0] is None or face[0] != group[-1][0] + 1):
                    start = group[0][0]
                    idx = sorted({i for _, prs in group for pair in prs for i in pair})
                    courses = max(1 + sum(1 for k in range(1, len(prs)) if prs[k] != prs[k - 1])
                                  for _, prs in group)
                    courses = max(courses, (hi - lo + 1) // PLATES_PER_COURSE)
                    x, z = (plane, start) if axis == "x" else (start, plane)
                    seams.append(Seam(axis, x + x0, z + z0, lo + y0, hi + y0, courses,
                                      len(group), tuple(placements[i] for i in idx)))
                    group = []
                if face[0] is not None:
                    group.append(face)
    return seams


def _single_stud_groups(graph: nx.Graph, placements: list[Placement],
                        grounded: set[int]) -> list[list[int]]:
    """Groups of two or more parts that the rest of the model holds through
    exactly one stud: the subtree an articulation point cuts off, when all
    the contact between that point and the subtree is one cell.  Edges
    carry the contact count as weight `w`.  A group that reaches the
    ground stands by itself and is not reported."""
    groups: list[list[int]] = []
    disc: dict[int, int] = {}
    low: dict[int, int] = {}
    parent: dict[int, int | None] = {}
    tout: dict[int, int] = {}
    order: list[int] = []
    ground_in_subtree: dict[int, bool] = {}
    # Contact between a subtree root's parent and the whole subtree.
    link: Counter[int] = Counter()
    for root in sorted(graph.nodes, key=lambda i: (i not in grounded, placements[i].y_plate, i)):
        if root in disc:
            continue
        parent[root] = None
        disc[root] = low[root] = len(order)
        order.append(root)
        stack = [(root, iter(graph[root]))]
        while stack:
            v, it = stack[-1]
            advanced = False
            for u in it:
                if u == parent[v]:
                    continue
                if u not in disc:
                    parent[u] = v
                    disc[u] = low[u] = len(order)
                    order.append(u)
                    link[u] += graph[v][u]["w"]
                    stack.append((u, iter(graph[u])))
                    advanced = True
                    break
                # Back edge to an ancestor (or a forward edge seen from the
                # ancestor side, which the descendant handles).
                if disc[u] < disc[v]:
                    low[v] = min(low[v], disc[u])
                    # The child of u whose subtree holds v is the node on
                    # the stack right after u.
                    for k in range(len(stack) - 1):
                        if stack[k][0] == u:
                            link[stack[k + 1][0]] += graph[v][u]["w"]
                            break
            if advanced:
                continue
            stack.pop()
            tout[v] = len(order)
            ground_in_subtree[v] = ground_in_subtree.get(v, False) or v in grounded
            pv = parent[v]
            if pv is not None:
                low[pv] = min(low[pv], low[v])
                ground_in_subtree[pv] = ground_in_subtree.get(pv, False) or ground_in_subtree[v]
                # The subtree is cut off by pv alone (always so under the
                # root); it hangs when its only contact with pv is one stud.
                if low[v] >= disc[pv] and link[v] == 1 and not ground_in_subtree[v]:
                    members = order[disc[v]:tout[v]]
                    if len(members) >= 2:
                        groups.append(members)
    return groups


def check(placements: list[Placement]) -> StabilityReport:
    tops: dict[int, dict[tuple[int, int], int]] = {}
    occupied: dict[tuple[int, int, int], int] = {}
    overlaps = set()
    graph = nx.Graph()
    graph.add_nodes_from(range(len(placements)))
    for i, p in enumerate(placements):
        studs = _footprint_studs(p)
        tops.setdefault(p.y_plate + p.part.height_units, {}).update(
            {s: i for s in top_studs(p)})
        for y in range(p.y_plate, p.y_plate + p.part.height_units):
            for x, z in studs:
                cell = (y, x, z)
                if cell in occupied:
                    overlaps.add((occupied[cell], i))
                occupied[cell] = i

    grounded = set()
    weak: list[Placement] = []
    cantilevers: list[Placement] = []
    ground_level = min((p.y_plate for p in placements), default=0)
    for i in sorted(range(len(placements)), key=lambda i: placements[i].y_plate):
        p = placements[i]
        below = tops.get(p.y_plate, {})
        support = {s for s in bottom_cells(p) if s in below}
        contacts = Counter(below[s] for s in support)
        for j, n in contacts.items():
            if graph.has_edge(i, j):
                graph[i][j]["w"] += n
            else:
                graph.add_edge(i, j, w=n)
        if p.y_plate == 0 or any(j in grounded for j in contacts):
            grounded.add(i)
        if p.y_plate > 0 and len(support) == 1 and p.w_world * p.d_world > 1:
            weak.append(p)
        if p.y_plate > ground_level and support and cantilever_reach(p, support) > CANTILEVER_STUDS:
            cantilevers.append(p)

    on_ground = {i for i, p in enumerate(placements) if p.y_plate == ground_level}
    # A column of 1x1 parts can only ever hang by one stud; it is not a
    # joint a longer part could have made stronger, so it is not reported.
    lift_groups = [[placements[i] for i in g]
                   for g in _single_stud_groups(graph, placements, on_ground)
                   if len({(placements[i].x_stud, placements[i].z_stud) for i in g}) > 1
                   or any(placements[i].w_world * placements[i].d_world > 1 for i in g)]
    seams = running_seams(placements)
    return StabilityReport(
        total=len(placements), supported=len(grounded),
        floating=[p for i, p in enumerate(placements) if i not in grounded],
        components=nx.number_connected_components(graph),
        overlaps=sorted(overlaps), weak_connections=len(weak), weak=weak,
        cantilevers=cantilevers, cantilever_count=len(cantilevers),
        running_seams=seams, running_seam_count=len(seams),
        lift_groups=lift_groups, lift_group_count=len(lift_groups),
    )
