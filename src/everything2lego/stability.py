"""Geometric stud-connection checks, not a simulation of strength or tipping."""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx

from .pack import Placement, slope_world_low


@dataclass
class StabilityReport:
    total: int
    supported: int
    floating: list[Placement]
    components: int = 0
    overlaps: list[tuple[int, int]] = field(default_factory=list)
    weak_connections: int = 0
    weak: list[Placement] = field(default_factory=list)   # held by a single stud

    @property
    def ok(self) -> bool:
        return self.total > 0 and not self.floating and not self.overlaps and self.components == 1


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
    for i in sorted(range(len(placements)), key=lambda i: placements[i].y_plate):
        p = placements[i]
        below = tops.get(p.y_plate, {})
        contacts = [below[s] for s in bottom_cells(p) if s in below]
        graph.add_edges_from((i, j) for j in set(contacts))
        if p.y_plate == 0 or any(j in grounded for j in contacts):
            grounded.add(i)
        if p.y_plate > 0 and len(contacts) == 1 and p.w_world * p.d_world > 1:
            weak.append(p)
    return StabilityReport(
        total=len(placements), supported=len(grounded),
        floating=[p for i, p in enumerate(placements) if i not in grounded],
        components=nx.number_connected_components(graph),
        overlaps=sorted(overlaps), weak_connections=len(weak), weak=weak,
    )
