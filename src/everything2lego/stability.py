"""Geometric stud-connection checks, not a simulation of strength or tipping."""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx

from .pack import Placement


@dataclass
class StabilityReport:
    total: int
    supported: int
    floating: list[Placement]
    components: int = 0
    overlaps: list[tuple[int, int]] = field(default_factory=list)
    weak_connections: int = 0

    @property
    def ok(self) -> bool:
        return self.total > 0 and not self.floating and not self.overlaps and self.components == 1


def _footprint_studs(p: Placement) -> set[tuple[int, int]]:
    return {(p.x_stud + dx, p.z_stud + dz)
            for dx in range(p.w_world) for dz in range(p.d_world)}


def check(placements: list[Placement]) -> StabilityReport:
    tops: dict[int, dict[tuple[int, int], int]] = {}
    occupied: dict[tuple[int, int, int], int] = {}
    overlaps = set()
    graph = nx.Graph()
    graph.add_nodes_from(range(len(placements)))
    for i, p in enumerate(placements):
        studs = _footprint_studs(p)
        if p.part.shape != "tile":
            tops.setdefault(p.y_plate + p.part.height_units, {}).update({s: i for s in studs})
        for y in range(p.y_plate, p.y_plate + p.part.height_units):
            for x, z in studs:
                cell = (y, x, z)
                if cell in occupied:
                    overlaps.add((occupied[cell], i))
                occupied[cell] = i

    grounded = set()
    weak = 0
    for i in sorted(graph, key=lambda i: placements[i].y_plate):
        p = placements[i]
        below = tops.get(p.y_plate, {})
        contacts = [below[s] for s in _footprint_studs(p) if s in below]
        graph.add_edges_from((i, j) for j in set(contacts))
        if p.y_plate == 0 or any(j in grounded for j in contacts):
            grounded.add(i)
        if p.y_plate > 0 and len(contacts) == 1 and p.w_world * p.d_world > 1:
            weak += 1
    return StabilityReport(
        total=len(placements), supported=len(grounded),
        floating=[p for i, p in enumerate(placements) if i not in grounded],
        components=nx.number_connected_components(graph),
        overlaps=sorted(overlaps), weak_connections=weak,
    )
