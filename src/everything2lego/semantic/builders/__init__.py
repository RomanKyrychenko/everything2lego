"""Per-region specialized builders.

Each builder converts a `Region` into a list of `pack.Placement`. Builders
are dispatched by RegionLabel; add a builder by placing it in this package
and updating `BUILDERS`.
"""

from __future__ import annotations

from typing import Callable

from ..regions import Region, RegionLabel
from ...pack import Placement
from .foundation import build_foundation

Builder = Callable[[Region], list[Placement]]

BUILDERS: dict[RegionLabel, Builder] = {
    RegionLabel.FOUNDATION: build_foundation,
    # TODO: RegionLabel.WALL: build_wall,
    # TODO: RegionLabel.ROOF: build_roof,
    # TODO: RegionLabel.OPENING: build_opening,
    # TODO: RegionLabel.DETAIL: build_detail,
}


def build(region: Region) -> list[Placement]:
    """Dispatch a Region to its builder."""
    fn = BUILDERS.get(region.label)
    if fn is None:
        return []  # unhandled label — silent fallback while the pipeline is scaffolded
    return fn(region)


__all__ = ["Builder", "BUILDERS", "build", "build_foundation"]
