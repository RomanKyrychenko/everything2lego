"""SemanticLegoPipeline — orchestrator for the v2 flow.

    mesh path → MeshTarget → RegionDetector → [Region] → BuilderRegistry
              → [Placement] → Assembly

Assembly is later passed to the shared exporters (io_ldraw, io_stl,
viewer_export, instructions).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ..pack import Placement
from ..bonding import bond
from ..pipeline import brace, resolve_overlaps
from .builders import BuilderRegistry, BuilderConfig
from .detector import RegionDetector, DetectorConfig
from .interior import InteriorVolumeFiller, InteriorConfig
from .mesh_target import MeshTarget
from .regions import PlanarRegion, Region


@dataclass
class PipelineConfig:
    max_studs: int = 32
    n_samples: int = 100_000
    color_samples: int = 120_000
    subvoxels: int = 1
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    builder: BuilderConfig = field(default_factory=BuilderConfig)
    interior: InteriorConfig = field(default_factory=InteriorConfig)
    fill_interior: bool = True
    # Drop region placements that overlap earlier ones (refilling what they
    # alone covered) and hold up floating parts with 1x1 pegs and columns.
    validate_structure: bool = True


@dataclass
class PipelineResult:
    placements: list[Placement]
    surface_placements: list[Placement]
    interior_placements: list[Placement]
    regions: list[Region]
    per_region_counts: dict[str, int] = field(default_factory=dict)
    dropped_overlaps: int = 0
    added_supports: int = 0


def _build_priority(region: Region) -> int:
    """Slabs before roofs before walls before the rest."""
    if isinstance(region, PlanarRegion):
        if region.is_horizontal:
            return 0
        if region.is_vertical:
            return 2
        return 1
    return 3


class SemanticLegoPipeline:
    """End-to-end mesh → placements."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self.detector = RegionDetector(self.config.detector)
        self.registry = BuilderRegistry()
        self.interior_filler = InteriorVolumeFiller(self.config.interior)
        # Rebuild each builder with the pipeline's shared config so knob
        # changes (e.g. MILP time limit) propagate.
        for b in self.registry.builders:
            b.config = self.config.builder

    def run(self, mesh_path: str | Path) -> PipelineResult:
        target = MeshTarget.from_path(
            mesh_path,
            max_studs=self.config.max_studs,
            n_samples=self.config.n_samples,
            color_samples=self.config.color_samples,
            subvoxels=self.config.subvoxels,
        )
        return self.run_on_target(target)

    def run_on_target(self, target: MeshTarget) -> PipelineResult:
        regions = self.detector.detect(target)
        surface: list[Placement] = []
        counts: dict[str, int] = {}
        # Horizontal slabs first, walls last: where a wall meets a slab the
        # slab owns the cells, and each wall is told what is already built so
        # it packs around it instead of overlapping.
        claimed = np.zeros(target.voxel_grid.occupied().shape, dtype=bool)
        for region in sorted(regions, key=_build_priority):
            builder = self.registry.dispatch(region)
            if builder is None:
                counts[f"{region.label.value}:unbuilt"] = 1
                continue
            builder.claimed = claimed
            new = builder.build(region, target)
            for p in new:
                claimed[p.y_plate:p.y_plate + p.part.height_units,
                        p.x_stud:p.x_stud + p.w_world, p.z_stud:p.z_stud + p.d_world] = True
            surface.extend(new)
            key = f"{region.label.value}:{type(builder).__name__}"
            counts[key] = counts.get(key, 0) + len(new)

        interior: list[Placement] = []
        if self.config.fill_interior:
            interior = self.interior_filler.fill(target, surface)
            counts["interior:InteriorVolumeFiller"] = len(interior)

        placements = surface + interior
        dropped = added = 0
        if self.config.validate_structure and placements:
            # Regions are built independently, so two of them can claim the
            # same cell; the surface placements come first and win.
            placements, dropped = resolve_overlaps(placements)
            placements, added = brace(placements)
            placements, _ = bond(placements)
            counts["structure:dropped_overlaps"] = dropped
            counts["structure:added_supports"] = added
        return PipelineResult(
            placements=placements,
            surface_placements=surface,
            interior_placements=interior,
            regions=regions,
            per_region_counts=counts,
            dropped_overlaps=dropped,
            added_supports=added,
        )
