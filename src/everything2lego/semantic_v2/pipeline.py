"""SemanticLegoPipeline — orchestrator for the v2 flow.

    mesh path → MeshTarget → RegionDetector → [Region] → BuilderRegistry
              → [Placement] → Assembly

Assembly is later passed to the shared exporters (io_ldraw, io_stl,
viewer_export, instructions).
"""

from __future__ import annotations

import warnings

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

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
    # Optional part labels (see parts.py): a colour-coded mesh from a part
    # segmenter such as SegviGen, or painted by hand, next to the input
    # mesh; or "builtin" to segment by colour, normal and connectivity on
    # the CPU.  Without either, regions come from the facet detector.
    parts_path: Optional[str] = None
    segmenter: str = "none"          # "none" | "builtin"
    # Optional: name the parts with a vision model (see labeller.py).  Needs
    # the anthropic package and an API key; without them the parts stay
    # unnamed and the pipeline carries on.  Turns the built-in segmenter on
    # when there is nothing else to name.
    label_parts: bool = False
    labeller_model: str = "claude-opus-5-5"
    labeller_hint: str = ""           # what the object is, if the user knows
    labeller_client: Any = None       # a fake in tests; the SDK's by default
    labeller_debug_dir: Optional[str] = None


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
        self.last_naming = None       # what the vision labeller said, when it ran
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
        labels = None
        if self.config.parts_path:
            from ..voxelize import load_mesh
            from .parts import load_part_labels
            # Face order survives the alignment `from_path` applies, so the
            # labels are matched against the mesh as loaded.
            labels = load_part_labels(self.config.parts_path, load_mesh(mesh_path))
        return self.run_on_target(target, labels=labels)

    def _regions(self, target: MeshTarget, labels) -> list[Region]:
        """Regions from part labels when given (or from the built-in
        segmenter when asked for), the facet detector otherwise."""
        from .parts import regions_from_labels, segment_builtin
        cfg = self.config
        if labels is None and (cfg.segmenter == "builtin" or cfg.label_parts):
            labels = segment_builtin(target.mesh)
        if labels is not None and cfg.label_parts:
            from . import labeller
            try:
                naming = labeller.name_parts(
                    target.mesh, labels, client=cfg.labeller_client, model=cfg.labeller_model,
                    hint=cfg.labeller_hint, debug_dir=cfg.labeller_debug_dir)
            except labeller.LabellerError as exc:
                warnings.warn(f"part naming skipped: {exc}", RuntimeWarning, stacklevel=2)
            else:
                labeller.apply(labels, naming)
                target.metadata["naming"] = naming
                self.last_naming = naming
        if labels is not None:
            regions = regions_from_labels(target, labels)
            if regions:
                return regions
        return self.detector.detect(target)

    def run_on_target(self, target: MeshTarget, labels=None) -> PipelineResult:
        regions = self._regions(target, labels)
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
