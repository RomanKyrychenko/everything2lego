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
    fill_interior: bool = True       # outside set mode only (a set is hollow)
    # Drop region placements that overlap earlier ones (refilling what they
    # alone covered) and hold up floating parts with 1x1 pegs and columns.
    validate_structure: bool = True
    # Optional part labels (see parts.py): a colour-coded mesh from a part
    # segmenter such as SegviGen, or painted by hand, next to the input
    # mesh; or "builtin" to segment by colour, normal and connectivity on
    # the CPU.  Without either, regions come from the facet detector.
    parts_path: Optional[str] = None
    segmenter: str = "none"          # "none" | "builtin"
    # Built-in segmenter knobs.  Library defaults match segment_builtin's
    # own (they're right for authored meshes with small genuine parts).
    # Photo-like marching-cubes meshes want a looser normal cone (~25°, so
    # surface ripple doesn't split a wall) and a larger minimum region
    # (~0.01, so confetti merges into its neighbour); the `semantic` CLI
    # defaults to those.
    seg_normal_degrees: float = 12.0
    seg_min_fraction: float = 0.002
    # Voxel islands below this many cells are dropped before anything is
    # built (see MeshTarget.from_mesh).  0 = off; the CLI uses 8.
    min_island_cells: int = 0
    # After segmentation, fold parts below this fraction of total mesh AREA
    # into the neighbour sharing the longest boundary (parts.merge_small_
    # regions).  0 = off; the CLI uses 0.003 for photo meshes.
    merge_small_region_area: float = 0.0
    # Build walls and pitched roofs with set-design recipes (setdesign.py):
    # snapped bonded walls with part openings, slope roofs with tiles and
    # ridge, gable ends in brick.  Other regions still use the builders.
    set_mode: bool = False
    # Where set mode reads its specs: "massing" (walls, roofs and decks off
    # the solid voxel volume — robust to segmentation noise) or "regions"
    # (the detector's planar regions snapped to the grid).
    set_source: str = "massing"
    # A measured front (tools/facade_depth.py → npz of metric points and a
    # mask from MapAnything).  In massing it is registered to the model
    # and overrides the front surface and the skyline — the photographed
    # side as measured, the rest as generated.
    facade_path: Optional[str] = None
    # A figure (a shape sculpted course by course) gets a pedestal under
    # the parts of its underside too steep for inverted slopes.  Colour is
    # a palette index; None = dark bluish grey.
    stand: bool = True
    stand_colour: Optional[int] = None
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
            min_island_cells=self.config.min_island_cells,
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
            labels = segment_builtin(target.mesh,
                                     normal_degrees=cfg.seg_normal_degrees,
                                     min_fraction=cfg.seg_min_fraction)
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
        if labels is not None and cfg.merge_small_region_area > 0:
            from .parts import merge_small_regions
            before = len(labels.ids)
            labels = merge_small_regions(target.mesh, labels,
                                         min_area_frac=cfg.merge_small_region_area)
            target.metadata["merged_small_regions"] = before - len(labels.ids)
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
        remaining = sorted(regions, key=_build_priority)
        owned = None

        def run_builders(regs: list[Region]) -> None:
            for region in regs:
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

        if self.config.set_mode:
            from .setdesign import build_set, plan_set
            facade = None
            if self.config.facade_path:
                from ..facade import Facade
                facade = Facade.load(self.config.facade_path)
            plan = plan_set(regions, target,
                            detect_openings=self.config.builder.detect_openings,
                            opening_delta_e=self.config.builder.opening_delta_e,
                            source=self.config.set_source, facade=facade,
                            stand=self.config.stand, stand_colour=self.config.stand_colour)
            if plan.facade_fit is not None:
                counts["facade:cells_changed"] = int(plan.facade_changed)
                counts["facade:inliers_pct"] = int(round(100 * plan.facade_fit.inliers))
                target.metadata["facade_fit"] = {
                    "axis": plan.facade_fit.axis, "sign": plan.facade_fit.sign,
                    "mirror": plan.facade_fit.mirror, "rms_mm": plan.facade_fit.rms_mm,
                    "inliers": plan.facade_fit.inliers}
            remaining = [r for r in remaining if id(r) not in plan.consumed]
            # Slabs first: the walls stand on them and build around them.
            slabs = [r for r in remaining if _build_priority(r) == 0]
            run_builders(slabs)
            built, recipe_counts = build_set(plan, claimed, target)
            surface.extend(built)
            counts.update(recipe_counts)
            counts["set:bases"] = len(plan.bases)
            counts["set:decks"] = len(plan.decks)
            counts["set:piers"] = len(plan.piers)
            counts["set:floors"] = len(plan.floors)
            counts["set:pruned_cells"] = int(plan.pruned)
            counts["set:walls"] = len(plan.walls)
            counts["set:gables"] = sum(1 for g in plan.gables if g.b is not None)
            counts["set:lean_tos"] = sum(1 for g in plan.gables if g.b is None)
            remaining = [r for r in remaining if _build_priority(r) != 0]
            # The recipes own their volumes outright; the builders and the
            # interior filler stay out even where the voxels say "mesh".
            owned = plan.owned(claimed.shape)
            claimed |= owned
        run_builders(remaining)

        interior: list[Placement] = []
        # A set is hollow: in set mode the recipes are the whole structure
        # (base plate, walls, floors, roof; a sculpture's shell and what
        # its courses stand on), and the inside stays open as in a store
        # set.  Filling it only adds hidden bricks that stand on nothing.
        if self.config.fill_interior and not self.config.set_mode:
            interior = self.interior_filler.fill(target, surface, exclude=owned)
            counts["interior:InteriorVolumeFiller"] = len(interior)

        placements = surface + interior
        dropped = added = 0
        if self.config.validate_structure and placements:
            # Regions are built independently, so two of them can claim the
            # same cell; the surface placements come first and win.
            placements, dropped = resolve_overlaps(placements)
            # Bracing holds parts up; it can't join two assemblies that sit
            # side by side with no stud between them.  Reconnect bridges
            # small gaps to the main body and sheds what can't be reached —
            # and shedding an island can take away what a brace column
            # landed on, so the passes run to a fixed point: a store set
            # is one assembly with nothing floating, not a report.
            from ..stability import check as _check, reconnect
            added = bridged = shed = 0
            for _round in range(4):
                placements, a = brace(placements)
                placements, _ = bond(placements)
                placements, b, s = reconnect(placements)
                added += a
                bridged += b
                shed += s
                rep = _check(placements)
                if (not rep.floating and rep.components <= 1) or (a == 0 and b == 0 and s == 0):
                    break
            counts["structure:dropped_overlaps"] = dropped
            counts["structure:added_supports"] = added
            counts["structure:bridges"] = bridged
            counts["structure:dropped_islands"] = shed
        if placements:
            # the build's joints against official models' (data/part_usage.json)
            from ..usage import joint_realism
            jr = joint_realism(placements)
            counts["joints:total"] = jr["joints"]
            counts["joints:unlike_sets_pct"] = jr["unlike_sets_pct"]
            counts["joints:unrecorded_pct"] = jr["unrecorded_pct"]
        return PipelineResult(
            placements=placements,
            surface_placements=surface,
            interior_placements=interior,
            regions=regions,
            per_region_counts=counts,
            dropped_overlaps=dropped,
            added_supports=added,
        )
