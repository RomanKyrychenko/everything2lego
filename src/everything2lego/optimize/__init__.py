"""Loss-based LEGO optimizer.

Given a mesh, define L(A) reflecting shape / color / diversity / coherence,
then search over discrete part-cell assignments to minimize L. Works on
any mesh — no category priors, no templates.

Usage (Python):
    from everything2lego.optimize import build_target, optimize
    target = build_target(mesh, max_studs=28)
    result = optimize(target, iters=20_000)
    result.export_placements() → list[Placement]

Usage (CLI):
    everything2lego optimize path/to/model.stl --out out_opt/ --iters 30000
"""

from __future__ import annotations

from .target import Target, build_target
from .loss import Weights, LossBreakdown, evaluate, delta_swap_cell
from .moves import (
    Move,
    propose_region_swap,
    propose_texture_band,
    propose_slope_align,
    propose_tile_top,
    propose_recolor,
)
from .search import OptResult, optimize
from .export import assembly_to_placements
from .init_slopes import initial_assembly_with_slopes

__all__ = [
    "Target",
    "build_target",
    "Weights",
    "LossBreakdown",
    "evaluate",
    "delta_swap_cell",
    "Move",
    "propose_region_swap",
    "propose_texture_band",
    "propose_slope_align",
    "propose_tile_top",
    "propose_recolor",
    "OptResult",
    "optimize",
    "assembly_to_placements",
    "initial_assembly_with_slopes",
]
