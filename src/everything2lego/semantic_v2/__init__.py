"""Semantic-MILP LEGO pipeline (v2).

Replaces the SA-over-cell-types loop with a two-stage design:

    Mesh → RegionDetector → [Region] → RegionBuilder → Assembly

Each RegionBuilder uses an MILP solver internally so the parts choice for
each region is provably near-optimal, not the artefact of greedy scan order.
See ARCHITECTURE_V2.md for the full design.
"""

from __future__ import annotations

from .mesh_target import MeshTarget
from .regions import (Region, RegionLabel, PlanarRegion, CylindricalRegion,
                        SphericalRegion, CornerRegion, FoundationRegion,
                        DetailRegion)
from .detector import RegionDetector
from .builders import (RegionBuilder, PlanarWallBuilder, PlanarRoofBuilder,
                         FoundationBuilder, CylindricalBuilder,
                         SphericalBuilder, GenericVoxelBuilder,
                         BuilderRegistry)
from .loss import LossFunction, LossWeights, LossBreakdown
from .milp import MILPPacker, PackResult
from .moves import (Move, RecolorPlacementMove, SwapPartMove,
                      SwapPartForTileMove, AddSlopeCapMove, TrimInteriorMove,
                      SplitPlacementMove, MergeAdjacentMove, RegionRebuildMove,
                      ReskinRoofMove, AddInsetLedgeMove, SurfacePolishMove,
                      build_move_pool, sample_move)
from .pipeline import SemanticLegoPipeline, PipelineConfig
from .refiner import Refiner, RefinerResult
from .gradient_refiner import GradientRefiner, GradientRefinerResult


def _build_region_cell_masks(target, regions, palette_idx, occ):
    """Rasterise each Region's bounding box into voxel space and pair
    it with the palette-indices seen inside that mask.  Returns a list
    of (mask, allowed_ids) tuples for the SoftAssembly's per-region
    palette lock.  Each region's allowed set is derived from the mesh
    itself — no hand-labelling.
    """
    import numpy as np
    out: list[tuple[np.ndarray, frozenset[int]]] = []
    Y, X, Z = occ.shape
    seen: set[bytes] = set()
    for reg in regions:
        try:
            mn = np.asarray(reg.bounds[0], dtype=np.float32)
            mx = np.asarray(reg.bounds[1], dtype=np.float32)
        except Exception:
            continue
        mn_h = np.array([mn[0], mn[1], mn[2], 1.0], dtype=np.float32)
        mx_h = np.array([mx[0], mx[1], mx[2], 1.0], dtype=np.float32)
        mn_v = (mn_h @ target.mesh_to_voxel.T)[:3]
        mx_v = (mx_h @ target.mesh_to_voxel.T)[:3]
        lo = np.floor(np.minimum(mn_v, mx_v)).astype(int)
        hi = np.ceil(np.maximum(mn_v, mx_v)).astype(int) + 1
        x0, y0, z0 = max(0, lo[0]), max(0, lo[1]), max(0, lo[2])
        x1, y1, z1 = min(X, hi[0]), min(Y, hi[1]), min(Z, hi[2])
        if x0 >= x1 or y0 >= y1 or z0 >= z1:
            continue
        mask = np.zeros((Y, X, Z), dtype=bool)
        mask[y0:y1, x0:x1, z0:z1] = occ[y0:y1, x0:x1, z0:z1]
        if not mask.any():
            continue
        key = mask.tobytes()
        if key in seen:
            continue
        seen.add(key)
        allowed = frozenset(int(v) for v in np.unique(palette_idx[mask])
                                if v >= 0)
        if allowed:
            out.append((mask, allowed))
    return out
try:
    from .soft_assembly import SoftAssembly
    from .soft_loss import SoftLoss, SoftLossWeights, SoftLossBreakdown
    from .soft_trainer import SoftTrainer, SoftTrainerResult
    from .soft_renderer import SoftRenderer, photo_loss
except ImportError:
    SoftAssembly = SoftLoss = SoftLossWeights = None
    SoftLossBreakdown = SoftTrainer = SoftTrainerResult = None
    SoftRenderer = photo_loss = None

__all__ = [
    "MeshTarget",
    "Region", "RegionLabel",
    "PlanarRegion", "CylindricalRegion", "SphericalRegion",
    "CornerRegion", "FoundationRegion", "DetailRegion",
    "RegionDetector",
    "RegionBuilder", "PlanarWallBuilder", "PlanarRoofBuilder",
    "FoundationBuilder", "CylindricalBuilder", "SphericalBuilder",
    "GenericVoxelBuilder", "BuilderRegistry",
    "MILPPacker", "PackResult",
    "SemanticLegoPipeline", "PipelineConfig",
    "LossFunction", "LossWeights", "LossBreakdown",
    "Move", "RecolorPlacementMove", "SwapPartMove", "SwapPartForTileMove",
    "AddSlopeCapMove", "TrimInteriorMove", "SplitPlacementMove",
    "MergeAdjacentMove", "RegionRebuildMove", "ReskinRoofMove",
    "AddInsetLedgeMove", "SurfacePolishMove",
    "build_move_pool", "sample_move",
    "Refiner", "RefinerResult",
    "GradientRefiner", "GradientRefinerResult",
    "SoftAssembly", "SoftLoss", "SoftLossWeights", "SoftLossBreakdown",
    "SoftTrainer", "SoftTrainerResult",
    "SoftRenderer", "photo_loss",
]
