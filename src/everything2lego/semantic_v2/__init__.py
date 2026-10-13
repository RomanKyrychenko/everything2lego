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


def _snap_target_to_region_palette(palette_idx, per_region_palette):
    """Return a copy of `palette_idx` where, inside each region mask, cells
    whose colour the region's lock forbids take the mask's most common
    allowed colour.  Empty cells (-1) are left alone.

    The lock alone is not enough: the soft trainer's colour loss compares
    against the per-cell target, so a locked-out cell carries a permanent
    error of 1.0 and the only way the optimiser can lower a weighted mean
    over occupied cells is to EMPTY that cell — the lock turns into erosion
    pressure on the very walls it was meant to flatten.  Snapping the target
    makes it attainable and the loss meaningful again.
    """
    import numpy as np
    out = np.array(palette_idx, copy=True)
    masks = [(np.asarray(m, dtype=bool), np.fromiter(a, dtype=np.int64))
             for m, a in per_region_palette if a]
    if not masks:
        return out
    # Union of allowed colours per cell, mirroring the lock in
    # SoftAssembly.from_placements (overlapping boxes keep every candidate).
    n_colors = int(max(int(out.max()), max(int(a.max()) for _, a in masks))) + 1
    cell_allowed = np.zeros(out.shape + (n_colors,), dtype=bool)
    covered = np.zeros(out.shape, dtype=bool)
    for m, a in masks:
        for c in a:
            cell_allowed[..., int(c)] |= m
        covered |= m
    occupied = out >= 0
    idx = np.clip(out, 0, n_colors - 1)
    fine = ~covered | ~occupied | np.take_along_axis(
        cell_allowed, idx[..., None], axis=-1)[..., 0]
    need = ~fine
    if not need.any():
        return out
    # Snap to the dominant allowed colour of the LARGEST region containing
    # the cell: process masks smallest → largest so the largest writes last.
    for m, a in sorted(masks, key=lambda t: int(t[0].sum())):
        sel = m & need
        if not sel.any():
            continue
        present = out[m & occupied]
        present = present[np.isin(present, a)]
        dominant = int(np.bincount(present).argmax()) if present.size else int(a[0])
        out[sel] = dominant
    return out


def _build_region_cell_masks(target, regions, palette_idx, occ,
                             dominant_share: float = 0.70):
    """One disjoint cell mask per PLANAR region, each paired with the
    colours that region is locked to.

    A region's bounding box is only a candidate area: boxes of a
    building's walls and roof pitches overlap most of the surface.  Each
    occupied cell inside one or more boxes is assigned to the region whose
    plane it lies closest to, so masks never overlap and every cell has
    exactly one lock.  (Locking cumulatively over overlapping boxes left
    cells with no allowed colour; taking the union left them with no lock.)

    The plane is the region's `plane_normal`/`plane_origin` when it has
    them, else the mean of its sampled `normals` and `points`.  Non-planar
    regions (detail) get no lock.  A planar region is one material, so its
    lock is the fewest colours covering `dominant_share` of its cells —
    usually one, two for a genuinely two-tone surface.
    """
    import numpy as np
    Y, X, Z = occ.shape
    T = np.asarray(target.mesh_to_voxel, dtype=np.float64)
    T_inv = np.linalg.inv(T)

    planar: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for reg in regions:
        label_name = getattr(getattr(reg, "label", None), "name", "")
        if label_name not in ("PLANAR_WALL", "PLANAR_ROOF", "FOUNDATION"):
            continue
        try:
            mn = np.asarray(reg.bounds[0], dtype=np.float64)
            mx = np.asarray(reg.bounds[1], dtype=np.float64)
        except Exception:
            continue
        mn_v = (np.append(mn, 1.0) @ T.T)[:3]
        mx_v = (np.append(mx, 1.0) @ T.T)[:3]
        lo = np.floor(np.minimum(mn_v, mx_v)).astype(int)
        hi = np.ceil(np.maximum(mn_v, mx_v)).astype(int) + 1
        x0, y0, z0 = max(0, lo[0]), max(0, lo[1]), max(0, lo[2])
        x1, y1, z1 = min(X, hi[0]), min(Y, hi[1]), min(Z, hi[2])
        if x0 >= x1 or y0 >= y1 or z0 >= z1:
            continue
        box = np.zeros((Y, X, Z), dtype=bool)
        box[y0:y1, x0:x1, z0:z1] = occ[y0:y1, x0:x1, z0:z1]
        if not box.any():
            continue
        n = getattr(reg, "plane_normal", None)
        o = getattr(reg, "plane_origin", None)
        if n is None or o is None:
            pts = getattr(reg, "points", None)
            nrm = getattr(reg, "normals", None)
            if pts is None or nrm is None or len(pts) == 0:
                continue
            n = np.asarray(nrm, dtype=np.float64).mean(axis=0)
            o = np.asarray(pts, dtype=np.float64).mean(axis=0)
        n = np.asarray(n, dtype=np.float64)
        nn = float(np.linalg.norm(n))
        if nn < 1e-9:
            continue
        planar.append((box, n / nn, np.asarray(o, dtype=np.float64)))
    if not planar:
        return []

    # Cell centres (voxel x+.5, y+.5, z+.5) back in mesh coordinates.
    yy, xx, zz = np.indices((Y, X, Z))
    centres = np.stack([xx + 0.5, yy + 0.5, zz + 0.5, np.ones_like(xx)], axis=-1
                       ).astype(np.float64) @ T_inv.T
    P = centres[..., :3]

    owner = np.full((Y, X, Z), -1, dtype=np.int64)
    best = np.full((Y, X, Z), np.inf)
    for i, (box, n, o) in enumerate(planar):
        d = np.abs((P - o) @ n)
        sel = box & (d < best)
        owner[sel] = i
        best[sel] = d[sel]

    out: list[tuple[np.ndarray, frozenset[int]]] = []
    for i in range(len(planar)):
        mask = owner == i
        if not mask.any():
            continue
        ids, counts = np.unique(palette_idx[mask], return_counts=True)
        valid = ids >= 0
        ids, counts = ids[valid], counts[valid]
        if not ids.size:
            continue
        order = np.argsort(-counts)
        cum = np.cumsum(counts[order]) / counts.sum()
        n_keep = int(np.searchsorted(cum, dominant_share) + 1)
        out.append((mask, frozenset(int(v) for v in ids[order[:n_keep]])))
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
