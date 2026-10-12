"""Prepare a parts inventory and a measurable geometric build report."""

from dataclasses import dataclass

import numpy as np

from . import colors, pack, stability
from .voxelize import VoxelGrid


@dataclass
class BuildResult:
    grid: VoxelGrid
    placements: list[pack.Placement]
    report: dict


def structure_report(placements: list[pack.Placement],
                     check: stability.StabilityReport | None = None) -> dict:
    """The part count and structural fields of a report, for the placements
    as they are now.  Every pass that changes placements after packing
    (texture, colour swaps, feature detection, variety) must refresh these,
    or the report describes a model that is not the one written out."""
    check = stability.check(placements) if check is None else check
    return {
        "parts": len(placements),
        "skus": len(pack.summarize(placements)),
        "geometrically_connected": check.ok,
        "connected_components": check.components,
        "unsupported_parts": len(check.floating),
        "overlapping_pairs": len(check.overlaps),
        "single_stud_overhangs": check.weak_connections,
    }


def assemble(grid: VoxelGrid, *, use_tiles: bool = True, use_slopes: bool = True,
             supports: bool = True, brace_mode: str = "auto") -> BuildResult:
    """Assemble a voxel grid into a packed LEGO model with a stability report.

    `brace_mode`:
        - "auto"  : if the model has an enclosed interior cavity (hollow shell),
                    add horizontal internal-floor braces at strategic Y heights
                    that connect opposite walls without filling the whole void.
                    Otherwise fall back to "columns".
        - "floors": force internal-floor bracing.
        - "columns": legacy — vertical columns from floor up to every floating
                    part. Fills interior cavities completely.
        - "none"  : add only the foundation slab; no interior connectors.
    """
    original = grid.palette_idx
    if original.ndim != 3 or not (original >= 0).any():
        raise ValueError("The model must contain occupied voxels")
    if not np.issubdtype(original.dtype, np.integer) or original.min() < -1 or original.max() >= len(colors.palette()):
        raise ValueError("Voxel colors must be palette indices or -1")
    coords = np.argwhere(original >= 0)
    lo, hi = coords.min(axis=0), coords.max(axis=0) + 1
    voxels = original[tuple(slice(a, b) for a, b in zip(lo, hi))].copy()
    model_voxels = int((voxels >= 0).sum())
    support_color = next(i for i, c in enumerate(colors.palette()) if c.code == 71)
    if supports:
        # Foundation slab. A 2-plate base gives support columns a common
        # ground and the model something to sit on.
        voxels = np.pad(voxels, ((2, 0), (1, 1), (1, 1)), constant_values=-1)
        voxels[:2] = support_color

        # Detect enclosed interior cavity (hollow shell) BEFORE deciding on
        # the bracing strategy.
        from scipy.ndimage import binary_fill_holes
        occupied = voxels >= 0
        filled_solid = binary_fill_holes(occupied)
        cavity = filled_solid & ~occupied
        has_cavity = int(cavity.sum()) > 0

        effective_mode = brace_mode
        if effective_mode == "auto":
            effective_mode = "floors" if has_cavity else "columns"

        if effective_mode == "floors" and has_cavity:
            # Add horizontal internal-floor slabs at ~every 12 plates
            # (4 bricks). Each floor is a plate that ties the walls together
            # laterally. Much less material than filling the whole cavity
            # with columns.
            H = voxels.shape[0]
            available = H - 2  # skip foundation
            n_floors = max(1, min(4, available // 15))
            floor_ys = [
                2 + i * (available // (n_floors + 1))
                for i in range(1, n_floors + 1)
            ]
            for y in floor_ys:
                if 0 <= y < H:
                    voxels[y][cavity[y]] = support_color

    for attempt in range(voxels.shape[0] + 1):
        prepared = VoxelGrid(voxels)
        placements = pack.pack(prepared, use_tiles=use_tiles, use_slopes=use_slopes,
                               foundation_layers=2 if supports else 0)
        check = stability.check(placements)
        if not supports or not check.floating:
            break
        if supports and brace_mode == "floors":
            # Floor bracing has done what it can; stop here and let the
            # report surface any remaining unsupported parts.
            break
        if supports and brace_mode == "none":
            break
        added = 0
        for p in check.floating:
            # Legacy: vertical column from floor to floating part.
            x = p.x_stud + p.w_world // 2
            z = p.z_stud + p.d_world // 2
            for y in range(p.y_plate - 1, -1, -1):
                if voxels[y, x, z] >= 0:
                    break
                voxels[y, x, z] = support_color
                added += 1
        if not added:
            raise RuntimeError("Could not repair unsupported parts")
    else:
        raise RuntimeError("Support repair did not converge")

    report = {
        "reconstruction": grid.metadata,
        "dimensions_yxz": list(voxels.shape),
        "source_voxels": model_voxels,
        "added_support_voxels": int((voxels >= 0).sum()) - model_voxels,
        **structure_report(placements, check),
        "limits": [
            "Geometric checks do not simulate clutch strength, bending or tipping.",
            "Part/color availability must be checked before purchasing.",
        ],
    }
    return BuildResult(prepared, placements, report)
