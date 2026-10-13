"""Prepare a parts inventory and a measurable geometric build report."""

from dataclasses import dataclass

import numpy as np

from . import bonding, bricks, colors, pack, stability
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
        # Advisory: a connected model can still fail in the hand.
        "cantilevers": check.cantilever_count,
        "running_seams": check.running_seam_count,
        "single_stud_groups": check.lift_group_count,
    }


def assemble(grid: VoxelGrid, *, use_tiles: bool = True, use_slopes: bool = True,
             supports: bool = True, brace_mode: str = "auto",
             use_cheese: bool = True) -> BuildResult:
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

    pegged_weak = False
    for attempt in range(voxels.shape[0] + 1):
        prepared = VoxelGrid(voxels)
        placements = pack.pack(prepared, use_tiles=use_tiles, use_slopes=use_slopes,
                               foundation_layers=2 if supports else 0,
                               use_cheese=use_cheese)
        check = stability.check(placements)
        if not supports:
            break
        if not check.floating:
            # Once everything is held, give parts hanging by a single stud a
            # second contact wherever the model offers a short gap under
            # them, then pack one last time.
            if pegged_weak or brace_mode == "none" or not check.weak:
                break
            pegged_weak = True
            occupied = voxels >= 0
            model = occupied.copy()
            model[:2] = False
            added = 0
            for p in check.weak:
                gap = best_gap(occupied, p, model)
                if gap is None or not gap[4] or gap[3] - gap[2] > 3:
                    continue
                x, z, y_lo, y_hi, _ = gap
                voxels[y_lo:y_hi, x, z] = p.color_idx
                occupied[y_lo:y_hi, x, z] = True
                added += 1
            if not added:
                break
            continue
        if supports and brace_mode == "floors":
            # Floor bracing has done what it can; stop here and let the
            # report surface any remaining unsupported parts.
            break
        if supports and brace_mode == "none":
            break
        added = 0
        occupied = voxels >= 0
        model = occupied.copy()
        if supports:
            model[:2] = False
        for p in check.floating:
            gap = best_gap(occupied, p, model)
            if gap is None:
                continue
            x, z, y_lo, y_hi, lands = gap
            # A short peg that lands on the model takes the part's colour
            # and reads as part of it; anything longer, or a column to the
            # foundation, is scaffolding and stays grey.  Grey columns also
            # pack on their own, so the repack leaves the model's parts as
            # they were.
            peg = lands and y_hi - y_lo <= 3
            voxels[y_lo:y_hi, x, z] = p.color_idx if peg else support_color
            occupied[y_lo:y_hi, x, z] = True
            added += y_hi - y_lo
        if not added:
            raise RuntimeError("Could not repair unsupported parts")
    else:
        raise RuntimeError("Support repair did not converge")

    # Support repair holds every part; bonding then breaks the vertical
    # seams that run through several courses, which the course-by-course
    # packer cannot see.  Same cells, same colours, fewer independent
    # columns.
    placements, bond_stats = bonding.bond(placements)
    check = stability.check(placements)

    report = {
        "reconstruction": grid.metadata,
        "dimensions_yxz": list(voxels.shape),
        "source_voxels": model_voxels,
        "added_support_voxels": int((voxels >= 0).sum()) - model_voxels,
        **structure_report(placements, check),
        "bonding": bond_stats,
        "limits": [
            "Geometric checks do not simulate clutch strength, bending or tipping.",
            "Part/color availability must be checked before purchasing.",
        ],
    }
    return BuildResult(prepared, placements, report)


def best_gap(occupied: np.ndarray, p: pack.Placement, model: np.ndarray | None = None,
             studs: np.ndarray | None = None) -> tuple[int, int, int, int, bool] | None:
    """Where to put support under a part that nothing holds up.

    For every cell of its footprint, the run of empty cells below it ends
    on an occupied voxel or on the ground.  A run that lands on the model
    beats any that goes to the ground, and the shortest of those wins, so
    support is a short peg hidden under the part wherever the model offers
    one.  A column to the ground stands under the cell nearest the part's
    centre: an edge column holds the part by one stud, and the repack that
    follows may then split the part so that the column holds a neighbour
    instead.  Returns (x, z, y_lo, y_hi, lands) with y_lo..y_hi the cells
    to fill, or None when nothing can be filled (the part sits directly on
    something it cannot grip, like a tile).  `model` marks the voxels that
    count as landing on the model; by default every occupied voxel does.
    `studs` marks the voxels that offer a stud on top; when given, a run
    may not end on an occupied voxel without one (a tile, the slant of a
    slope) — there is nothing there to grip.
    """
    Y, X, Z = occupied.shape
    cx, cz = p.x_stud + (p.w_world - 1) / 2, p.z_stud + (p.d_world - 1) / 2
    best = None
    for dx in range(p.w_world):
        for dz in range(p.d_world):
            x, z = p.x_stud + dx, p.z_stud + dz
            if not (0 <= x < X and 0 <= z < Z) or p.y_plate <= 0:
                continue
            y = p.y_plate - 1
            while y >= 0 and not occupied[y, x, z]:
                y -= 1
            gap = p.y_plate - 1 - y
            if gap == 0:
                continue
            lands = y >= 0 and (model is None or bool(model[y, x, z]))
            if y >= 0 and studs is not None and not studs[y, x, z]:
                continue        # a tile, a slope's slant: nothing to grip there
            centre = abs(x - cx) + abs(z - cz)
            key = (0, gap, centre) if lands else (1, centre, gap)
            if best is None or key < best[0]:
                best = (key, (x, z, y + 1, p.y_plate, lands))
    return best[1] if best else None


def occupancy(placements: list[pack.Placement]) -> np.ndarray:
    """(Y, X, Z) boolean occupancy of a set of placements."""
    if not placements:
        return np.zeros((0, 0, 0), dtype=bool)
    Y = max(p.y_plate + p.part.height_units for p in placements)
    X = max(p.x_stud + p.w_world for p in placements)
    Z = max(p.z_stud + p.d_world for p in placements)
    occ = np.zeros((Y, X, Z), dtype=bool)
    for p in placements:
        occ[p.y_plate:p.y_plate + p.part.height_units,
            p.x_stud:p.x_stud + p.w_world, p.z_stud:p.z_stud + p.d_world] = True
    return occ


def resolve_overlaps(placements: list[pack.Placement]) -> tuple[list[pack.Placement], int]:
    """Keep each cell for the first placement that claims it.  A later
    placement that overlaps one is dropped and the cells it alone covered
    are refilled with 1x1 plates of its colour, so no overlap and no hole
    remain.  Returns the placements and how many were dropped."""
    plate_1x1 = next(p for p in bricks.catalog() if p.ldraw == "3024.dat")
    claimed: set[tuple[int, int, int]] = set()
    out: list[pack.Placement] = []
    dropped = 0
    for p in placements:
        cells = [(y, x, z)
                 for y in range(p.y_plate, p.y_plate + p.part.height_units)
                 for x in range(p.x_stud, p.x_stud + p.w_world)
                 for z in range(p.z_stud, p.z_stud + p.d_world)]
        free = [c for c in cells if c not in claimed]
        if len(free) == len(cells):
            out.append(p)
        else:
            dropped += 1
            out.extend(pack.Placement(plate_1x1, p.color_idx, x, z, y, 1, 1, 0)
                       for y, x, z in free)
        claimed.update(free)
    return out, dropped


def brace(placements: list[pack.Placement], *, max_rounds: int = 50
          ) -> tuple[list[pack.Placement], int]:
    """Hold up every floating part with 1x1 bricks and plates in the gap
    `best_gap` picks, without repacking anything.  For models that are
    already parts rather than voxels (the semantic pipeline).  Returns the
    placements and the number of parts added."""
    brick_1x1 = next(p for p in bricks.catalog() if p.ldraw == "3005.dat")
    plate_1x1 = next(p for p in bricks.catalog() if p.ldraw == "3024.dat")
    support_color = next(i for i, c in enumerate(colors.palette()) if c.code == 71)
    placements = list(placements)
    added = 0
    for _ in range(max_rounds):
        check = stability.check(placements)
        if not check.floating:
            break
        occ = occupancy(placements)
        # A column may land only where the part under it offers studs: the
        # top plate of a brick or plate, the high strip of a slope — never
        # a tile or a slope's slant (landing there held nothing, and the
        # next check found the column itself floating).
        studs = np.zeros_like(occ)
        for q in placements:
            top = q.y_plate + q.part.height_units - 1
            for sx, sz in stability.top_studs(q):
                if 0 <= sx < occ.shape[1] and 0 <= sz < occ.shape[2] and 0 <= top < occ.shape[0]:
                    studs[top, sx, sz] = True
        new: list[pack.Placement] = []
        for p in check.floating:
            gap = best_gap(occ, p, studs=studs)
            if gap is None:
                continue
            x, z, y_lo, y_hi, lands = gap
            color = p.color_idx if lands and y_hi - y_lo <= 3 else support_color
            y = y_lo
            while y < y_hi:
                part = brick_1x1 if y_hi - y >= 3 else plate_1x1
                new.append(pack.Placement(part, color, x, z, y, 1, 1, 0))
                y += part.height_units
            occ[y_lo:y_hi, x, z] = True
            studs[y_hi - 1, x, z] = True
        if not new:
            break
        placements.extend(new)
        added += len(new)
    return placements, added
