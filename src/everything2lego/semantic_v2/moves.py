"""Region-aware proposal moves for the v3 refiner.

Each Move is a small mutation on the Assembly: it knows how to `apply`
itself (returning a new list of Placements) and how much of the assembly
it changed (for cheap delta-loss). Moves are region-aware — they operate
inside a specific region so refinement doesn't stray across boundaries.

Move class hierarchy:
    Move (abstract)
    ├── RecolorPlacementMove       swap one placement's color
    ├── SwapPartMove               swap one placement's part (keep footprint)
    ├── MergeAdjacentMove          collapse two adjacent same-color same-family placements into one bigger part
    └── SplitPlacementMove         split one placement into smaller pieces with varied colors/types

More move types can be added by subclassing Move without touching the
Refiner.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..bricks import catalog, Part
from ..colors import palette
from ..pack import Placement
from ..templates.base import part_by_ldraw


@dataclass
class MoveResult:
    """Delta produced by applying a move — bookkeeping for Refiner."""
    new_placements: list[Placement]
    changed_indices: set[int]      # indices into the original placement list
    added_placements: list[Placement]  # newly-added Placements (not in original)


class Move(ABC):
    """Abstract mutation on an Assembly."""

    @abstractmethod
    def apply(self, placements: list[Placement],
                rng: random.Random) -> Optional[MoveResult]:
        """Return the new assembly plus which indices changed, or None if
        this move can't be constructed on the current assembly."""


# ---------------------------------------------------------------------------
# Recolor
# ---------------------------------------------------------------------------

class RecolorPlacementMove(Move):
    """Change one placement's colour to one of the palette entries whose
    LAB is close to the target colour at that placement's centre."""

    def __init__(self, target_lookup: "callable[[Placement], int]",
                  color_pool: list[int]) -> None:
        # target_lookup(placement) returns the best-match palette index.
        self.target_lookup = target_lookup
        self.color_pool = color_pool

    def apply(self, placements, rng):
        if not placements:
            return None
        i = rng.randrange(len(placements))
        old = placements[i]
        new_color = self.target_lookup(old)
        if new_color == old.color_idx:
            new_color = rng.choice(self.color_pool)
        if new_color == old.color_idx:
            return None
        new_p = Placement(part=old.part, color_idx=int(new_color),
                            x_stud=old.x_stud, y_plate=old.y_plate,
                            z_stud=old.z_stud, w_world=old.w_world,
                            d_world=old.d_world, rotation_deg=old.rotation_deg)
        new_placements = list(placements)
        new_placements[i] = new_p
        return MoveResult(new_placements=new_placements,
                            changed_indices={i}, added_placements=[])


# ---------------------------------------------------------------------------
# Swap part (same footprint, different shape/family)
# ---------------------------------------------------------------------------

class SwapPartMove(Move):
    """Swap a placement's Part for another Part with the same footprint —
    same size but a different shape (e.g. plate ↔ tile) or texture (plain
    ↔ log ↔ masonry).  Used to add texture/tile variety at surface without
    disturbing the packing."""

    def __init__(self) -> None:
        self._by_footprint: dict[tuple[int, int], list[Part]] = {}
        for p in catalog():
            self._by_footprint.setdefault(p.footprint, []).append(p)

    def apply(self, placements, rng):
        if not placements:
            return None
        # Pick a random placement with alternatives at its footprint.
        idxs = list(range(len(placements)))
        rng.shuffle(idxs)
        for i in idxs[:20]:
            old = placements[i]
            alts = self._by_footprint.get(old.part.footprint, [])
            others = [p for p in alts if p.ldraw != old.part.ldraw
                        and p.height_units == old.part.height_units]
            if not others:
                continue
            new_part = rng.choice(others)
            new_p = Placement(part=new_part, color_idx=old.color_idx,
                                x_stud=old.x_stud, y_plate=old.y_plate,
                                z_stud=old.z_stud, w_world=old.w_world,
                                d_world=old.d_world, rotation_deg=old.rotation_deg)
            new_placements = list(placements)
            new_placements[i] = new_p
            return MoveResult(new_placements=new_placements,
                                changed_indices={i}, added_placements=[])
        return None


# ---------------------------------------------------------------------------
# Merge two adjacent placements into one bigger part
# ---------------------------------------------------------------------------

class MergeAdjacentMove(Move):
    """Find two face-adjacent placements sharing (color, shape) whose
    combined footprint matches a catalog part. Replace them with the
    merged part. Reduces total part count."""

    def __init__(self) -> None:
        self._parts_by_footprint: dict[tuple[int, int], Part] = {}
        for p in catalog():
            self._parts_by_footprint.setdefault(p.footprint, p)

    def apply(self, placements, rng):
        if len(placements) < 2:
            return None
        # Try up to N random pairs of same-y-plate, same-color, same-shape,
        # same-orientation placements.
        idxs = list(range(len(placements)))
        rng.shuffle(idxs)
        for a_idx in idxs[:40]:
            a = placements[a_idx]
            for b_idx in idxs:
                if a_idx == b_idx:
                    continue
                b = placements[b_idx]
                if (a.color_idx != b.color_idx
                        or a.part.shape != b.part.shape
                        or a.y_plate != b.y_plate
                        or a.part.height_units != b.part.height_units):
                    continue
                merged = _try_merge_footprints(a, b, self._parts_by_footprint)
                if merged is None:
                    continue
                new_placements = [p for i, p in enumerate(placements)
                                     if i != a_idx and i != b_idx]
                new_placements.append(merged)
                return MoveResult(new_placements=new_placements,
                                    changed_indices=set(),
                                    added_placements=[merged])
        return None


def _try_merge_footprints(a: Placement, b: Placement,
                            parts_by_fp: dict[tuple[int, int], Part]
                            ) -> Optional[Placement]:
    """If a and b are edge-adjacent in the XZ plane, try to form a bigger
    rectangle that matches a catalog part."""
    # a and b share y, height_units, color, shape.
    # Check adjacency: same z and touching in X, or same x and touching in Z.
    a_x1 = a.x_stud + a.w_world
    b_x1 = b.x_stud + b.w_world
    a_z1 = a.z_stud + a.d_world
    b_z1 = b.z_stud + b.d_world

    # Edge-adjacent in X (same z range).
    if a.z_stud == b.z_stud and a.d_world == b.d_world:
        if a_x1 == b.x_stud:
            merged_x, merged_w = a.x_stud, a.w_world + b.w_world
            merged_z, merged_d = a.z_stud, a.d_world
        elif b_x1 == a.x_stud:
            merged_x, merged_w = b.x_stud, a.w_world + b.w_world
            merged_z, merged_d = a.z_stud, a.d_world
        else:
            return None
    # Edge-adjacent in Z (same x range).
    elif a.x_stud == b.x_stud and a.w_world == b.w_world:
        if a_z1 == b.z_stud:
            merged_x, merged_w = a.x_stud, a.w_world
            merged_z, merged_d = a.z_stud, a.d_world + b.d_world
        elif b_z1 == a.z_stud:
            merged_x, merged_w = a.x_stud, a.w_world
            merged_z, merged_d = b.z_stud, a.d_world + b.d_world
        else:
            return None
    else:
        return None

    fp = (merged_w, merged_d)
    part = parts_by_fp.get(fp) or parts_by_fp.get((fp[1], fp[0]))
    if part is None or part.shape != a.part.shape:
        return None
    rot = 0 if part.footprint == fp else 90
    return Placement(part=part, color_idx=a.color_idx,
                       x_stud=merged_x, y_plate=a.y_plate, z_stud=merged_z,
                       w_world=merged_w, d_world=merged_d, rotation_deg=rot)


# ---------------------------------------------------------------------------
# Move pool factory
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# SwapPartForTileMove — plate → tile at exposed top surfaces
# ---------------------------------------------------------------------------

class SwapPartForTileMove(Move):
    """When a plate placement has no placement covering its top face, swap
    the plate for a tile of the same footprint. Reduces `shape_surface`
    (studs poking above the target surface) and `coherence` at surface
    tops (matching the smoothness of a mesh surface)."""

    def __init__(self) -> None:
        # Tile parts by footprint (bigger first).
        self._tile_by_fp: dict[tuple[int, int], Part] = {}
        for p in catalog():
            if p.shape != "tile":
                continue
            for w, d in p.orientations():
                self._tile_by_fp.setdefault((w, d), p)

    def apply(self, placements, rng):
        if not placements:
            return None
        # Build a fast look-up of placements at each cell for "is anything
        # covering my top" — only look at placements whose y_plate is above.
        by_cell: dict[tuple[int, int, int], int] = {}
        for i, p in enumerate(placements):
            for dx in range(p.w_world):
                for dz in range(p.d_world):
                    by_cell[(p.y_plate + p.part.height_units,
                              p.x_stud + dx, p.z_stud + dz)] = i
        idxs = list(range(len(placements)))
        rng.shuffle(idxs)
        for i in idxs[:60]:
            old = placements[i]
            if old.part.shape != "plate":
                continue
            # Check if any placement sits directly above this cell.
            covered = any((old.y_plate + old.part.height_units,
                            old.x_stud + dx, old.z_stud + dz) in by_cell
                          for dx in range(old.w_world)
                          for dz in range(old.d_world))
            if covered:
                continue
            new_part = self._tile_by_fp.get(old.part.footprint) \
                        or self._tile_by_fp.get((old.part.footprint[1],
                                                    old.part.footprint[0]))
            if new_part is None or new_part.height_units != old.part.height_units:
                continue
            new_p = Placement(part=new_part, color_idx=old.color_idx,
                                x_stud=old.x_stud, y_plate=old.y_plate,
                                z_stud=old.z_stud, w_world=old.w_world,
                                d_world=old.d_world, rotation_deg=old.rotation_deg)
            new_placements = list(placements)
            new_placements[i] = new_p
            return MoveResult(new_placements=new_placements,
                                changed_indices={i}, added_placements=[])
        return None


# ---------------------------------------------------------------------------
# AddSlopeCapMove — add a slope on top of an edge plate
# ---------------------------------------------------------------------------

class AddSlopeCapMove(Move):
    """Add a 45° slope 3040 on top of an EDGE placement — a placement
    with no adjacent placement in one horizontal direction. The slope
    orientation points OUTWARD from the assembly, reducing shape_surface
    by beveling the exposed step edge."""

    def __init__(self) -> None:
        self._slope = part_by_ldraw("3040.dat")

    def apply(self, placements, rng):
        if self._slope is None or not placements:
            return None
        # Build cell occupancy.
        occ_top: dict[tuple[int, int, int], int] = {}
        occ_side: dict[tuple[int, int, int], int] = {}
        for i, p in enumerate(placements):
            top_y = p.y_plate + p.part.height_units
            for dx in range(p.w_world):
                for dz in range(p.d_world):
                    x = p.x_stud + dx
                    z = p.z_stud + dz
                    occ_top[(top_y, x, z)] = i
                    for y in range(p.y_plate, top_y):
                        occ_side[(y, x, z)] = i
        idxs = list(range(len(placements)))
        rng.shuffle(idxs)
        for i in idxs[:60]:
            old = placements[i]
            top_y = old.y_plate + old.part.height_units
            # Not a top surface: skip.
            if any((top_y, old.x_stud + dx, old.z_stud + dz) in occ_side
                    for dx in range(old.w_world)
                    for dz in range(old.d_world)):
                continue
            # Find the exposed edge direction (no neighbor in +X/-X/+Z/-Z at top).
            edges = {"+x": True, "-x": True, "+z": True, "-z": True}
            for dx in range(old.w_world):
                for dz in range(old.d_world):
                    for dr, key in ((( 1, 0), "+x"), ((-1, 0), "-x"),
                                     (( 0, 1), "+z"), (( 0,-1), "-z")):
                        nx = old.x_stud + dx + dr[0]
                        nz = old.z_stud + dz + dr[1]
                        if (old.y_plate, nx, nz) in occ_side:
                            edges[key] = False
                            break
            open_edges = [k for k, v in edges.items() if v]
            if not open_edges:
                continue
            low_dir = rng.choice(open_edges)
            rot = {"+x": 0, "-z": 90, "-x": 180, "+z": 270}[low_dir]
            # 3040 footprint: 2 studs long, 1 wide. Anchor 2 in low_dir axis.
            # Simplest: add a 1-cell 3040 aligned with the edge cell.
            if low_dir in ("+x", "-x"):
                w, d = 2, 1
                ax = old.x_stud + (old.w_world - 1 if low_dir == "-x" else 0)
                az = old.z_stud
                if ax < 0 or ax + w > old.x_stud + old.w_world + 1:
                    continue
            else:
                w, d = 1, 2
                ax = old.x_stud
                az = old.z_stud + (old.d_world - 1 if low_dir == "-z" else 0)
            # Skip if the cap would sit where another placement already is.
            for dx in range(w):
                for dz in range(d):
                    if (top_y, ax + dx, az + dz) in occ_side:
                        break
                else:
                    continue
                break
            else:
                new_slope = Placement(part=self._slope, color_idx=old.color_idx,
                                        x_stud=ax, y_plate=top_y, z_stud=az,
                                        w_world=w, d_world=d, rotation_deg=rot)
                new_placements = list(placements) + [new_slope]
                return MoveResult(new_placements=new_placements,
                                    changed_indices=set(),
                                    added_placements=[new_slope])
        return None


# ---------------------------------------------------------------------------
# TrimInteriorMove — remove interior placements that are hidden
# ---------------------------------------------------------------------------

class TrimInteriorMove(Move):
    """Remove a placement whose cells are entirely INSIDE the mesh volume
    (no cells on the surface). Interior isn't visible, so a hidden brick
    is dead-weight — SA accepts if the loss weights reward count.

    Requires a mesh_occupancy grid (target's voxel_grid.occupied()) so the
    move knows the mesh boundary.  Also requires a `surface_mask` (cells
    with at least one empty face-neighbour)."""

    def __init__(self, mesh_occ, surface_mask) -> None:
        import numpy as np
        self.mesh_occ = mesh_occ
        self.surface_mask = surface_mask

    def apply(self, placements, rng):
        if not placements:
            return None
        Y, X, Z = self.mesh_occ.shape
        idxs = list(range(len(placements)))
        rng.shuffle(idxs)
        for i in idxs[:60]:
            p = placements[i]
            fully_interior = True
            for dy in range(p.part.height_units):
                y = p.y_plate + dy
                if not (0 <= y < Y):
                    fully_interior = False; break
                for dx in range(p.w_world):
                    x = p.x_stud + dx
                    if not (0 <= x < X):
                        fully_interior = False; break
                    for dz in range(p.d_world):
                        z = p.z_stud + dz
                        if not (0 <= z < Z):
                            fully_interior = False; break
                        if self.surface_mask[y, x, z]:
                            fully_interior = False; break
                    if not fully_interior: break
                if not fully_interior: break
            if not fully_interior:
                continue
            new_placements = list(placements)
            del new_placements[i]
            return MoveResult(new_placements=new_placements,
                                changed_indices=set(), added_placements=[])
        return None


# ---------------------------------------------------------------------------
# SplitPlacementMove — split a big placement into 2+ smaller ones
# ---------------------------------------------------------------------------

class SplitPlacementMove(Move):
    """Split a placement whose footprint area ≥ 4 into two smaller pieces
    with different (part, color). Boosts diversity + variety."""

    def __init__(self) -> None:
        # Small-footprint parts sorted by area ascending (want small pieces
        # to split into).
        self._small_by_fp: dict[tuple[int, int], list[Part]] = {}
        for p in catalog():
            if p.footprint[0] * p.footprint[1] > 4:
                continue
            self._small_by_fp.setdefault(p.footprint, []).append(p)

    def apply(self, placements, rng):
        if not placements:
            return None
        idxs = list(range(len(placements)))
        rng.shuffle(idxs)
        for i in idxs[:60]:
            old = placements[i]
            area = old.w_world * old.d_world
            if area < 4:
                continue
            # Split along the longer axis.
            if old.w_world >= old.d_world:
                w1 = old.w_world // 2
                w2 = old.w_world - w1
                pieces = [(old.x_stud,              old.z_stud, w1, old.d_world),
                            (old.x_stud + w1,         old.z_stud, w2, old.d_world)]
            else:
                d1 = old.d_world // 2
                d2 = old.d_world - d1
                pieces = [(old.x_stud, old.z_stud,          old.w_world, d1),
                            (old.x_stud, old.z_stud + d1,    old.w_world, d2)]
            new_ps: list[Placement] = []
            for (px, pz, pw, pd) in pieces:
                cands = self._small_by_fp.get((pw, pd)) \
                          or self._small_by_fp.get((pd, pw))
                if not cands:
                    new_ps = []
                    break
                # Prefer a DIFFERENT shape from the original for diversity.
                choices = [c for c in cands if c.shape != old.part.shape] or cands
                new_part = rng.choice(choices)
                rot = 0 if new_part.footprint == (pw, pd) else 90
                new_ps.append(Placement(part=new_part, color_idx=old.color_idx,
                                          x_stud=px, y_plate=old.y_plate,
                                          z_stud=pz, w_world=pw, d_world=pd,
                                          rotation_deg=rot))
            if len(new_ps) != len(pieces):
                continue
            new_placements = [p for j, p in enumerate(placements) if j != i]
            new_placements.extend(new_ps)
            return MoveResult(new_placements=new_placements,
                                changed_indices=set(),
                                added_placements=new_ps)
        return None


# ---------------------------------------------------------------------------
# RegionRebuildMove — re-run a region's builder with tweaked params
# ---------------------------------------------------------------------------

class RegionRebuildMove(Move):
    """Placeholder for the region-level rebuild: swap the entire set of
    placements from one region for the output of a *different* Builder
    on the same Region.  Requires the pipeline to have tracked region-→
    placements mapping.  For v3 we short-circuit to a no-op when the
    mapping isn't available; a full implementation would keep the region
    tracking around and hand it to this move."""

    def __init__(self, region_placements: dict | None = None,
                  registry=None, target=None) -> None:
        self.region_placements = region_placements or {}
        self.registry = registry
        self.target = target

    def apply(self, placements, rng):
        if not self.region_placements or self.registry is None or self.target is None:
            return None
        # Pick a random region; try to rebuild it with a different builder.
        region_ids = list(self.region_placements.keys())
        if not region_ids:
            return None
        region = rng.choice(region_ids)
        placements_in_region = self.region_placements[region]
        # Try each candidate builder that would fire, except the one
        # currently responsible.
        idxs_to_drop = {id(p) for p in placements_in_region}
        new_placements = [p for p in placements if id(p) not in idxs_to_drop]
        # For now, just drop the region — SA can accept if loss decreases,
        # else revert. Real rebuild = call an alternative Builder here.
        if len(new_placements) == len(placements):
            return None
        return MoveResult(new_placements=new_placements,
                            changed_indices=set(), added_placements=[])


# ---------------------------------------------------------------------------
# ReskinRoofMove — merge 3040 pairs into 15068 curved 2x2
# ---------------------------------------------------------------------------

class ReskinRoofMove(Move):
    """Find two adjacent 45° slope 3040 placements at the same y_plate,
    same orientation, same color, and MERGE them into one Slope Curved 2×2
    (15068)."""

    def __init__(self) -> None:
        self._curved = part_by_ldraw("15068.dat")

    def apply(self, placements, rng):
        if self._curved is None or len(placements) < 2:
            return None
        by_key: dict[tuple[int, int], list[int]] = {}
        for i, p in enumerate(placements):
            if p.part.ldraw != "3040.dat":
                continue
            by_key.setdefault((p.rotation_deg, p.y_plate), []).append(i)
        candidates: list[tuple[int, int]] = []
        for group in by_key.values():
            if len(group) < 2:
                continue
            for j, a_idx in enumerate(group):
                a = placements[a_idx]
                for b_idx in group[j+1:]:
                    b = placements[b_idx]
                    if a.color_idx != b.color_idx:
                        continue
                    if a.rotation_deg in (0, 180):
                        if a.x_stud == b.x_stud and abs(a.z_stud - b.z_stud) == 1:
                            candidates.append((a_idx, b_idx))
                    else:
                        if a.z_stud == b.z_stud and abs(a.x_stud - b.x_stud) == 1:
                            candidates.append((a_idx, b_idx))
        if not candidates:
            return None
        a_idx, b_idx = rng.choice(candidates)
        a, b = placements[a_idx], placements[b_idx]
        if a.rotation_deg in (0, 180):
            px, pz = a.x_stud, min(a.z_stud, b.z_stud)
        else:
            px, pz = min(a.x_stud, b.x_stud), a.z_stud
        curved_p = Placement(part=self._curved, color_idx=a.color_idx,
                                x_stud=px, y_plate=a.y_plate, z_stud=pz,
                                w_world=2, d_world=2, rotation_deg=a.rotation_deg)
        new_placements = [p for i, p in enumerate(placements)
                             if i not in (a_idx, b_idx)]
        new_placements.append(curved_p)
        return MoveResult(new_placements=new_placements,
                            changed_indices=set(),
                            added_placements=[curved_p])


# ---------------------------------------------------------------------------
# AddInsetLedgeMove — add a small tile at an exposed wall top
# ---------------------------------------------------------------------------

class AddInsetLedgeMove(Move):
    """At a placement whose top face is fully exposed and whose horizontal
    neighbour cells are also empty, add a 1×1 tile abutting it — an eave
    overhang / roof trim.  Reduces shape_surface at wall-to-roof junctions."""

    def __init__(self) -> None:
        self._tile = part_by_ldraw("3070b.dat")

    def apply(self, placements, rng):
        if self._tile is None or not placements:
            return None
        occ: set[tuple[int, int, int]] = set()
        for p in placements:
            for dy in range(p.part.height_units):
                for dx in range(p.w_world):
                    for dz in range(p.d_world):
                        occ.add((p.y_plate + dy,
                                  p.x_stud + dx, p.z_stud + dz))
        idxs = list(range(len(placements)))
        rng.shuffle(idxs)
        for i in idxs[:60]:
            old = placements[i]
            top_y = old.y_plate + old.part.height_units
            if any((top_y, old.x_stud + dx, old.z_stud + dz) in occ
                    for dx in range(old.w_world)
                    for dz in range(old.d_world)):
                continue
            for dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ax = old.x_stud + (old.w_world if dr[0] > 0 else -1 if dr[0] < 0 else 0)
                az = old.z_stud + (old.d_world if dr[1] > 0 else -1 if dr[1] < 0 else 0)
                if dr[0] == 0:
                    ax = old.x_stud
                if dr[1] == 0:
                    az = old.z_stud
                if (top_y - 1, ax, az) in occ or (top_y, ax, az) in occ:
                    continue
                ledge = Placement(part=self._tile, color_idx=old.color_idx,
                                    x_stud=ax, y_plate=top_y - 1, z_stud=az,
                                    w_world=1, d_world=1, rotation_deg=0)
                return MoveResult(new_placements=list(placements) + [ledge],
                                    changed_indices=set(),
                                    added_placements=[ledge])
        return None


# ---------------------------------------------------------------------------
# SurfacePolishMove — delete placements outside the mesh volume
# ---------------------------------------------------------------------------

class SurfacePolishMove(Move):
    """Delete a placement whose majority of cells sit OUTSIDE the mesh
    occupancy — that placement is over-shooting the target shape.  Direct
    shape_volume + shape_surface reduction."""

    def __init__(self, mesh_occ) -> None:
        self.mesh_occ = mesh_occ

    def apply(self, placements, rng):
        if not placements:
            return None
        Y, X, Z = self.mesh_occ.shape
        idxs = list(range(len(placements)))
        rng.shuffle(idxs)
        for i in idxs[:60]:
            p = placements[i]
            outside = 0
            total = 0
            for dy in range(p.part.height_units):
                y = p.y_plate + dy
                for dx in range(p.w_world):
                    x = p.x_stud + dx
                    for dz in range(p.d_world):
                        z = p.z_stud + dz
                        total += 1
                        if not (0 <= y < Y and 0 <= x < X and 0 <= z < Z):
                            outside += 1
                        elif not self.mesh_occ[y, x, z]:
                            outside += 1
            if outside * 2 >= total:
                new_placements = list(placements)
                del new_placements[i]
                return MoveResult(new_placements=new_placements,
                                    changed_indices=set(),
                                    added_placements=[])
        return None


# ---------------------------------------------------------------------------
# Move pool factory
# ---------------------------------------------------------------------------

def build_move_pool(target_color_lookup, catalog_color_pool: list[int],
                       *,
                       mesh_occ=None, surface_mask=None,
                       region_placements=None, registry=None, target=None,
                       ) -> list[tuple[Move, int]]:
    """Return the default move pool with sampling weights.

    Optional args wire in the moves that need extra context:
      - `mesh_occ` + `surface_mask` enable `TrimInteriorMove`.
      - `region_placements` + `registry` + `target` enable
        `RegionRebuildMove`.
    """
    pool: list[tuple[Move, int]] = [
        (RecolorPlacementMove(target_color_lookup, catalog_color_pool), 2),
        (SwapPartMove(),                                                    3),
        (SwapPartForTileMove(),                                             3),
        (MergeAdjacentMove(),                                               4),
        (SplitPlacementMove(),                                              2),
        (AddSlopeCapMove(),                                                 3),
        (ReskinRoofMove(),                                                  5),
        (AddInsetLedgeMove(),                                               2),
    ]
    if mesh_occ is not None:
        pool.append((SurfacePolishMove(mesh_occ),                          5))
    if mesh_occ is not None and surface_mask is not None:
        pool.append((TrimInteriorMove(mesh_occ, surface_mask),              2))
    if region_placements is not None and registry is not None and target is not None:
        pool.append((RegionRebuildMove(region_placements, registry, target), 1))
    return pool


def sample_move(pool: list[tuple[Move, int]], rng: random.Random) -> Move:
    total = sum(w for _, w in pool)
    r = rng.random() * total
    acc = 0
    for m, w in pool:
        acc += w
        if r <= acc:
            return m
    return pool[-1][0]
