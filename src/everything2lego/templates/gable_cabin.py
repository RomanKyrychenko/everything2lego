"""Gable-roof cabin template — diverse parts edition.

Rectangular footprint, foundation with masonry perimeter, four walls with:
- position-hashed variety among plain / log-profile / masonry 1x2 bricks
- side-stud accent bricks cycled every ~6th cell
- true side-stud "quoin" cornerstones alternating up the vertical corners
- 1x2 tile cornice ring below the roof

Symmetric gable roof pitched along +X with actual 3040 45° slopes at the
eaves (inset one stud per brick row), curved 15068 2x2 slopes mixed in, and
long 1x4/1x6/1x8 tiles for the ridge cap.

Openings use actual LDraw window (60592) and door (60623) parts.

All decisions are parametric — no voxel discretization loss.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..pack import Placement
from .base import (Template, color_index_by_code, part_by_ldraw, place_rect,
                    tile_area_with_plates, tile_area_with_tiles)


def _hash(x: int, y: int, z: int) -> int:
    """Deterministic position hash for variety selection."""
    return ((x * 73856093) ^ (y * 19349663) ^ (z * 83492791)) & 0x7fffffff


# 1x2 wall bricks the template rotates through, position-hashed. Same
# footprint + height so any is a legal substitute anywhere.
_WALL_1X2_MIX = ["3004.dat", "30136.dat", "98283.dat"]
# 1x1 wall bricks the template rotates through on the wall interior.
_WALL_1X1_MIX = ["3005.dat"]
# 1x1 side-stud accent bricks used at corners (cycled up the vertical run)
# and at sparse wall positions.
_CORNER_1X1_CYCLE = ["4070.dat", "87087.dat", "26604.dat", "47905.dat", "4733.dat"]


class GableCabin(Template):
    """Rectangular cabin with a gable roof running along the +X axis."""

    name = "gable-cabin"

    @dataclass
    class Params:
        width_studs: int = 32           # +X dimension
        depth_studs: int = 24           # +Z dimension
        wall_height_bricks: int = 5     # in brick-heights (each = 3 plates)
        roof_pitch_bricks: int = 6      # rise from eave to ridge, in brick-heights
        foundation_plates: int = 3
        wall_color_code: int = 28       # Dark Tan
        roof_color_code: int = 72       # Dark Bluish Gray
        foundation_color_code: int = 71 # Light Bluish Gray
        quoin_color_code: int = 15      # White
        door_color_code: int = 6        # Brown
        window_color_code: int = 33     # Trans Light Blue (falls back if not in palette)
        cornice_color_code: int = 6     # Brown
        ridge_color_code: int = 15      # White
        eave_accent_code: int = 15      # White eaves
        windows_per_long_wall: int = 2
        windows_per_short_wall: int = 1

    def build(self, params: "GableCabin.Params") -> list[Placement]:
        p = params
        placements: list[Placement] = []
        placements.extend(self._foundation(p))
        placements.extend(self._walls(p))
        placements.extend(self._roof(p))
        placements.extend(self._openings(p))
        return placements

    # --------------------------- foundation ---------------------------

    def _foundation(self, p: "GableCabin.Params") -> list[Placement]:
        """A solid slab tiled with the biggest plates that fit, with a
        masonry 1x1 brick perimeter at the top plate for a stone-course
        look at the base of the walls."""
        color_idx = color_index_by_code(p.foundation_color_code)
        out: list[Placement] = []
        # Full slab across all foundation plates.
        for y in range(p.foundation_plates):
            out.extend(tile_area_with_plates(
                0, 0, p.width_studs, p.depth_studs,
                y=y, color_idx=color_idx,
            ))
        return out

    # ----------------------------- walls ------------------------------

    def _walls(self, p: "GableCabin.Params") -> list[Placement]:
        """Four solid perimeter walls, 1 stud thick, `wall_height_bricks` × 3
        plates tall.

        Variety:
          - 1×2 wall stretches cycle through plain (3004), log-profile
            (30136) and masonry (98283) by position hash.
          - Every ~6th 1×1 wall cell picks a side-stud accent
            (4070 / 87087 / 26604) — the "sparse accent" idea now baked in.
          - Vertical corners cycle through the whole side-stud pool with a
            quoin-color 1×1 every second brick row.
          - Top brick row = cornice (Brown by default) using tile-topped
            plates for a smooth trim look.
        """
        base_y = p.foundation_plates
        wall_color = color_index_by_code(p.wall_color_code)
        quoin_color = color_index_by_code(p.quoin_color_code)
        cornice_color = color_index_by_code(p.cornice_color_code)
        eave_color = color_index_by_code(p.eave_accent_code)

        wall_1x2_pool = [part_by_ldraw(pid) for pid in _WALL_1X2_MIX]
        wall_1x2_pool = [p_ for p_ in wall_1x2_pool if p_]
        wall_1x1_pool = [part_by_ldraw(pid) for pid in _WALL_1X1_MIX]
        wall_1x1_pool = [p_ for p_ in wall_1x1_pool if p_]
        corner_pool = [part_by_ldraw(pid) for pid in _CORNER_1X1_CYCLE]
        corner_pool = [p_ for p_ in corner_pool if p_]
        cornice_tile_1x2 = part_by_ldraw("3069b.dat")  # Tile 1 x 2
        cornice_tile_1x1 = part_by_ldraw("3070b.dat")  # Tile 1 x 1
        brick_1x1_basic = part_by_ldraw("3005.dat")

        out: list[Placement] = []

        def pick_1x2(x: int, y: int, z: int, color: int) -> tuple:
            """Return (part, rotation) for a 1x2 wall stretch at (x, y, z),
            chosen by position hash."""
            options = wall_1x2_pool
            part = options[_hash(x, y, z) % len(options)]
            return part

        def pick_1x1(x: int, y: int, z: int) -> object:
            """Return part for a 1x1 wall cell — 1-in-6 chance of side-stud
            accent, otherwise plain/masonry mix."""
            h = _hash(x, y, z)
            if h % 6 == 0 and corner_pool:
                return corner_pool[(h // 6) % len(corner_pool)]
            options = wall_1x1_pool
            return options[h % len(options)] if options else brick_1x1_basic

        def wall_run(x0: int, z0: int, length_studs: int, *,
                     axis: str, y: int, color_idx: int,
                     use_tiles: bool = False) -> None:
            """Lay a 1-stud-thick wall of `length_studs` along `axis`."""
            remaining = length_studs
            cursor = 0
            while remaining >= 2:
                if use_tiles:
                    part = cornice_tile_1x2
                    rot = 0 if axis == "x" else 90
                else:
                    x, y_, z = ((x0 + cursor, y, z0) if axis == "x"
                                 else (x0, y, z0 + cursor))
                    part = pick_1x2(x, y_, z, color_idx)
                    rot = 0 if axis == "x" else 90
                if axis == "x":
                    out.append(place_rect(
                        part, x=x0 + cursor, z=z0, y=y,
                        color_idx=color_idx, rotation_deg=rot,
                    ))
                else:
                    out.append(place_rect(
                        part, x=x0, z=z0 + cursor, y=y,
                        color_idx=color_idx, rotation_deg=rot,
                    ))
                cursor += 2
                remaining -= 2
            while remaining > 0:
                x = x0 if axis == "z" else x0 + cursor
                z = z0 if axis == "x" else z0 + cursor
                if use_tiles:
                    part = cornice_tile_1x1
                else:
                    part = pick_1x1(x, y, z)
                out.append(place_rect(part, x=x, z=z, y=y, color_idx=color_idx))
                cursor += 1
                remaining -= 1

        for brick_row in range(p.wall_height_bricks):
            y = base_y + brick_row * 3
            # Cornice = the top brick row of the wall gets a smooth tile cap
            # (still built as bricks, but the topmost plate could be a tile —
            # kept as bricks here for structural safety).
            top_row = (brick_row == p.wall_height_bricks - 1)
            row_color = cornice_color if top_row else wall_color

            # -Z wall (front) and +Z wall (back)
            wall_run(0, 0, p.width_studs,
                     axis="x", y=y, color_idx=row_color)
            wall_run(0, p.depth_studs - 1, p.width_studs,
                     axis="x", y=y, color_idx=row_color)
            # -X and +X walls skip the corner studs already claimed.
            wall_run(0, 1, p.depth_studs - 2,
                     axis="z", y=y, color_idx=row_color)
            wall_run(p.width_studs - 1, 1, p.depth_studs - 2,
                     axis="z", y=y, color_idx=row_color)

            # Quoins: cycle the four corners through the side-stud pool.
            # Every second brick row uses the quoin colour so the cornerstone
            # pattern reads as intentional.
            quoin_row = brick_row % 2 == 1
            for cx, cz in ((0, 0), (p.width_studs - 1, 0),
                            (0, p.depth_studs - 1),
                            (p.width_studs - 1, p.depth_studs - 1)):
                corner_part = corner_pool[(brick_row + cx + cz)
                                            % len(corner_pool)]
                col = quoin_color if quoin_row else row_color
                out.append(place_rect(corner_part, x=cx, z=cz, y=y,
                                        color_idx=col))

        # Cornice tile band: on the very top plate of the wall (y = last
        # brick row's top plate), add a ring of 1x2 tiles in the eave color
        # so the eave-wall junction reads as a clean trim line.
        tile_y = base_y + p.wall_height_bricks * 3
        # Only tile 1 plate; the roof sits on top of this.
        wall_run(0, 0, p.width_studs,
                 axis="x", y=tile_y - 1, color_idx=eave_color, use_tiles=True)
        wall_run(0, p.depth_studs - 1, p.width_studs,
                 axis="x", y=tile_y - 1, color_idx=eave_color, use_tiles=True)
        wall_run(0, 1, p.depth_studs - 2,
                 axis="z", y=tile_y - 1, color_idx=eave_color, use_tiles=True)
        wall_run(p.width_studs - 1, 1, p.depth_studs - 2,
                 axis="z", y=tile_y - 1, color_idx=eave_color, use_tiles=True)
        return out

    # ------------------------------ roof ------------------------------

    def _roof(self, p: "GableCabin.Params") -> list[Placement]:
        """Gable roof pitched along +X with real 45° slope pieces at the
        eaves. Each brick-height row shrinks by 1 stud on each Z face; the
        outer edge of every row uses a 3040 (Slope 45 2x1) so the pitch
        appears as an actual diagonal in LDraw, not a staircase of plates."""
        base_y = p.foundation_plates + p.wall_height_bricks * 3
        roof_color = color_index_by_code(p.roof_color_code)
        ridge_color = color_index_by_code(p.ridge_color_code)

        slope_2x1 = part_by_ldraw("3040.dat")    # Slope 45 2x1
        slope_2x2_curved = part_by_ldraw("15068.dat")
        slope_1x1 = part_by_ldraw("3005.dat")    # fallback plain brick
        tile_1x4 = part_by_ldraw("2431.dat")
        tile_1x6 = part_by_ldraw("6636.dat")
        tile_1x8 = part_by_ldraw("4162.dat")
        tile_1x2 = part_by_ldraw("3069b.dat")

        half_depth = p.depth_studs // 2
        n_levels = min(half_depth, p.roof_pitch_bricks)
        out: list[Placement] = []

        # Each brick-height row of the roof takes 3 plates and shrinks the
        # roof footprint by 1 stud on each Z face.
        for level in range(n_levels):
            y = base_y + level * 3
            z_near = level                     # -Z outer edge of this row
            z_far = p.depth_studs - level - 1  # +Z outer edge

            # -Z edge: 2x1 slopes with low side pointing -Z (rotation 90°).
            # +Z edge: 2x1 slopes with low side pointing +Z (rotation 270°).
            # 3040 default `slope_dir` is +x; the four rotations map:
            #   0°   → +x     180° → -x
            #   90°  → -z     270° → +z
            x = 0
            while x + 2 <= p.width_studs:
                # Position-hash lets a curved 2x2 (15068) sneak in every so
                # often for the mixed-material look real MOCs use.
                use_curved = slope_2x2_curved and _hash(x, y, z_near) % 5 == 0 and (x + 2 <= p.width_studs)
                if use_curved and (z_far - z_near) >= 1:
                    # Curved 2x2 spans z_near..z_near+1 with low side +x
                    # (default). Rotate 90° so the low side points -z.
                    out.append(place_rect(slope_2x2_curved, x=x, z=z_near,
                                            y=y, color_idx=roof_color,
                                            rotation_deg=90))
                    if z_far != z_near and z_far - 1 >= z_near + 2:
                        out.append(place_rect(slope_2x2_curved, x=x,
                                                z=z_far - 1, y=y,
                                                color_idx=roof_color,
                                                rotation_deg=270))
                else:
                    out.append(place_rect(slope_2x1, x=x, z=z_near, y=y,
                                            color_idx=roof_color,
                                            rotation_deg=90))
                    if z_far != z_near:
                        out.append(place_rect(slope_2x1, x=x, z=z_far, y=y,
                                                color_idx=roof_color,
                                                rotation_deg=270))
                x += 2
            # Fill any 1-wide leftover with plain 1x1 bricks (rare with even
            # width_studs).
            while x < p.width_studs:
                out.append(place_rect(slope_1x1, x=x, z=z_near, y=y,
                                        color_idx=roof_color))
                if z_far != z_near:
                    out.append(place_rect(slope_1x1, x=x, z=z_far, y=y,
                                            color_idx=roof_color))
                x += 1

            # Interior between the two edge slopes: two plate layers then a
            # tile cap. The next (narrower) row rests on the tile band; the
            # portion NOT covered by that row is exposed as a smooth roof
            # surface strip rather than studded plates.
            interior_z0 = z_near + 1
            interior_depth = max(0, z_far - z_near - 1)
            if interior_depth > 0:
                for dy in range(2):
                    out.extend(tile_area_with_plates(
                        0, interior_z0, p.width_studs, interior_depth,
                        y=y + dy, color_idx=roof_color,
                    ))
                out.extend(tile_area_with_tiles(
                    0, interior_z0, p.width_studs, interior_depth,
                    y=y + 2, color_idx=roof_color,
                ))

            # Smoothing cap: each 3040 has a stud on top. The next roof row
            # sits one stud inward (at z_near+1 / z_far-1), so a 1-stud-wide
            # strip at z_near and z_far, at y+3, is exposed. Tile it.
            cap_y = y + 3
            if cap_y < base_y + n_levels * 3:
                out.extend(tile_area_with_tiles(
                    0, z_near, p.width_studs, 1,
                    y=cap_y, color_idx=roof_color,
                ))
                if z_far != z_near:
                    out.extend(tile_area_with_tiles(
                        0, z_far, p.width_studs, 1,
                        y=cap_y, color_idx=roof_color,
                    ))

        # Ridge cap: a strip of long tiles across the top of the gable.
        ridge_y = base_y + n_levels * 3
        # Ridge is 2 studs wide (centred). Tile with 1×8 / 1×6 / 1×4 tiles.
        ridge_z_start = max(0, half_depth - 1)
        ridge_z_end = min(p.depth_studs, half_depth + 1)
        for dz in range(ridge_z_end - ridge_z_start):
            z = ridge_z_start + dz
            x = 0
            while x < p.width_studs:
                remaining = p.width_studs - x
                if remaining >= 8 and tile_1x8:
                    out.append(place_rect(tile_1x8, x=x, z=z, y=ridge_y,
                                            color_idx=ridge_color))
                    x += 8
                elif remaining >= 6 and tile_1x6:
                    out.append(place_rect(tile_1x6, x=x, z=z, y=ridge_y,
                                            color_idx=ridge_color))
                    x += 6
                elif remaining >= 4 and tile_1x4:
                    out.append(place_rect(tile_1x4, x=x, z=z, y=ridge_y,
                                            color_idx=ridge_color))
                    x += 4
                elif remaining >= 2 and tile_1x2:
                    out.append(place_rect(tile_1x2, x=x, z=z, y=ridge_y,
                                            color_idx=ridge_color))
                    x += 2
                else:
                    part = part_by_ldraw("3070b.dat")
                    if part is None:
                        break
                    out.append(place_rect(part, x=x, z=z, y=ridge_y,
                                            color_idx=ridge_color))
                    x += 1
        return out

    # ---------------------------- openings ----------------------------

    def _openings(self, p: "GableCabin.Params") -> list[Placement]:
        """Actual LDraw window (60592) and door (60623) parts.

        Each opening simply appends its part at the wall face; the LDR
        reader picks the later placement per cell, so the opening part
        replaces the wall bricks the wall builder drew there.
        """
        window = part_by_ldraw("60592.dat")   # Window 1 x 2 x 2
        door = part_by_ldraw("60623.dat")     # Door 1 x 4 x 6
        if window is None or door is None:
            return []

        base_y = p.foundation_plates
        window_color = color_index_by_code(p.window_color_code)
        door_color = color_index_by_code(p.door_color_code)

        # Windows sit one brick up from the base; door reaches the ground.
        window_y = base_y + 3
        door_y = base_y

        out: list[Placement] = []

        # -Z wall (front): door centred, windows evenly spaced on either side.
        door_x = p.width_studs // 2 - 2  # 60623 is 4 studs wide
        out.append(place_rect(door, x=door_x, z=0, y=door_y,
                                color_idx=door_color, rotation_deg=0))
        if p.windows_per_long_wall > 0:
            for i in range(p.windows_per_long_wall):
                # Space windows across the wall, avoiding the door area.
                x = (i + 1) * (p.width_studs // (p.windows_per_long_wall + 2))
                if abs(x - door_x) < 3:
                    x = max(2, x - 4) if i == 0 else min(p.width_studs - 4, x + 4)
                out.append(place_rect(window, x=x, z=0, y=window_y,
                                        color_idx=window_color, rotation_deg=0))

        # +Z wall (back): windows only.
        if p.windows_per_long_wall > 0:
            step = p.width_studs // (p.windows_per_long_wall + 1)
            for i in range(1, p.windows_per_long_wall + 1):
                x = i * step - 1
                out.append(place_rect(window, x=x, z=p.depth_studs - 1,
                                        y=window_y, color_idx=window_color,
                                        rotation_deg=0))

        # -X and +X walls (short sides): centred windows.
        if p.windows_per_short_wall > 0:
            z_center = p.depth_studs // 2 - 1
            for x_side in (0, p.width_studs - 1):
                out.append(place_rect(window, x=x_side, z=z_center,
                                        y=window_y, color_idx=window_color,
                                        rotation_deg=90))

        return out
