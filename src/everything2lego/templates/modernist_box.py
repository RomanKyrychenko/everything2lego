"""Modernist box template — flat-roof house with large glazed openings.

Fundamentally different from `GableCabin`: no pitch, no cornice, no quoins.
Instead: a monolithic slab foundation, contrasting-band walls, a flat
parapet roof capped in tiles, and large picture-window openings arranged
in horizontal ribbons on the long walls.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..pack import Placement
from .base import (Template, color_index_by_code, part_by_ldraw, place_rect,
                    tile_area_with_plates, tile_area_with_tiles)


class ModernistBox(Template):
    """Rectangular flat-roof house — think International Style."""

    name = "modernist-box"

    @dataclass
    class Params:
        width_studs: int = 40           # +X dimension
        depth_studs: int = 20           # +Z dimension
        wall_height_bricks: int = 6
        foundation_plates: int = 2
        parapet_bricks: int = 1         # additional wall above the flat roof deck
        wall_color_code: int = 15       # White
        band_color_code: int = 72       # Dark Bluish Gray (horizontal band)
        band_at_brick_row: int = 3      # 0-indexed brick row that gets the band
        foundation_color_code: int = 0  # Black
        roof_deck_color_code: int = 71  # Light Bluish Gray
        parapet_color_code: int = 72    # Dark Bluish Gray
        window_color_code: int = 33     # Trans Light Blue
        door_color_code: int = 0        # Black
        ribbon_windows_per_long_wall: int = 4
        window_row_brick: int = 2       # 0-indexed brick row for windows

    def build(self, params: "ModernistBox.Params") -> list[Placement]:
        p = params
        out: list[Placement] = []
        out.extend(self._foundation(p))
        out.extend(self._walls(p))
        out.extend(self._roof(p))
        out.extend(self._openings(p))
        return out

    def _foundation(self, p) -> list[Placement]:
        color_idx = color_index_by_code(p.foundation_color_code)
        out: list[Placement] = []
        for y in range(p.foundation_plates):
            out.extend(tile_area_with_plates(
                0, 0, p.width_studs, p.depth_studs,
                y=y, color_idx=color_idx,
            ))
        return out

    def _walls(self, p) -> list[Placement]:
        base_y = p.foundation_plates
        wall_color = color_index_by_code(p.wall_color_code)
        band_color = color_index_by_code(p.band_color_code)

        brick_1x2 = part_by_ldraw("3004.dat")
        brick_1x1 = part_by_ldraw("3005.dat")
        tile_1x2 = part_by_ldraw("3069b.dat")
        tile_1x1 = part_by_ldraw("3070b.dat")

        def wall_run(x0, z0, length, *, axis, y, color_idx, tiles=False):
            remaining = length
            cursor = 0
            while remaining >= 2:
                part = tile_1x2 if tiles else brick_1x2
                rot = 0 if axis == "x" else 90
                pos = ((x0 + cursor, z0) if axis == "x"
                        else (x0, z0 + cursor))
                out.append(place_rect(part, x=pos[0], z=pos[1], y=y,
                                        color_idx=color_idx, rotation_deg=rot))
                cursor += 2
                remaining -= 2
            while remaining > 0:
                part = tile_1x1 if tiles else brick_1x1
                pos = ((x0 + cursor, z0) if axis == "x"
                        else (x0, z0 + cursor))
                out.append(place_rect(part, x=pos[0], z=pos[1], y=y,
                                        color_idx=color_idx))
                cursor += 1
                remaining -= 1

        out: list[Placement] = []
        total_rows = p.wall_height_bricks + p.parapet_bricks
        for brick_row in range(total_rows):
            y = base_y + brick_row * 3
            in_parapet = brick_row >= p.wall_height_bricks
            if in_parapet:
                row_color = color_index_by_code(p.parapet_color_code)
            elif brick_row == p.band_at_brick_row:
                row_color = band_color
            else:
                row_color = wall_color

            wall_run(0, 0, p.width_studs,
                     axis="x", y=y, color_idx=row_color)
            wall_run(0, p.depth_studs - 1, p.width_studs,
                     axis="x", y=y, color_idx=row_color)
            wall_run(0, 1, p.depth_studs - 2,
                     axis="z", y=y, color_idx=row_color)
            wall_run(p.width_studs - 1, 1, p.depth_studs - 2,
                     axis="z", y=y, color_idx=row_color)
        return out

    def _roof(self, p) -> list[Placement]:
        """Flat roof deck at wall-top. Interior tiled smoothly."""
        deck_y = p.foundation_plates + p.wall_height_bricks * 3
        deck_color = color_index_by_code(p.roof_deck_color_code)
        parapet_top = deck_y + p.parapet_bricks * 3
        out: list[Placement] = []
        # Roof deck: full-area tiles at deck_y (bottom of parapet).
        out.extend(tile_area_with_tiles(
            1, 1, p.width_studs - 2, p.depth_studs - 2,
            y=deck_y, color_idx=deck_color,
        ))
        # Parapet cap: 1-plate tile ring on top of the parapet.
        out.extend(tile_area_with_tiles(
            0, 0, p.width_studs, 1,
            y=parapet_top - 1, color_idx=color_index_by_code(p.parapet_color_code),
        ))
        out.extend(tile_area_with_tiles(
            0, p.depth_studs - 1, p.width_studs, 1,
            y=parapet_top - 1, color_idx=color_index_by_code(p.parapet_color_code),
        ))
        out.extend(tile_area_with_tiles(
            0, 1, 1, p.depth_studs - 2,
            y=parapet_top - 1, color_idx=color_index_by_code(p.parapet_color_code),
        ))
        out.extend(tile_area_with_tiles(
            p.width_studs - 1, 1, 1, p.depth_studs - 2,
            y=parapet_top - 1, color_idx=color_index_by_code(p.parapet_color_code),
        ))
        return out

    def _openings(self, p) -> list[Placement]:
        window = part_by_ldraw("60592.dat")   # Window 1x2x2
        door = part_by_ldraw("60623.dat")     # Door 1x4x6
        if window is None or door is None:
            return []
        base_y = p.foundation_plates
        window_color = color_index_by_code(p.window_color_code)
        door_color = color_index_by_code(p.door_color_code)
        window_y = base_y + p.window_row_brick * 3
        door_y = base_y

        out: list[Placement] = []
        door_x = p.width_studs // 2 - 2
        out.append(place_rect(door, x=door_x, z=0, y=door_y,
                                color_idx=door_color, rotation_deg=0))

        # Ribbon of picture windows on both long walls, evenly spaced away
        # from the door.
        n = p.ribbon_windows_per_long_wall
        step = p.width_studs // (n + 1)
        for i in range(1, n + 1):
            x = i * step - 1
            # Skip if too close to the door on the front wall.
            if not (door_x - 1 <= x <= door_x + 4):
                out.append(place_rect(window, x=x, z=0, y=window_y,
                                        color_idx=window_color, rotation_deg=0))
            out.append(place_rect(window, x=x, z=p.depth_studs - 1,
                                    y=window_y, color_idx=window_color,
                                    rotation_deg=0))
        return out
