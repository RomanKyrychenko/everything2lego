"""RegionBuilder subclasses — one per primitive family.

Each Builder converts a Region into a list of `pack.Placement`. It uses
MILPPacker internally so its parts choice is provably near-optimal for its
own subproblem.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..bricks import Part, catalog
from ..colors import LegoColor, palette
from ..pack import Placement
from ..templates.base import part_by_ldraw
from .mesh_target import MeshTarget
from .milp import MILPPacker, PackResult, Placement2D
from .regions import (Region, RegionLabel, PlanarRegion, CylindricalRegion,
                        SphericalRegion, CornerRegion, FoundationRegion,
                        DetailRegion)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _nearest_palette(rgb: np.ndarray) -> int:
    """Return palette index for the given RGB."""
    from ..colors import _palette_lab as pallab
    pal = palette()
    # Use LAB nearest.
    lab_pal = pallab()  # shape (P, 3)
    # Convert one RGB to LAB.
    from .._compat import rgb_to_lab_single  # type: ignore
    return _nearest_palette_fallback(rgb, pal)


def _nearest_palette_fallback(rgb: np.ndarray, pal: tuple) -> int:
    """Simplest sRGB L2 distance in case Lab is unavailable."""
    r, g, b = float(rgb[0]), float(rgb[1]), float(rgb[2])
    best_i, best_d = 0, float("inf")
    for i, c in enumerate(pal):
        cr, cg, cb = c.rgb
        d = (r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2
        if d < best_d:
            best_d, best_i = d, i
    return best_i


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

@dataclass
class BuilderConfig:
    mm_per_stud: float = 8.0
    mm_per_plate: float = 3.2
    diversity_weight: float = 0.02
    milp_time_limit_s: float = 5.0


class RegionBuilder(ABC):
    """A RegionBuilder converts one Region into LEGO Placements.

    Builders receive the parent `MeshTarget` in `build()` so they can map
    mesh-mm coordinates through the voxelizer's fit transform before
    quantising to stud/plate indices.  This is the whole reason v2 v1.1
    had the coord-frame bug — surface builders MUST go through the
    transform.
    """

    def __init__(self, config: BuilderConfig | None = None) -> None:
        self.config = config or BuilderConfig()

    @abstractmethod
    def can_build(self, region: Region) -> bool:
        """Return True if this builder handles the given region."""

    @abstractmethod
    def build(self, region: Region, target: MeshTarget) -> list[Placement]:
        """Emit Placements for the region, mapped into voxel-index frame."""

    # ------------ shared utilities ------------------------------------

    def _plate_parts(self) -> list[Part]:
        return sorted([p for p in catalog() if p.shape == "plate"],
                       key=lambda p: -(p.footprint[0] * p.footprint[1]))

    def _tile_parts(self) -> list[Part]:
        return sorted([p for p in catalog() if p.shape == "tile"],
                       key=lambda p: -(p.footprint[0] * p.footprint[1]))

    def _brick_parts(self) -> list[Part]:
        return sorted([p for p in catalog() if p.shape == "brick"],
                       key=lambda p: -(p.footprint[0] * p.footprint[1]))

    def _footprints(self, parts: list[Part]) -> list[tuple[int, int]]:
        return [p.footprint for p in parts]


# ---------------------------------------------------------------------------
# Foundation
# ---------------------------------------------------------------------------

class FoundationBuilder(RegionBuilder):
    """A horizontal slab at the model's ground level.

    Tiles the slab's XZ footprint with the largest available plates, using
    MILPPacker for optimality on the 2D mask.
    """

    def can_build(self, region: Region) -> bool:
        return isinstance(region, FoundationRegion) or (
            isinstance(region, PlanarRegion) and region.is_horizontal
            and float(region.plane_normal[1]) > 0.85
        )

    def build(self, region: Region, target: MeshTarget) -> list[Placement]:
        assert isinstance(region, PlanarRegion)
        return _pack_horizontal_slab(
            region, target=target,
            part_family=self._plate_parts(),
            y_plate=0,  # ground level
            config=self.config,
        )


# ---------------------------------------------------------------------------
# Planar wall
# ---------------------------------------------------------------------------

class PlanarWallBuilder(RegionBuilder):
    """A vertical planar region.

    Builds the wall as a stack of brick courses. Each course is a 1-stud-thick
    strip whose length equals the wall's horizontal extent.  MILPPacker
    optimally chooses which brick fits at each horizontal position.
    """

    def can_build(self, region: Region) -> bool:
        return (isinstance(region, PlanarRegion) and region.is_vertical
                and not isinstance(region, FoundationRegion))

    def build(self, region: Region, target: MeshTarget) -> list[Placement]:
        assert isinstance(region, PlanarRegion)
        return _pack_vertical_wall(
            region, target=target,
            part_family=self._brick_parts(),
            config=self.config,
        )


# ---------------------------------------------------------------------------
# Planar roof
# ---------------------------------------------------------------------------

class PlanarRoofBuilder(RegionBuilder):
    """A tilted planar region (roof slope).

    Emits a proper slope-ring: the roof's *bottom edge* (eave) is a row of
    45° slope bricks (3040), and each subsequent brick row up the pitch is
    inset one stud closer to the ridge.  Interior plates fill behind the
    slope layer so the roof is solid inside.
    """

    def can_build(self, region: Region) -> bool:
        return (isinstance(region, PlanarRegion) and region.is_tilted_up
                and region.label != RegionLabel.FOUNDATION)

    def build(self, region: Region, target: MeshTarget) -> list[Placement]:
        assert isinstance(region, PlanarRegion)
        return _pack_roof_pitch(region, target=target, config=self.config)


# ---------------------------------------------------------------------------
# Cylinder / Sphere — Phase 4 stubs
# ---------------------------------------------------------------------------

class CylindricalBuilder(RegionBuilder):
    """A cylinder — round tower.

    Walks height in brick-row layers. At each layer, tiles a ring at the
    fitted radius using curved slopes (15068 curved 2x2 primitives) and
    cheese slopes (54200) to fill orientation gaps.
    """
    def can_build(self, region: Region) -> bool:
        return isinstance(region, CylindricalRegion)

    def build(self, region: Region, target: MeshTarget) -> list[Placement]:
        assert isinstance(region, CylindricalRegion)
        return _pack_cylinder_shell(region, target=target, config=self.config)


class SphericalBuilder(RegionBuilder):
    """A sphere or dome.

    Walks latitudes (Y-slices). Each latitude is a ring at radius
    √(R² - (y − cy)²). Tiles each ring with curved slopes oriented radially,
    cheese slopes near the poles where the ring is too small for 2×2.
    """
    def can_build(self, region: Region) -> bool:
        return isinstance(region, SphericalRegion)

    def build(self, region: Region, target: MeshTarget) -> list[Placement]:
        assert isinstance(region, SphericalRegion)
        return _pack_sphere_shell(region, target=target, config=self.config)


# ---------------------------------------------------------------------------
# Fallback: voxelise + greedy pack (delegates to v1 export)
# ---------------------------------------------------------------------------

class GenericVoxelBuilder(RegionBuilder):
    """The safety net for DetailRegion.

    Voxelises the region's local bounding box (points snap to LEGO stud/
    plate grid) and packs the result with the size-first-across-positions
    biggest-first greedy — same algorithm we validated in v1 export, now
    scoped to a single region.
    """

    def can_build(self, region: Region) -> bool:
        return isinstance(region, DetailRegion)

    def build(self, region: Region, target: MeshTarget) -> list[Placement]:
        return _voxel_pack_region(region, target=target, config=self.config)


# ---------------------------------------------------------------------------
# Builder registry — dispatches a region to the first matching builder
# ---------------------------------------------------------------------------

class BuilderRegistry:
    """Ordered list of builders; first `can_build` wins."""

    def __init__(self, builders: Optional[list[RegionBuilder]] = None) -> None:
        self.builders: list[RegionBuilder] = builders or self._default_builders()

    @staticmethod
    def _default_builders() -> list[RegionBuilder]:
        return [
            FoundationBuilder(),
            PlanarWallBuilder(),
            PlanarRoofBuilder(),
            CylindricalBuilder(),
            SphericalBuilder(),
            GenericVoxelBuilder(),  # last-resort
        ]

    def dispatch(self, region: Region) -> RegionBuilder | None:
        for b in self.builders:
            if b.can_build(region):
                return b
        return None


# ---------------------------------------------------------------------------
# Internal — packing helpers used by concrete builders
# ---------------------------------------------------------------------------

def _pack_horizontal_slab(region: PlanarRegion, *,
                            target: MeshTarget,
                            part_family: list[Part],
                            y_plate: int,
                            config: BuilderConfig) -> list[Placement]:
    """Rasterise the region's XZ points into a stud grid and pack it.

    Coordinates: region.points are in mesh-mm; we map each through
    `target.mesh_to_voxel` before quantising to a stud index.
    """
    mn_vox = target.mm_point_to_voxel(region.bounds[0])
    mx_vox = target.mm_point_to_voxel(region.bounds[1])
    x0 = int(np.floor(min(mn_vox[0], mx_vox[0])))
    z0 = int(np.floor(min(mn_vox[2], mx_vox[2])))
    x1 = int(np.ceil(max(mn_vox[0], mx_vox[0])))
    z1 = int(np.ceil(max(mn_vox[2], mx_vox[2])))
    W, D = max(1, x1 - x0), max(1, z1 - z0)

    # Coverage mask: which stud cells this region needs.
    mask = np.zeros((W, D), dtype=bool)
    for p in region.points:
        v = target.mm_point_to_voxel(p)
        ix = int(np.floor(v[0])) - x0
        iz = int(np.floor(v[2])) - z0
        if 0 <= ix < W and 0 <= iz < D:
            mask[ix, iz] = True

    if not mask.any():
        return []

    footprints = _distinct_footprints(part_family)
    packer = MILPPacker(parts=footprints,
                         diversity_weight=config.diversity_weight,
                         time_limit_s=config.milp_time_limit_s)
    result = packer.pack(mask)

    # Dominant palette colour for the region.
    color_idx = _dominant_palette(region.colors)
    part_lookup = _lookup_parts_by_footprint(part_family)
    out: list[Placement] = []
    for pl in result.placements:
        fw, fd = footprints[pl.part_index]
        part = part_lookup.get((fw, fd)) or part_lookup.get((fd, fw))
        if part is None:
            continue
        out.append(Placement(
            part=part, color_idx=color_idx,
            x_stud=x0 + pl.x, z_stud=z0 + pl.z, y_plate=y_plate,
            w_world=pl.w, d_world=pl.d, rotation_deg=pl.rotation_deg,
        ))
    return out


def _pack_vertical_wall(region: PlanarRegion, *,
                         target: MeshTarget,
                         part_family: list[Part],
                         config: BuilderConfig) -> list[Placement]:
    """Discretise the vertical plane into brick courses and pack each course.

    Uses target.mesh_to_voxel for correct stud/plate quantisation.
    """
    mn_vox = target.mm_point_to_voxel(region.bounds[0])
    mx_vox = target.mm_point_to_voxel(region.bounds[1])

    span_x = float(abs(mx_vox[0] - mn_vox[0]))
    span_z = float(abs(mx_vox[2] - mn_vox[2]))
    axis_x = span_x >= span_z
    length_studs = max(1, int(np.ceil(span_x if axis_x else span_z)))

    y_bottom = int(np.floor(min(mn_vox[1], mx_vox[1])))
    y_top = int(np.ceil(max(mn_vox[1], mx_vox[1])))
    height_plates = max(1, y_top - y_bottom)

    coverage = np.zeros(length_studs, dtype=bool)
    origin_stud = int(np.floor(min(mn_vox[0], mx_vox[0]) if axis_x
                                 else min(mn_vox[2], mx_vox[2])))
    for p in region.points:
        v = target.mm_point_to_voxel(p)
        c = int(np.floor(v[0] if axis_x else v[2])) - origin_stud
        if 0 <= c < length_studs:
            coverage[c] = True
    if not coverage.any():
        return []

    footprints = _distinct_footprints(part_family)
    packer = MILPPacker(parts=footprints,
                         diversity_weight=config.diversity_weight,
                         time_limit_s=config.milp_time_limit_s)

    color_idx = _dominant_palette(region.colors)
    part_lookup = _lookup_parts_by_footprint(part_family)

    other_origin = int(np.floor((mn_vox[2] if axis_x else mn_vox[0])))
    other_axis_ext = 1  # 1-stud thick wall for v1
    out: list[Placement] = []
    course_plates = 3
    n_courses = height_plates // course_plates
    if n_courses == 0:
        n_courses = 1
    for course in range(n_courses):
        y = y_bottom + course * course_plates
        mask = np.zeros((length_studs, other_axis_ext), dtype=bool)
        mask[coverage, 0] = True
        result = packer.pack(mask)
        for pl in result.placements:
            fw, fd = footprints[pl.part_index]
            part = part_lookup.get((fw, fd)) or part_lookup.get((fd, fw))
            if part is None:
                continue
            if axis_x:
                px, pz = origin_stud + pl.x, other_origin
            else:
                px, pz = other_origin, origin_stud + pl.x
            out.append(Placement(
                part=part, color_idx=color_idx,
                x_stud=px, z_stud=pz, y_plate=y,
                w_world=pl.w if axis_x else 1,
                d_world=1 if axis_x else pl.w,
                rotation_deg=0 if axis_x else 90,
            ))
    return out


def _voxel_pack_region(region: DetailRegion, *,
                          target: MeshTarget,
                          config: BuilderConfig) -> list[Placement]:
    """Fallback: rasterise the region's points into the LEGO grid and pack
    with the biggest-first plate + brick greedy."""
    pts = region.points
    if len(pts) == 0:
        return []

    pts_vox = np.array([target.mm_point_to_voxel(p) for p in pts],
                         dtype=np.float32)
    x0 = int(np.floor(pts_vox[:, 0].min()))
    z0 = int(np.floor(pts_vox[:, 2].min()))
    y0 = int(np.floor(pts_vox[:, 1].min()))
    x1 = int(np.ceil(pts_vox[:, 0].max()))
    z1 = int(np.ceil(pts_vox[:, 2].max()))
    y1 = int(np.ceil(pts_vox[:, 1].max()))
    W = max(1, x1 - x0)
    D = max(1, z1 - z0)
    H = max(1, y1 - y0)
    occ = np.zeros((H, W, D), dtype=bool)
    for v in pts_vox:
        yi = int(np.floor(v[1])) - y0
        xi = int(np.floor(v[0])) - x0
        zi = int(np.floor(v[2])) - z0
        if 0 <= yi < H and 0 <= xi < W and 0 <= zi < D:
            occ[yi, xi, zi] = True
    if not occ.any():
        return []

    color_idx = _dominant_palette(region.colors)
    plate_lookup = _lookup_parts_by_footprint(
        sorted([p for p in catalog() if p.shape == "plate"],
                 key=lambda p: -(p.footprint[0] * p.footprint[1])))
    plate_prefs = sorted({fp for fp in plate_lookup.keys()},
                          key=lambda wd: (-wd[0] * wd[1], -max(wd)))
    out: list[Placement] = []
    for y in range(H):
        layer = occ[y]
        if not layer.any():
            continue
        filled = ~layer.copy()
        for pw, pd in plate_prefs:
            part = plate_lookup.get((pw, pd))
            if part is None:
                continue
            rot = 0 if (pw, pd) == part.footprint else 90
            for x in range(W - pw + 1):
                for z in range(D - pd + 1):
                    if filled[x, z]:
                        continue
                    if filled[x:x + pw, z:z + pd].any():
                        continue
                    out.append(Placement(
                        part=part, color_idx=color_idx,
                        x_stud=x0 + x, z_stud=z0 + z, y_plate=y0 + y,
                        w_world=pw, d_world=pd, rotation_deg=rot,
                    ))
                    filled[x:x + pw, z:z + pd] = True
    return out


def _pack_sphere_shell(region: SphericalRegion, *,
                          target: MeshTarget,
                          config: BuilderConfig) -> list[Placement]:
    """Emit a spherical shell of curved slopes + cheese slopes.

    For each stud-column (x, z) that intersects the sphere, place a cheese
    slope 54200 at the top of the intersection with the outward-facing
    orientation, and stack plates below. This is the simplest sphere-
    building technique that produces a genuinely round surface at any
    resolution.
    """
    if region.radius <= 0:
        return []

    # Map sphere center + radius from mesh-mm to voxel-index (anisotropic).
    c_vox = target.mm_point_to_voxel(region.center)
    cx, cy, cz = float(c_vox[0]), float(c_vox[1]), float(c_vox[2])
    r_stud = target.mm_length_to_stud(region.radius)
    r_plate = target.mm_length_to_plate(region.radius)

    x_min = int(np.floor(cx - r_stud))
    x_max = int(np.ceil(cx + r_stud))
    z_min = int(np.floor(cz - r_stud))
    z_max = int(np.ceil(cz + r_stud))
    y_min = max(0, int(np.floor(cy - r_plate)))
    y_max = int(np.ceil(cy + r_plate))

    color_idx = _dominant_palette(region.colors)
    cheese = part_by_ldraw("54200.dat")
    curved = part_by_ldraw("15068.dat")
    plate_1x1 = part_by_ldraw("3024.dat")
    if cheese is None or plate_1x1 is None:
        return []

    out: list[Placement] = []
    # 1) Cheese-slope skin: at each (x, z) column intersecting the sphere,
    #    place a cheese slope at the top of the intersected Y range,
    #    oriented radially outward. Plates below fill the interior column.
    for x in range(x_min, x_max):
        for z in range(z_min, z_max):
            # Column horizontal distance from center.
            dx = (x + 0.5) - cx
            dz = (z + 0.5) - cz
            horiz2 = dx * dx + dz * dz
            if horiz2 > r_stud * r_stud:
                continue
            horiz_frac = horiz2 / (r_stud * r_stud + 1e-9)
            # Vertical half-extent of the column inside the sphere.
            # Vertical half-extent inside the sphere.  We converted the
            # radius separately to both stud- and plate-units already,
            # so the same fraction of vertical extent is r_plate * √(…).
            v_extent_plate = np.sqrt(max(0.0, 1.0 - horiz_frac)) * r_plate
            col_top = int(np.floor(cy + v_extent_plate))
            col_bot = int(np.ceil(cy - v_extent_plate))
            if col_top < col_bot:
                continue
            # Top of the column gets a cheese slope facing outward.
            # Pick orientation by which horizontal axis dominates.
            if abs(dx) >= abs(dz):
                rot = 0 if dx > 0 else 180
            else:
                rot = 270 if dz > 0 else 90
            out.append(Placement(
                part=cheese, color_idx=color_idx,
                x_stud=x, z_stud=z, y_plate=col_top,
                w_world=1, d_world=1, rotation_deg=rot,
            ))
            # Plates below to keep the column solid down to col_bot.
            for y in range(col_bot, col_top):
                out.append(Placement(
                    part=plate_1x1, color_idx=color_idx,
                    x_stud=x, z_stud=z, y_plate=y,
                    w_world=1, d_world=1, rotation_deg=0,
                ))
    return out


def _pack_cylinder_shell(region: CylindricalRegion, *,
                            target: MeshTarget,
                            config: BuilderConfig) -> list[Placement]:
    """Emit a cylindrical shell — one ring per brick-height layer."""
    if region.radius <= 0 or region.height <= 0:
        return []
    if abs(float(region.axis[1])) < 0.8:
        return []   # oblique cylinders are Phase 4.5

    c_vox = target.mm_point_to_voxel(region.center)
    cx, cz = float(c_vox[0]), float(c_vox[2])
    r_stud = target.mm_length_to_stud(region.radius)
    half_h_plate = target.mm_length_to_plate(region.height / 2)
    y_min = int(np.floor(c_vox[1] - half_h_plate))
    y_max = int(np.ceil(c_vox[1] + half_h_plate))
    x_min = int(np.floor(cx - r_stud))
    x_max = int(np.ceil(cx + r_stud))
    z_min = int(np.floor(cz - r_stud))
    z_max = int(np.ceil(cz + r_stud))

    color_idx = _dominant_palette(region.colors)
    cheese = part_by_ldraw("54200.dat")
    plate_1x1 = part_by_ldraw("3024.dat")
    if cheese is None or plate_1x1 is None:
        return []

    out: list[Placement] = []
    for y in range(y_min, y_max):
        # Ring cells at this y-plate: cells within `r_stud` of (cx, cz).
        for x in range(x_min, x_max):
            for z in range(z_min, z_max):
                dx = (x + 0.5) - cx
                dz = (z + 0.5) - cz
                d2 = dx * dx + dz * dz
                if d2 > r_stud * r_stud:
                    continue
                # Distance to the ring surface.
                d = np.sqrt(d2)
                if d >= r_stud - 0.6:  # outer ring — slope skin
                    if abs(dx) >= abs(dz):
                        rot = 0 if dx > 0 else 180
                    else:
                        rot = 270 if dz > 0 else 90
                    out.append(Placement(
                        part=cheese, color_idx=color_idx,
                        x_stud=x, z_stud=z, y_plate=y,
                        w_world=1, d_world=1, rotation_deg=rot,
                    ))
                else:  # interior fill for solid cylinder
                    out.append(Placement(
                        part=plate_1x1, color_idx=color_idx,
                        x_stud=x, z_stud=z, y_plate=y,
                        w_world=1, d_world=1, rotation_deg=0,
                    ))
    return out


def _pack_roof_pitch(region: PlanarRegion, *,
                       target: MeshTarget,
                       config: BuilderConfig) -> list[Placement]:
    """Emit a slope-ring for a tilted planar roof region.

    Geometry derived from POINTS (not the normal) so it works with either
    winding:
      - Ridge = points at max Y.
      - Eave  = points at min Y.
      - Pitch axis = the horizontal axis where eave-center and ridge-center
        differ most.
      - Pitch sign = +1 if ridge is at higher coord than eave in the pitch
        axis, else -1.
      - Outward direction (visible slanted face) = OPPOSITE of pitch_sign
        in the pitch axis.  Slope 3040 is placed so its low side faces
        outward.
    """
    slope_part = part_by_ldraw("3040.dat")
    if slope_part is None:
        return []
    pts_vox = np.array([target.mm_point_to_voxel(p) for p in region.points],
                         dtype=np.float32)
    y_min = float(pts_vox[:, 1].min())
    y_max = float(pts_vox[:, 1].max())
    if y_max - y_min < 1.0:
        return []

    # Sample top-row and bottom-row points to find ridge and eave centres.
    y_band = max(1.0, (y_max - y_min) * 0.15)
    top_pts = pts_vox[pts_vox[:, 1] > y_max - y_band]
    bot_pts = pts_vox[pts_vox[:, 1] < y_min + y_band]
    if len(top_pts) < 2 or len(bot_pts) < 2:
        return []
    top_c = top_pts.mean(axis=0)
    bot_c = bot_pts.mean(axis=0)

    # Pitch axis = horizontal axis of the eave→ridge vector.
    dx = float(top_c[0] - bot_c[0])
    dz = float(top_c[2] - bot_c[2])
    if abs(dx) < 0.5 and abs(dz) < 0.5:
        return []
    if abs(dx) > abs(dz):
        pitch_axis = "x"
        pitch_sign = 1 if dx > 0 else -1
        ridge_min = float(pts_vox[:, 2].min())
        ridge_max = float(pts_vox[:, 2].max())
        eave_pitch = float(bot_c[0])
    else:
        pitch_axis = "z"
        pitch_sign = 1 if dz > 0 else -1
        ridge_min = float(pts_vox[:, 0].min())
        ridge_max = float(pts_vox[:, 0].max())
        eave_pitch = float(bot_c[2])

    ridge_length_studs = max(1, int(round(ridge_max - ridge_min)))
    pitch_rise_plates = max(1, int(round(y_max - y_min)))
    n_rows = max(1, pitch_rise_plates // 3)

    # Slope 3040 (2x1, 3 plates). LDraw default low_dir = +x. Rotations
    # cycle (+x → -z → -x → +z) per 90° via pack._LOW_DIR_CYCLE.
    # We want the LOW side of the slope to face OUTWARD (opposite pitch_sign).
    if pitch_axis == "x":
        # outward = -pitch_sign · X. Low +x ⇒ rot 0; low -x ⇒ rot 180.
        rot = 180 if pitch_sign > 0 else 0
    else:
        # outward = -pitch_sign · Z. Low +z ⇒ rot 270; low -z ⇒ rot 90.
        rot = 90 if pitch_sign > 0 else 270

    color_idx = _dominant_palette(region.colors)
    ridge_origin = int(np.floor(ridge_min))
    eave_origin = int(np.floor(eave_pitch))
    y_bottom = int(np.floor(y_min))

    out: list[Placement] = []
    for row in range(n_rows):
        y_plate = y_bottom + row * 3
        # Row inset from eave TOWARD ridge = row * pitch_sign studs.
        current = eave_origin + row * pitch_sign
        # Anchor so the LOW-side edge sits at `current` (outward face).
        if pitch_sign > 0:
            # Ridge is at + direction; part extends from `current` inward.
            anchor_pitch = current
        else:
            # Ridge is at - direction; part occupies (current-1, current).
            anchor_pitch = current - 1
        for j in range(ridge_length_studs):
            r = ridge_origin + j
            if pitch_axis == "x":
                out.append(Placement(
                    part=slope_part, color_idx=color_idx,
                    x_stud=anchor_pitch, z_stud=r, y_plate=y_plate,
                    w_world=2, d_world=1, rotation_deg=rot,
                ))
            else:
                out.append(Placement(
                    part=slope_part, color_idx=color_idx,
                    x_stud=r, z_stud=anchor_pitch, y_plate=y_plate,
                    w_world=1, d_world=2, rotation_deg=rot,
                ))
    return out


def _distinct_footprints(parts: list[Part]) -> list[tuple[int, int]]:
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for p in parts:
        fp = p.footprint
        if fp in seen:
            continue
        seen.add(fp)
        out.append(fp)
    return out


def _lookup_parts_by_footprint(parts: list[Part]) -> dict[tuple[int, int], Part]:
    lookup: dict[tuple[int, int], Part] = {}
    for p in parts:
        lookup.setdefault(p.footprint, p)
        # Also register rotated variant.
        w, d = p.footprint
        if w != d:
            lookup.setdefault((d, w), p)
    return lookup


def _dominant_palette(rgb_samples: np.ndarray) -> int:
    """Palette index that best matches the majority colour of the samples."""
    if len(rgb_samples) == 0:
        return 0
    pal = palette()
    # Quantise each sample to nearest palette then vote.
    pal_rgb = np.array([c.rgb for c in pal], dtype=np.int32)
    diffs = rgb_samples[:, None, :].astype(np.int32) - pal_rgb[None, :, :]
    d2 = (diffs * diffs).sum(axis=2)
    nearest = d2.argmin(axis=1)
    return int(Counter(nearest.tolist()).most_common(1)[0][0])
