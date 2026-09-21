"""Silhouette space-carving from a turntable image sequence.

Given N images of one object captured from evenly-spaced turntable angles
(default: 0, 360/N, ... degrees around the +Y axis, orthographic projection),
build a colored voxel grid at LEGO scale via the classical visual-hull method:

  1. Foreground/background segmentation per image. Background estimated from
     the four corner pixels, unless the caller supplies a hex color.
  2. Every voxel is projected into every view; a voxel is filled iff at least
     `min_views` views agree it lands on foreground.
  3. Transfer colors from visible surface pixels using a coarse depth buffer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage

from . import bricks, colors
from .voxelize import VoxelGrid


@dataclass
class View:
    image_rgb: np.ndarray   # (H, W, 3) uint8
    mask: np.ndarray        # (H, W) bool
    theta_deg: float        # rotation around +Y (CCW viewed from +Y)


def _parse_hex(hex_str: str) -> tuple[int, int, int]:
    s = hex_str.lstrip("#")
    if len(s) != 6:
        raise ValueError(f"Bad hex color: {hex_str}")
    return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)


def _auto_background(img: np.ndarray, sample: int = 8) -> np.ndarray:
    """Median color of corner patches."""
    h, w = img.shape[:2]
    s = sample
    patches = [img[:s, :s], img[:s, -s:], img[-s:, :s], img[-s:, -s:]]
    return np.median(np.concatenate([p.reshape(-1, 3) for p in patches], axis=0), axis=0)


def _foreground_mask(img: np.ndarray, background: np.ndarray | None,
                     tolerance: int = 28) -> np.ndarray:
    if background is None:
        background = _auto_background(img)
    diff = np.abs(img.astype(np.int16) - background.astype(np.int16)).max(axis=-1)
    return diff > tolerance


def load_views(paths: Sequence[str | Path], *,
               angles: Sequence[float] | None = None,
               background_hex: str | None = None,
               tolerance: int = 28,
               masks: Sequence[str | Path] | None = None) -> list[View]:
    """Load images and pair each with an angle around the +Y axis.

    If `angles` is None, angles are inferred as evenly-spaced turntable
    positions starting at 0 degrees.
    """
    paths = list(paths)
    if len(paths) < 2:
        raise ValueError("At least two views are required; four or more are recommended")
    if not 0 <= tolerance <= 255:
        raise ValueError("tolerance must be between 0 and 255")
    if masks is not None and len(masks) != len(paths):
        raise ValueError("masks length must match images length")
    if angles is None:
        angles = [i * 360.0 / len(paths) for i in range(len(paths))]
    elif len(angles) != len(paths):
        raise ValueError("angles length must match images length")
    if not np.isfinite(angles).all():
        raise ValueError("angles must be finite numbers")

    bg = np.array(_parse_hex(background_hex), dtype=np.uint8) if background_hex else None
    views: list[View] = []
    for i, (p, theta) in enumerate(zip(paths, angles)):
        with Image.open(p) as source:
            pil = ImageOps.exif_transpose(source).convert("RGBA")
            rgba = np.array(pil)
        img = rgba[..., :3]
        if masks is not None:
            with Image.open(masks[i]) as source:
                mask = np.array(ImageOps.exif_transpose(source).convert("L")) > 127
            if mask.shape != img.shape[:2]:
                raise ValueError(f"{p}: mask dimensions must match the image")
        elif (rgba[..., 3] < 255).any():
            mask = rgba[..., 3] > 127
        else:
            mask = _foreground_mask(img, bg, tolerance=tolerance)
        if mask.sum() < 50:
            raise ValueError(
                f"{p}: foreground mask has {mask.sum()} pixels — is the background wrong?"
            )
        views.append(View(image_rgb=img, mask=mask, theta_deg=float(theta)))
    return views


def _view_bounds(view: View) -> tuple[int, int, int, int]:
    """Bounding box (u0, v0, u1, v1) of the foreground in the image."""
    ys, xs = np.where(view.mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _carve_grid(views: list[View], max_studs: int, min_views_fraction: float
                ) -> tuple[np.ndarray, np.ndarray]:
    """Intersect silhouettes, then transfer color from visible surfaces only."""
    if not 1 <= max_studs <= 128:
        raise ValueError("max_studs must be between 1 and 128")
    if not 0 < min_views_fraction <= 1:
        raise ValueError("min_views_fraction must be greater than 0 and at most 1")
    # Establish a common object bounding box across views. Widest horizontal
    # extent -> max_studs; tallest -> plate count.
    max_w_px = max(_view_bounds(v)[2] - _view_bounds(v)[0] for v in views)
    max_h_px = max(_view_bounds(v)[3] - _view_bounds(v)[1] for v in views)

    # World-mm per pixel: we want max_w_px pixels == max_studs * MM_PER_STUD mm.
    mm_per_px = (max_studs * bricks.MM_PER_STUD) / max_w_px
    world_h_mm = max_h_px * mm_per_px

    W = max_studs
    D = max_studs
    H = max(1, int(round(world_h_mm / bricks.MM_PER_PLATE)))
    if H * W * D > 8_000_000:
        raise ValueError("Object is too tall for this resolution; reduce max_studs")

    # Voxel centers in world mm, centered at origin horizontally, y=0 at bottom.
    xs = (np.arange(W) + 0.5 - W / 2) * bricks.MM_PER_STUD
    zs = (np.arange(D) + 0.5 - D / 2) * bricks.MM_PER_STUD
    ys = (np.arange(H) + 0.5) * bricks.MM_PER_PLATE

    # Vote grid: number of views that see each voxel as foreground.
    votes = np.zeros((H, W, D), dtype=np.uint32)
    color_sum = np.zeros((H, W, D, 3), dtype=np.uint64)
    color_hits = np.zeros((H, W, D), dtype=np.uint32)

    projections = []
    for view in views:
        img = view.image_rgb
        mask = view.mask
        img_h, img_w = mask.shape
        u0, v0, u1, v1 = _view_bounds(view)
        cu = (u0 + u1) / 2.0  # horizontal center of object in image
        vbottom = v1

        theta = math.radians(view.theta_deg)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        # Broadcast: rotate voxel world x/z into camera-plane u.
        # For each voxel (yi, xi, zi):
        #   world_x = xs[xi], world_z = zs[zi]
        #   rotated_x =  world_x * cos_t + world_z * sin_t
        #   image_u = cu + rotated_x / mm_per_px
        #   image_v = vbottom - world_y / mm_per_px
        # Vectorized meshgrid: (W, D)
        rx = xs[:, None] * cos_t + zs[None, :] * sin_t  # (W, D)
        image_u_grid = cu + rx / mm_per_px              # (W, D)
        # (H,) row: image_v per plate
        image_v_col = vbottom - ys / mm_per_px          # (H,)

        # Convert to integer image pixels.
        u_int = np.floor(image_u_grid).astype(np.int32)
        v_int = np.floor(image_v_col).astype(np.int32)
        depth = -xs[:, None] * sin_t + zs[None, :] * cos_t
        bins = np.floor(rx / bricks.MM_PER_STUD + W + D).astype(int)
        projections.append((view, u_int, v_int, depth, bins))

        # For each y-slice, look up mask[v_int[y], u_int]. Vectorize over W, D.
        # Clip u_int, v_int to image bounds; out-of-image treated as background.
        u_ok = (u_int >= 0) & (u_int < img_w)
        # We'll process per y for memory reasons; grid can be big.
        for yi in range(H):
            vy = v_int[yi]
            if vy < 0 or vy >= img_h:
                continue
            mask_row = mask[vy]  # (img_w,)
            u_clipped = np.clip(u_int, 0, img_w - 1)
            fg = mask_row[u_clipped] & u_ok  # (W, D)
            votes[yi] += fg

    threshold = max(1, int(math.ceil(min_views_fraction * len(views))))
    occupied = votes >= threshold

    # A coarse z-buffer uses one stud-wide ray per bin. Hidden back surfaces
    # must not contribute their color to the front of the model.
    for view, us, vs, depth, bins in projections:
        height, width = view.mask.shape
        valid_u = (us >= 0) & (us < width)
        clipped = np.clip(us, 0, width - 1)
        for yi, row in enumerate(vs):
            if not 0 <= row < height:
                continue
            candidates = occupied[yi] & valid_u & view.mask[row, clipped]
            if not candidates.any():
                continue
            front = np.full(3 * (W + D), -np.inf)
            np.maximum.at(front, bins[candidates], depth[candidates])
            visible = candidates & (depth >= front[bins] - bricks.MM_PER_STUD * 0.75)
            color_sum[yi][visible] += view.image_rgb[row, clipped][visible]
            color_hits[yi][visible] += 1

    # Compute mean color where visible.
    mean_rgb = np.zeros((H, W, D, 3), dtype=np.uint8)
    with np.errstate(invalid="ignore"):
        seen = color_hits > 0
        for c in range(3):
            channel = np.where(seen, color_sum[..., c] // np.maximum(color_hits, 1), 0)
            mean_rgb[..., c] = channel.astype(np.uint8)

    if seen.any():
        nearest = ndimage.distance_transform_edt(~seen, return_distances=False,
                                                return_indices=True)
        mean_rgb[~seen] = mean_rgb[tuple(nearest[:, ~seen])]

    return occupied, mean_rgb


def carve(paths: Sequence[str | Path], *,
          max_studs: int = 32,
          angles: Sequence[float] | None = None,
          background_hex: str | None = None,
          tolerance: int = 28,
          min_views_fraction: float = 1.0,
          masks: Sequence[str | Path] | None = None) -> VoxelGrid:
    """Space-carve `paths` into a colored VoxelGrid at LEGO scale.

    `min_views_fraction`: 1.0 = strict (intersection of all views). 0.9 tolerates
    one view voting wrong (useful with 10+ views).
    """
    views = load_views(paths, angles=angles, background_hex=background_hex,
                        tolerance=tolerance, masks=masks)
    occupied, rgb = _carve_grid(views, max_studs=max_studs,
                                 min_views_fraction=min_views_fraction)
    if not occupied.any():
        raise RuntimeError(
            "Space carving produced no occupied voxels. Try a different "
            "--background hex or lower --min-views-fraction."
        )

    H, W, D = occupied.shape
    palette_idx = np.full((H, W, D), -1, dtype=np.int16)
    occ_pts = np.argwhere(occupied)
    voxel_colors = rgb[occ_pts[:, 0], occ_pts[:, 1], occ_pts[:, 2]]
    idx = colors.snap_rgb_to_palette(voxel_colors)
    palette_idx[occ_pts[:, 0], occ_pts[:, 1], occ_pts[:, 2]] = idx
    return VoxelGrid(palette_idx=palette_idx, metadata={
        "method": "visual_hull", "max_studs": max_studs,
        "images": [str(p) for p in paths], "angles": [v.theta_deg for v in views],
        "foreground_pixels": [int(v.mask.sum()) for v in views],
        "tolerance": tolerance, "min_views_fraction": min_views_fraction,
        "assumptions": "Level turntable views, fixed scale, centered object, orthographic projection.",
        "limitation": "Silhouettes cannot recover hidden concavities or arbitrary camera poses.",
    })
