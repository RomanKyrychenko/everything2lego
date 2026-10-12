"""LDraw color palette + nearest-color lookup."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class LegoColor:
    code: int         # LDraw color code
    bl: int           # Bricklink color id
    name: str
    rgb: tuple[int, int, int]


@lru_cache(maxsize=1)
def palette() -> tuple[LegoColor, ...]:
    raw = json.loads(files("everything2lego.data").joinpath("ldraw_colors.json").read_text())
    return tuple(
        LegoColor(code=c["code"], bl=c["bl"], name=c["name"], rgb=tuple(c["rgb"]))
        for c in raw["colors"]
    )


@lru_cache(maxsize=1)
def _palette_lab() -> np.ndarray:
    """Palette as CIE Lab for perceptual nearest-neighbor lookup."""
    return _rgb_to_lab(np.array([c.rgb for c in palette()], dtype=np.float32) / 255.0)


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB [0,1] -> CIE Lab. Vectorized. D65 white point."""
    # sRGB companding
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    # sRGB -> XYZ (D65)
    m = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ], dtype=np.float32)
    xyz = linear @ m.T
    # Reference white D65
    xyz /= np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
    delta = 6 / 29
    f = np.where(xyz > delta ** 3, np.cbrt(xyz), xyz / (3 * delta ** 2) + 4 / 29)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def snap_rgb_to_palette(rgb: np.ndarray) -> np.ndarray:
    """Map an (N,3) or (H,W,3) uint8 array to an index array into `palette()`."""
    orig_shape = rgb.shape[:-1]
    flat = rgb.reshape(-1, 3).astype(np.float32) / 255.0
    lab = _rgb_to_lab(flat)
    pal = _palette_lab()
    # (N, 1, 3) - (1, P, 3) -> (N, P) squared distance
    d = ((lab[:, None, :] - pal[None, :, :]) ** 2).sum(axis=-1)
    idx = np.argmin(d, axis=1)
    return idx.reshape(orig_shape)


def by_code(code: int) -> LegoColor:
    for c in palette():
        if c.code == code:
            return c
    raise KeyError(f"Unknown LDraw color code {code}")


def codes(indices: Iterable[int]) -> list[int]:
    p = palette()
    return [p[i].code for i in indices]


def by_code_index(code: int) -> int:
    """Index into `palette()` of the LDraw color `code`."""
    for i, c in enumerate(palette()):
        if c.code == code:
            return i
    raise KeyError(f"Unknown LDraw color code {code}")


# The "house" palette: a small set of plausible building colours.  Each
# entry is (sRGB the input is matched against, LDraw code it snaps to).
# Walls: white, tan, light/dark bluish grey, reddish brown, red.
# Roofs: dark bluish grey, dark red, reddish brown, black.  Trim: white,
# black.  Glass: trans-light-blue (LDraw 43); when that code is not in the
# shipped palette it falls back to opaque light bluish grey (71).
_HOUSE_ENTRIES: tuple[tuple[tuple[int, int, int], int], ...] = (
    ((255, 255, 255), 15),   # White
    ((228, 205, 158), 19),   # Tan
    ((160, 165, 169), 71),   # Light Bluish Gray
    ((108, 110, 104), 72),   # Dark Bluish Gray
    ((88, 42, 18), 70),      # Reddish Brown
    ((201, 26, 9), 4),       # Red
    ((114, 14, 15), 320),    # Dark Red
    ((27, 42, 52), 0),       # Black
    ((193, 223, 240), 43),   # Trans Light Blue (glass)
)
HOUSE_GLASS_FALLBACK_CODE = 71

PALETTE_MODES = ("house", "auto", "full")


@lru_cache(maxsize=1)
def house_palette() -> tuple[tuple[np.ndarray, int], ...]:
    """(Lab row, palette index) per house entry, with the glass fallback
    applied when trans-light-blue is not in the LDraw palette."""
    known = {c.code for c in palette()}
    out = []
    for rgb, code in _HOUSE_ENTRIES:
        if code not in known:
            code = HOUSE_GLASS_FALLBACK_CODE
        lab = _rgb_to_lab(np.array(rgb, dtype=np.float32)[None] / 255.0)[0]
        out.append((lab, by_code_index(code)))
    return tuple(out)


def snap_rgb_to_house(rgb: np.ndarray) -> np.ndarray:
    """Map (N,3) uint8 sRGB to palette indices of the nearest house colour
    (CIE Lab distance)."""
    orig_shape = rgb.shape[:-1]
    flat = rgb.reshape(-1, 3).astype(np.float32) / 255.0
    lab = _rgb_to_lab(flat)
    entries = house_palette()
    centres = np.stack([e[0] for e in entries])
    targets = np.array([e[1] for e in entries])
    d = ((lab[:, None, :] - centres[None, :, :]) ** 2).sum(axis=-1)
    return targets[np.argmin(d, axis=1)].reshape(orig_shape)


def remap_to_house(palette_idx: np.ndarray) -> np.ndarray:
    """Snap every occupied cell of a palette-index grid to the house
    palette.  Empty cells (-1) are left alone."""
    used = np.unique(palette_idx[palette_idx >= 0])
    if len(used) == 0:
        return palette_idx.copy()
    pal = palette()
    rgb = np.array([pal[i].rgb for i in used], dtype=np.uint8)
    lut = dict(zip(used.tolist(), snap_rgb_to_house(rgb).tolist()))
    out = palette_idx.copy()
    for src, dst in lut.items():
        out[palette_idx == src] = dst
    return out


def reduce_palette(palette_idx: np.ndarray, k: int) -> np.ndarray:
    """Cluster the colours used by a palette-index grid down to at most `k`
    LDraw colours.  Deterministic k-means in Lab space, weighted by how
    many cells each colour covers, seeded by farthest-point sampling from
    the most common colour.  Each cluster takes the LDraw colour of its
    most-used member, so no new colours are introduced.  Empty cells (-1)
    are left alone."""
    if k < 1:
        raise ValueError("k must be positive")
    used, counts = np.unique(palette_idx[palette_idx >= 0], return_counts=True)
    if len(used) <= k:
        return palette_idx.copy()
    lab = _palette_lab()[used]
    w = counts.astype(np.float64)
    # Farthest-point seeding from the dominant colour.
    centres = [lab[int(np.argmax(w))]]
    for _ in range(k - 1):
        d = np.min([((lab - c) ** 2).sum(axis=1) for c in centres], axis=0)
        centres.append(lab[int(np.argmax(d * w))])
    centres = np.stack(centres)
    labels = np.zeros(len(used), dtype=int)
    for _ in range(50):
        d = ((lab[:, None, :] - centres[None, :, :]) ** 2).sum(axis=-1)
        new_labels = np.argmin(d, axis=1)
        if np.array_equal(new_labels, labels) and _ > 0:
            break
        labels = new_labels
        for j in range(k):
            members = labels == j
            if members.any():
                centres[j] = (lab[members] * w[members, None]).sum(axis=0) / w[members].sum()
    out = palette_idx.copy()
    for j in range(k):
        members = np.flatnonzero(labels == j)
        if len(members) == 0:
            continue
        target = used[members[int(np.argmax(w[members]))]]
        for src in used[members]:
            if src != target:
                out[palette_idx == src] = target
    return out
