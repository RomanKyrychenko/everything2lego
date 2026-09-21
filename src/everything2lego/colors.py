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
