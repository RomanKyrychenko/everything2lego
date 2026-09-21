"""SoftRenderer — differentiable rasterisation of a SoftAssembly to an image.

Given a `SoftAssembly` (per-cell soft categorical over types + colours),
render it to a 2-D image via **volumetric orthographic projection** along
one axis:

    For each pixel (i, j), integrate colour along the projection axis
    weighted by the "probability of being the first opaque cell":

        α_c = P(cell c is occupied) = 1 − P(EMPTY)
        T_c = ∏_{c' before c} (1 − α_{c'})            [transmittance]
        img[i, j] = Σ_c   T_c · α_c · palette_rgb @ soft_color_c

This is a classical NeRF-style volumetric-rendering equation and is fully
differentiable — gradients of a pixel-space L2 loss flow into every cell's
type-logits and colour-logits.

Rendering three orthogonal views (front / side / top) gives the trainer
three complementary losses, enough to shape the assembly against a set of
photos without needing an explicit 3D reconstruction step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

from ..colors import palette
from .soft_assembly import SoftAssembly, EMPTY


class SoftRenderer:
    """Orthographic volumetric renderer.

    Axis convention: `axis='z'` projects along +Z (front view) → image
    shape (Y, X, 3). `axis='y'` (top view) → (Z, X, 3). `axis='x'`
    (side view) → (Y, Z, 3).
    """

    def __init__(self, n_colors: int, device: str = "cpu",
                  background: tuple[int, int, int] = (240, 240, 240)) -> None:
        assert _HAS_TORCH, "PyTorch required for SoftRenderer"
        pal = palette()
        # palette_rgb: (n_colors, 3) float32 normalised to [0,1]
        rgb = np.array([c.rgb for c in pal], dtype=np.float32) / 255.0
        # Trim / pad to n_colors so indices align.
        if len(rgb) < n_colors:
            rgb = np.concatenate([rgb, np.zeros((n_colors - len(rgb), 3),
                                                    dtype=np.float32)])
        self.palette_rgb = torch.from_numpy(rgb[:n_colors]).to(device)
        self.background = (torch.tensor(background, dtype=torch.float32,
                                             device=device) / 255.0)

    def render(self, a: SoftAssembly, tau: float, *,
                axis: str = "z") -> "torch.Tensor":
        """Return an RGB image of shape (H, W, 3) in [0, 1].

        Gradient flows back through `tau` since `soft_type_probs(tau)` and
        `soft_color_probs(tau)` are differentiable in the assembly params.
        """
        soft_types = a.soft_type_probs(tau)      # (Y, X, Z, N_TYPES)
        soft_colors = a.soft_color_probs(tau)    # (Y, X, Z, N_COLORS)
        alpha = 1.0 - soft_types[..., EMPTY]     # (Y, X, Z)
        # Color per cell in RGB: soft_colors @ palette (broadcasts to (Y,X,Z,3)).
        cell_rgb = soft_colors @ self.palette_rgb

        # Orient tensors so the projection axis is LAST for a clean cumprod.
        if axis == "z":
            a_ = alpha
            c_ = cell_rgb
        elif axis == "y":
            a_ = alpha.permute(2, 1, 0)          # (Z, X, Y)
            c_ = cell_rgb.permute(2, 1, 0, 3)
        elif axis == "x":
            a_ = alpha.permute(0, 2, 1)          # (Y, Z, X)
            c_ = cell_rgb.permute(0, 2, 1, 3)
        else:
            raise ValueError(f"axis must be x/y/z, got {axis!r}")

        # Volumetric transmittance along the last axis.
        # For rendering FROM +axis toward -axis (viewer at +axis side),
        # we walk cells in REVERSE order.  Flip so cell 0 is nearest camera.
        a_rev = torch.flip(a_, dims=(-1,))
        c_rev = torch.flip(c_, dims=(-2,))

        # T_c = product over c' < c of (1 - α_{c'}).  cumprod is inclusive,
        # so shift by one: T_c = cumprod up to c-1.
        one_minus = 1.0 - a_rev
        # prepend a 1 for c=0.
        pad = torch.ones_like(one_minus[..., :1])
        one_minus_padded = torch.cat([pad, one_minus[..., :-1]], dim=-1)
        T = torch.cumprod(one_minus_padded, dim=-1)

        # weight per cell = T · α
        w = T * a_rev  # (..., D)
        # image = Σ_c  w_c · cell_rgb_c
        # w: (..., D)  →  (..., D, 1);  c_rev: (..., D, 3)
        img = (w.unsqueeze(-1) * c_rev).sum(dim=-2)
        # residual transmittance goes to the background colour.
        residual = T[..., -1] * (1.0 - a_rev[..., -1])
        img = img + residual.unsqueeze(-1) * self.background
        return img

    def render_multiple(self, a: SoftAssembly, tau: float,
                          axes: Iterable[str] = ("z", "x", "y")
                          ) -> dict[str, "torch.Tensor"]:
        return {ax: self.render(a, tau, axis=ax) for ax in axes}


def photo_loss(rendered: "torch.Tensor", target: "torch.Tensor") -> "torch.Tensor":
    """Simple L2 image-space loss.  Both tensors (H, W, 3) in [0, 1].

    If shapes mismatch, target is bilinearly resampled to `rendered`.  A
    more advanced version would use perceptual (VGG feature) loss."""
    import torch.nn.functional as F
    if rendered.shape != target.shape:
        # torch.nn.functional.interpolate expects (N, C, H, W).
        tgt = target.permute(2, 0, 1).unsqueeze(0)
        tgt = F.interpolate(tgt, size=rendered.shape[:2],
                             mode="bilinear", align_corners=False)
        target = tgt.squeeze(0).permute(1, 2, 0)
    return ((rendered - target) ** 2).mean()
