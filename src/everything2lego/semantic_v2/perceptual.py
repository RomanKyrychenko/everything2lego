"""PerceptualLoss — VGG-feature-based image loss for semantic photo matching.

Pixel-space L2 (`photo_loss`) is sensitive to per-pixel alignment: a
slightly-shifted match reads as high loss. Perceptual loss compares
feature maps from a pretrained VGG16, so two images with the same
"content" (edges, colours, textures) score low even under small
translations. This is what makes "looks like the photo" tractable at
LEGO's coarse resolution.

Uses torchvision's VGG16 with ImageNet weights. First call downloads
weights; subsequent calls are cached.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


_VGG_MEAN = (0.485, 0.456, 0.406)
_VGG_STD  = (0.229, 0.224, 0.225)


class PerceptualLoss:
    """Compare two images via distances in VGG16 feature space.

    Layers we tap (default): relu1_2, relu2_2, relu3_3, relu4_3 — the
    classical Johnson et al. perceptual set.
    """

    def __init__(self, layers: tuple[str, ...] = ("relu1_2", "relu2_2",
                                                        "relu3_3", "relu4_3"),
                  device: str = "cpu") -> None:
        assert _HAS_TORCH, "PyTorch required for PerceptualLoss"
        from torchvision.models import vgg16, VGG16_Weights
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg.to(device)
        self.device = device
        # torchvision VGG16 feature-index of each named layer.
        self._name_to_idx = {
            "relu1_2": 3, "relu2_2": 8, "relu3_3": 15, "relu4_3": 22,
        }
        self.tap_indices = sorted(self._name_to_idx[n] for n in layers)
        self.mean = torch.tensor(_VGG_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(_VGG_STD, device=device).view(1, 3, 1, 1)

    def _prep(self, img: "torch.Tensor") -> "torch.Tensor":
        """Convert (H, W, 3) [0,1] to (1, 3, H', W') VGG-normalised."""
        x = img.permute(2, 0, 1).unsqueeze(0)
        # VGG expects at least ~32 px. Upsample tiny renderings.
        if min(x.shape[-2:]) < 64:
            x = F.interpolate(x, size=(64, 64), mode="bilinear",
                                align_corners=False)
        x = (x - self.mean) / self.std
        return x

    def compute(self, rendered: "torch.Tensor",
                 target: "torch.Tensor") -> "torch.Tensor":
        r = self._prep(rendered)
        t = self._prep(target).detach()  # target has no grad
        # VGG needs matching spatial dims across the pair — resize target
        # to the rendered size before feature extraction.  Rendered is
        # typically the coarser LEGO-grid resolution; the target photo is
        # arbitrary, so we downsample the photo to the render size.
        if t.shape[-2:] != r.shape[-2:]:
            t = F.interpolate(t, size=r.shape[-2:], mode="bilinear",
                                align_corners=False)
        loss = torch.zeros(1, device=self.device)
        # Forward through VGG, extracting activations at tap points.
        for name, module in enumerate(self.vgg):
            r = module(r); t = module(t)
            if name in self.tap_indices:
                loss = loss + F.mse_loss(r, t)
        return loss.squeeze()
