"""Semantic-decomposition LEGO pipeline.

Alternative to voxel-and-pack: segment the input into semantic regions
(foundation, wall, roof, opening, detail), give each to a specialized
builder, then reconcile.

Currently scaffolding — Foundation segmenter and builder are working; the
rest are stubbed with TODO markers. See ARCHITECTURE.md for the full design.
"""

from __future__ import annotations

from .regions import Region, RegionLabel
from .segment import segment
from . import builders

__all__ = ["Region", "RegionLabel", "segment", "builders"]
