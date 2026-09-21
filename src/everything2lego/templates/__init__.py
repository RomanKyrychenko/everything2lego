"""Parametric LEGO templates — hand-designed architectural types.

Each template exposes a `.params` dataclass and a `.build(params)` method
that returns a list of `pack.Placement`. Templates are registered by name in
`TEMPLATES` so the CLI can dispatch by string name.
"""

from __future__ import annotations

from .base import Template, register
from .gable_cabin import GableCabin
from .modernist_box import ModernistBox

# Populate the registry.
TEMPLATES: dict[str, type[Template]] = {}
register(TEMPLATES, GableCabin)
register(TEMPLATES, ModernistBox)

__all__ = ["Template", "TEMPLATES", "GableCabin", "ModernistBox"]
