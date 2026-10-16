"""Load the mined OMR pattern catalogue (tools/learn/mine_patterns.py)
as pair entries compatible with hinges.load_pairs — the sampler's
hinge_swap move already does "place B at (dy, dx, dz) relative to A",
so wiring the catalogue through there turns the catalogue into
realised placements without any new move type.

Catalogue entries from mine_patterns:
    {"a": {"ldraw", "yaw"}, "b": {"ldraw", "yaw"},
     "offset": {"dy", "dx", "dz"}, "models", "count"}

Normalised to the same shape as hinges.load_pairs: adds a_yaw / b_yaw
(hinges always had a_yaw=0 implicitly) and the three offsets so the
unified consumer can treat both the same.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

CATALOGUE_PATH = Path(os.environ.get(
    "CATALOGUE_JSON",
    "/Volumes/PRO-G40/lego_data/analysis/patterns_size2.json"))

DEFAULT_LIMIT = int(os.environ.get("CATALOGUE_LIMIT", "500"))


@lru_cache(maxsize=4)
def load_pairs(limit: int = DEFAULT_LIMIT) -> tuple[dict, ...]:
    """Top `limit` catalogue patterns (by model count), ldraw-pair format.
    Empty tuple if the file isn't present (so builds work without the
    catalogue in paths that don't need it)."""
    if not CATALOGUE_PATH.exists():
        return ()
    rows = json.loads(CATALOGUE_PATH.read_text())
    rows = sorted(rows, key=lambda r: -r["models"])[:limit]
    out = []
    for r in rows:
        out.append({
            "a_ldraw": r["a"]["ldraw"], "a_yaw": int(r["a"]["yaw"]) % 360,
            "b_ldraw": r["b"]["ldraw"], "yaw_s": int(r["b"]["yaw"]) % 360,
            "dy_s": int(r["offset"]["dy"]), "dx_s": int(r["offset"]["dx"]),
            "dz_s": int(r["offset"]["dz"]),
            "stacked": False,                                         # explicit offset, not the hinge stack convention
            "models": int(r["models"]),
        })
    return tuple(out)
