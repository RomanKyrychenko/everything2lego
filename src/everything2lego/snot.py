"""SNOT templates: real-set-observed pairs where B sits beside A with a
non-yaw rotation (studs-not-up).  Data produced by tools/partsdb/snot_adj.py.

This module turns the JSON into a lookup for the sampler: given a placed
anchor A at (ldraw, yaw), the templates that attach B in a known
sideways rotation at a known relative offset.

Each pair record (after enrichment in snot_adj.py v2) carries:

    a_ldraw, b_ldraw     parts (A.dat, B.dat)
    face_a, face_b       one of {"U", "S", "D"} — anchor face / decor face
    rot                  B's rotation in A's local frame, as "<rx><ry>"
                         e.g. "x+y+" (identity, B is studs-up relative to A),
                         "x+z-" (B rotated 90° about +X: studs face A's +z),
                         ... one of 24 cube symmetries
    dy_s, dx_s, dz_s     B's lo-corner minus A's lo-corner in A's frame,
                         dy in plates, dx/dz in studs
    models               distinct OMR models carrying this template
    dyaw_world           world-frame yaw diff (legacy; prefer `rot`)

Backward compat: records produced by snot_adj.py v1 (no `rot` field) are
skipped by this loader — the sampler needs the full rotation to render.

Sampler integration (deferred): a `snot_swap` move picks an A in the
state matching `(a_ldraw, a_yaw)`, picks a template from that list, and
tries to place B at `(y + dy_s, x + dx'_s, z + dz'_s)` where `(dx'_s,
dz'_s)` is `(dx_s, dz_s)` rotated by A's placement yaw into the world
frame.  B is rendered with the rotation implied by `rot` composed with
A's yaw.  Requires: Item.snot_rot + Placement.snot_rot + io_ldraw
rendering + State.add/remove cell accounting for the sideways footprint
(see tasks #254, #255).
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

SNOT_PAIRS_PATH = Path(os.environ.get(
    "SNOT_PAIRS_JSON",
    "/Volumes/PRO-G40/lego_data/analysis/snot_adj.json"))


@lru_cache(maxsize=1)
def load_pairs() -> tuple[dict, ...]:
    """SNOT pair records with the enriched rotation.  Records missing the
    `rot` field (v1 schema) are dropped; re-run snot_adj.py v2 to produce
    them."""
    if not SNOT_PAIRS_PATH.exists():
        return ()
    rows = json.loads(SNOT_PAIRS_PATH.read_text())
    out = []
    for r in rows:
        rot = r.get("rot")
        if rot is None:
            continue                                                   # legacy schema; skip
        a_ldraw = f"{r['a']}.dat"
        b_ldraw = f"{r['b']}.dat"
        out.append({
            "a_ldraw": a_ldraw,
            "b_ldraw": b_ldraw,
            "face_a": r["face_a"],
            "face_b": r["face_b"],
            "rot": rot,
            "dy_s": int(r["dy"]), "dx_s": int(r["dx"]), "dz_s": int(r["dz"]),
            "models": int(r.get("models", 0)),
        })
    return tuple(out)


def by_a_ldraw() -> dict[str, list[dict]]:
    """`A.dat` → list of templates anchored at that part.  Keyed only by
    ldraw (not yaw): A's yaw is composed in at placement time, since
    offsets and `rot` are already in A's local frame."""
    out: dict[str, list[dict]] = {}
    for p in load_pairs():
        out.setdefault(p["a_ldraw"], []).append(p)
    return out
