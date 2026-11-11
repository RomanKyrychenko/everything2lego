"""Hinge pairs: real-set-observed pairs of parts with finger
connectors (hinge brick base + top, hinge plate base + top).  Data
produced by tools/partsdb/hinges.py — step A of item 4.

This module turns the JSON into a move for the sampler: a
Metropolis step that attaches B at its canonical pose when an A
instance exists in the state.

The pair data stores LDraw-relative poses `(yaw_ldraw, dy_ldraw_plate,
dx_ldraw_stud, dz_ldraw_stud)`.  In LDraw conventions, two hinge parts
often share their LDraw origin at the pivot; in the sampler's grid
that's cell overlap, which we don't allow.  The convention here:

- If the LDraw pose is `(yaw=0, dy=dx=dz=0)` (coincident origin, hinge
  closed), stack B directly on top of A in sampler space:
  `y_B = y_A + h_A, x_B = x_A, z_B = z_A, yaw_B = 0`.
- Otherwise (an open-hinge or offset pose), convert the LDraw
  offsets to sampler offsets:
  `y_B = y_A + dy, x_B = x_A + dx, z_B = z_A + dz, yaw_B = yaw_ldraw`.

A per-pair cache stores (ldraw_a, ldraw_b, dy_s, dx_s, dz_s, yaw_s)
tuples; the sampler move looks these up by `a.vp.part.ldraw`.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

HINGE_PAIRS_PATH = Path(os.environ.get(
    "HINGE_PAIRS_JSON",
    "/Volumes/PRO-G40/lego_data/analysis/hinge_pairs.json"))


@lru_cache(maxsize=1)
def load_pairs() -> tuple[dict, ...]:
    """Hinge pairs from the dedicated JSON.  Each pair:
    {a_ldraw, a_yaw, b_ldraw, yaw_s, dy_s, dx_s, dz_s, stacked, models}.
    Hinge pairs always have a_yaw=0 (hinge geometry is captured relative
    to the base part's identity transform)."""
    if not HINGE_PAIRS_PATH.exists():
        return ()
    rows = json.loads(HINGE_PAIRS_PATH.read_text())
    out = []
    for r in rows:
        a_ldraw = f"{r['a']}.dat"
        b_ldraw = f"{r['b']}.dat"
        yaw_ldraw = int(r.get("yaw", 0)) % 360
        dy_ldraw, dx_ldraw, dz_ldraw = int(r["dy"]), int(r["dx"]), int(r["dz"])
        stacked = (yaw_ldraw == 0 and dy_ldraw == 0 and dx_ldraw == 0 and dz_ldraw == 0)
        out.append({
            "a_ldraw": a_ldraw, "a_yaw": 0,
            "b_ldraw": b_ldraw, "yaw_s": yaw_ldraw,
            "dy_s": dy_ldraw, "dx_s": dx_ldraw, "dz_s": dz_ldraw,
            "stacked": stacked,
            "models": int(r.get("models", 0)),
        })
    return tuple(out)


def load_all_pairs() -> tuple[dict, ...]:
    """Hinge pairs + top catalogue pairs (if the catalogue is present).
    See catalogue.load_pairs for the limit (default 500 by model count)."""
    from . import catalogue
    return load_pairs() + catalogue.load_pairs()


def by_a_yaw() -> dict[tuple[str, int], list[dict]]:
    """(ldraw, yaw) of A -> list of pair entries.  Combined index over
    the hinge catalogue and the mined pattern catalogue."""
    out: dict[tuple[str, int], list[dict]] = {}
    for p in load_all_pairs():
        out.setdefault((p["a_ldraw"], p["a_yaw"]), []).append(p)
    return out


# backwards-compatible alias (hinge_swap used this once); callers should
# prefer by_a_yaw now
def by_a_ldraw() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for p in load_all_pairs():
        out.setdefault(p["a_ldraw"], []).append(p)
    return out
