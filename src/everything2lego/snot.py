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
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

# LDU: 20 per stud (horizontal), 8 per plate (vertical).  Matches io_ldraw's
# LDU_PER_STUD / LDU_PER_PLATE but we keep a local copy so this module has no
# runtime import of io_ldraw (circular import once io_ldraw calls into here).
_LDU_STUD = 20
_LDU_PLATE = 8

# Axis-code basis vectors in LDraw world (y down, right-handed).  Reading
# "x+" / "x-" / "y+" / "y-" / "z+" / "z-" into a 3-tuple.
_AXIS = {"x+": (1, 0, 0), "x-": (-1, 0, 0),
         "y+": (0, 1, 0), "y-": (0, -1, 0),
         "z+": (0, 0, 1), "z-": (0, 0, -1)}


@dataclass(frozen=True)
class SnotChild:
    """A sideways-placed part parented to a studs-up Item.  Lives in LDU
    offsets from the parent's grid position + a 24-way rotation in the
    parent's local frame.  Not an Item: no sampler cells, no energy
    contribution, no adjacency — rendering-only until the dual-grid
    collision + energy work lands (see tasks #254 slice 2 + #255)."""
    b_ldraw: str
    b_colour: int
    rot: str                 # "<rx><ry>" cube-symmetry code; see parse_rot
    dy_LDU: int              # B's lo-corner offset from parent in LDU (y down)
    dx_LDU: int
    dz_LDU: int
    models: int = 0          # how many OMR models carried this template

SNOT_PAIRS_PATH = Path(os.environ.get(
    "SNOT_PAIRS_JSON",
    "/Volumes/PRO-G40/lego_data/analysis/snot_adj.json"))


@lru_cache(maxsize=1)
def load_pairs() -> tuple[dict, ...]:
    """SNOT pair records with the enriched rotation.  Records missing the
    `rot` field (v1 schema) are dropped; re-run snot_adj.py v2 to produce
    them.

    Only face_a == 'U' records are returned: the sampler places parts
    studs-up, so A's frame in these records matches the sampler's
    placement frame and the stored (rot, dx, dy, dz) can be applied
    directly.  S+U records are dropped — handling those would need the
    loader to swap A and B and invert the pose, future work."""
    if not SNOT_PAIRS_PATH.exists():
        return ()
    rows = json.loads(SNOT_PAIRS_PATH.read_text())
    out = []
    for r in rows:
        rot = r.get("rot")
        if rot is None:
            continue                                                   # legacy schema; skip
        if r.get("face_a") != "U":
            continue                                                   # A sideways: pose not in sampler-placeable frame
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


def parse_rot(rot: str) -> tuple[int, ...]:
    """2-letter axis code (e.g. "x+z-") → 9-int row-major 3x3 matrix giving
    B's basis vectors in A's local frame.  Columns are B's +X, +Y, +Z
    axes expressed in A's frame.  The third column is derived from the
    first two by right-hand rule in LDraw's y-down world (so cross
    product is +X × +Y = +Z).

    LDraw matrices are column-stored row-major as (a b c d e f g h i)
    where (a,d,g) is where B's +X goes, (b,e,h) is where B's +Y goes,
    (c,f,i) is where B's +Z goes."""
    if len(rot) != 4:
        raise ValueError(f"bad rot code {rot!r}")
    rx = _AXIS.get(rot[:2]); ry = _AXIS.get(rot[2:])
    if rx is None or ry is None:
        raise ValueError(f"bad rot code {rot!r}")
    # +Z = +X × +Y (y-down, right-handed: same handedness as a +y-up world
    # with mirrored y; cross product formula is unchanged).
    rz = (rx[1]*ry[2] - rx[2]*ry[1],
          rx[2]*ry[0] - rx[0]*ry[2],
          rx[0]*ry[1] - rx[1]*ry[0])
    return (rx[0], ry[0], rz[0],
            rx[1], ry[1], rz[1],
            rx[2], ry[2], rz[2])


def _mat_mul(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:
    """3x3 * 3x3, row-major."""
    out = [0] * 9
    for r in range(3):
        for c in range(3):
            out[r*3 + c] = sum(a[r*3 + k] * b[k*3 + c] for k in range(3))
    return tuple(out)


def _mat_vec(m: tuple[int, ...], v: tuple[int, int, int]) -> tuple[int, int, int]:
    """3x3 * 3-vector, row-major."""
    return (m[0]*v[0] + m[1]*v[1] + m[2]*v[2],
            m[3]*v[0] + m[4]*v[1] + m[5]*v[2],
            m[6]*v[0] + m[7]*v[1] + m[8]*v[2])


def decorate(placements, seed: int = 0, rate: float = 0.2) -> dict:
    """Post-process a list of Placements into a decorations dict (keyed
    by Placement, value is a list of SnotChild).  Each matching anchor
    gets a child with probability `rate`, drawn by model-count weight.
    `rate=0.2` targets real-set SNOT fraction (17-32% per OMR audit).
    Doesn't need sampler state — works after any engine.

    Returns {} when snot_adj.json is absent / lacks the v2 `rot` field.
    Children inherit the anchor's colour (b_colour = palette()[anchor.color_idx].code)
    unless and until a vocab-driven colour match is added."""
    import random
    from .colors import palette
    pal = palette()
    rnd = random.Random(seed)
    idx = by_a_ldraw()
    if not idx:
        return {}
    out: dict = {}
    for p in placements:
        if p in out:
            continue
        tmpls = idx.get(p.part.ldraw)
        if not tmpls:
            continue
        if rnd.random() >= rate:
            continue
        total = 0
        for t in tmpls:
            total += int(t.get("models", 1))
        if total <= 0:
            continue
        r = rnd.random() * total
        acc = 0
        tmpl = tmpls[-1]
        for t in tmpls:
            acc += int(t.get("models", 1))
            if r <= acc:
                tmpl = t
                break
        anchor_code = pal[p.color_idx].code
        child = SnotChild(
            b_ldraw=tmpl["b_ldraw"],
            b_colour=anchor_code,
            rot=tmpl["rot"],
            dy_LDU=int(tmpl["dy_s"]) * _LDU_PLATE,
            dx_LDU=int(tmpl["dx_s"]) * _LDU_STUD,
            dz_LDU=int(tmpl["dz_s"]) * _LDU_STUD,
            models=int(tmpl.get("models", 0)),
        )
        out[p] = [child]
    return out


def render_child(parent_mat: tuple[int, ...], parent_x: int, parent_y: int, parent_z: int,
                 child, child_part_origin: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> str:
    """One LDraw type-1 line for a SnotChild attached to a parent at
    LDraw world coords (parent_x, parent_y, parent_z) with 3x3 rotation
    `parent_mat` (row-major, 9-tuple).

    Child's rotation in world = parent_mat @ parse_rot(child.rot).
    Child's world position = parent position + parent_mat @ (dx_LDU,
    dy_LDU, dz_LDU) — the per-template offsets stored in A's local
    frame, rotated into world by A's matrix.

    `child_part_origin` is B's own LDraw-frame origin offset (bricks.Part
    has one for parts whose origin isn't at footprint-centre / body-top);
    pass that through to subtract after rotation.  Default zero works
    for parts whose origin is already the footprint centre."""
    local = parse_rot(child.rot)
    world_mat = _mat_mul(parent_mat, local)
    dx_w, dy_w, dz_w = _mat_vec(parent_mat, (child.dx_LDU, child.dy_LDU, child.dz_LDU))
    x = parent_x + dx_w
    y = parent_y + dy_w
    z = parent_z + dz_w
    ox, oy, oz = child_part_origin
    if ox or oy or oz:
        # Subtract B's own origin offset rotated into world
        wx, wy, wz = _mat_vec(world_mat, (ox, oy, oz))
        x -= wx; y -= wy; z -= wz
    def _fmt(n):
        return str(int(n)) if float(n).is_integer() else str(n)
    mat = " ".join(_fmt(v) for v in world_mat)
    return f"1 {child.b_colour} {_fmt(x)} {_fmt(y)} {_fmt(z)} {mat} {child.b_ldraw}"
