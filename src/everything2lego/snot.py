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
    loader to swap A and B and invert the pose, future work.

    Also filtered: B parts not in the brick catalogue are dropped.  The
    mining passes through anything that `part_bbox` can measure, which
    includes LDraw primitives like `rect3.dat`, `box5.dat`, `s/2431s01.dat`
    — internal shapes used inside parts, which render as plain blocks
    when placed on their own.  Keeping only catalogue parts ensures the
    SNOT children are real bricks."""
    if not SNOT_PAIRS_PATH.exists():
        return ()
    rows = json.loads(SNOT_PAIRS_PATH.read_text())
    # Lazy catalogue load so this module can import without bricks' deps.
    try:
        from .bricks import full_catalog
        catalogue_ldraws = frozenset(p.ldraw for p in full_catalog())
    except Exception:
        catalogue_ldraws = frozenset()
    out = []
    for r in rows:
        rot = r.get("rot")
        if rot is None:
            continue                                                   # legacy schema; skip
        if r.get("face_a") != "U":
            continue                                                   # A sideways: pose not in sampler-placeable frame
        a_ldraw = f"{r['a']}.dat"
        b_ldraw = f"{r['b']}.dat"
        if catalogue_ldraws and (a_ldraw not in catalogue_ldraws or b_ldraw not in catalogue_ldraws):
            continue                                                   # primitive or non-catalogue part — would render as a plain block
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


# Yaw rotation matrices in LDraw world (y down, right-handed), used to rotate
# A-frame SNOT offsets into world LDU.  Keyed by degrees; mirrors io_ldraw._ROT
# but kept local so snot.py has no io_ldraw import at module load.
_YAW_MAT = {
    0:   (1, 0, 0,  0, 1, 0,  0, 0, 1),
    90:  (0, 0, 1,  0, 1, 0, -1, 0, 0),
    180: (-1, 0, 0, 0, 1, 0,  0, 0, -1),
    270: (0, 0, -1, 0, 1, 0,  1, 0, 0),
}


def _bbox_LDU(p) -> tuple[int, int, int, int, int, int]:
    """A placement's LDU bbox (x0, x1, y0, y1, z0, z1), y down.  Based on
    the Placement's grid position and footprint; approximate (ignores
    non-rectangular parts, part.origin shifts, etc)."""
    x0 = p.x_stud * _LDU_STUD
    x1 = (p.x_stud + p.w_world) * _LDU_STUD
    z0 = p.z_stud * _LDU_STUD
    z1 = (p.z_stud + p.d_world) * _LDU_STUD
    y0 = -(p.y_plate + p.part.height_units) * _LDU_PLATE
    y1 = -p.y_plate * _LDU_PLATE
    return x0, x1, y0, y1, z0, z1


def decorate(placements, seed: int = 0, rate: float = 0.2) -> dict:
    """Post-process a list of Placements into a decorations dict (keyed
    by Placement, value is a list of SnotChild).  Each matching anchor
    gets a child with probability `rate`, drawn by model-count weight,
    IFF the child's world position doesn't fall inside another
    placement's bbox (coarse sub-plate collision check).  `rate=0.2`
    targets real-set SNOT fraction (17-32% per OMR audit).  Doesn't need
    sampler state — works after any engine.

    Returns {} when snot_adj.json is absent / lacks the v2 `rot` field.
    Colour matching: child uses the anchor's colour if B is produced in
    that colour (bricks._production), otherwise falls back to a colour
    B is actually made in.  Position: offsets corrected for both A's
    and B's lo-to-origin shifts (vocab-driven), so non-1x1 B parts
    render at their true pose.

    The collision check is coarse: compares the child's lo-corner LDU
    point against each other placement's axis-aligned LDU bbox.  Catches
    the common "SNOT fell inside a wall" case; doesn't guarantee full-
    bbox non-overlap."""
    import random
    from .colors import palette
    from .bricks import full_catalog, _production
    pal = palette()
    rnd = random.Random(seed)
    idx = by_a_ldraw()
    if not idx:
        return {}
    cat_by_ldraw = {p.ldraw: p for p in full_catalog()}
    production = _production()
    placements = list(placements)
    bboxes = [(p,) + _bbox_LDU(p) for p in placements]
    # Model's centre of mass in LDU (used for the outward-facing check: a child
    # whose world position is CLOSER to the COM than the anchor is heading
    # inward, which usually hides the decoration and looks wrong).
    if placements:
        import numpy as _np
        _cs = _np.array([[((2*p.x_stud + p.w_world - 1) * (_LDU_STUD // 2)),
                            -(p.y_plate + p.part.height_units) * _LDU_PLATE,
                            ((2*p.z_stud + p.d_world - 1) * (_LDU_STUD // 2))]
                            for p in placements])
        com = _cs.mean(axis=0)
    else:
        com = (0, 0, 0)
    out: dict = {}
    rejected_overlap = 0
    rejected_inward = 0
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
        # Child's LDU lo-corner in world: parent position + yaw-rotated A-frame offset.
        # Correction: mining stored B.lo - A.lo (lo-corner-to-lo-corner in A's local
        # frame) but render_child needs B.origin - A.origin.  io_ldraw places the part
        # origin at the "centre of the footprint, top of body"; in LDU that offset-
        # from-lo is ((w-1)*10, 0, (d-1)*10).  Subtract A's and add B's lo-to-origin
        # shifts (B's rotated into A's frame via `rot`) to turn the mined lo-to-lo
        # offset into A.origin-to-B.origin.
        parent_x = (2 * p.x_stud + p.w_world - 1) * (_LDU_STUD // 2)
        parent_y = -(p.y_plate + p.part.height_units) * _LDU_PLATE
        parent_z = (2 * p.z_stud + p.d_world - 1) * (_LDU_STUD // 2)
        m = _YAW_MAT[(p.rotation_deg + p.part.yaw) % 360]
        # A part at x_stud=0 has its world lo at x=-10 (half a stud left of the
        # footprint centre); io_ldraw puts the LDraw origin at (w-1)*10.  So
        # lo-to-origin = (w-1)*10 - (-10) = w*10.  Same for z.
        a_cx = p.w_world * (_LDU_STUD // 2)
        a_cz = p.d_world * (_LDU_STUD // 2)
        b_part = cat_by_ldraw.get(tmpl["b_ldraw"])
        if b_part is None:
            b_corr = (0, 0, 0)
        else:
            b_cx = b_part.footprint[0] * (_LDU_STUD // 2)
            b_cz = b_part.footprint[1] * (_LDU_STUD // 2)
            r = parse_rot(tmpl["rot"])
            # B's lo-to-origin vector in B's frame, rotated into A's frame
            b_corr = (r[0] * b_cx + r[2] * b_cz,
                       r[3] * b_cx + r[5] * b_cz,
                       r[6] * b_cx + r[8] * b_cz)
        dxs = int(tmpl["dx_s"]) * _LDU_STUD - a_cx + b_corr[0]
        dys = int(tmpl["dy_s"]) * _LDU_PLATE + b_corr[1]
        dzs = int(tmpl["dz_s"]) * _LDU_STUD - a_cz + b_corr[2]
        wdx = m[0] * dxs + m[1] * dys + m[2] * dzs
        wdy = m[3] * dxs + m[4] * dys + m[5] * dzs
        wdz = m[6] * dxs + m[7] * dys + m[8] * dzs
        bx = parent_x + wdx
        by = parent_y + wdy
        bz = parent_z + wdz
        # Full-bbox collision: B's own LDU bbox (from vocab) vs every other
        # placement's bbox.  Rotate B's local extent via rot before translating.
        if b_part is None:
            # Unknown B footprint: fall back to the old lo-corner-point check.
            b_cells = [(bx, by, bz, bx + 1, by + 1, bz + 1)]
        else:
            # In B's own frame: span (0..w*20, 0..h*8, 0..d*20) relative to B.lo.
            b_w = b_part.footprint[0] * _LDU_STUD
            b_d = b_part.footprint[1] * _LDU_STUD
            b_h = b_part.height_units * _LDU_PLATE
            # Transform B's 8 corners by (parent_mat @ parse_rot(rot)) and offset to world.
            r_local = parse_rot(tmpl["rot"])
            r_world = _mat_mul(m, r_local)
            corners_local = [(0, 0, 0), (b_w, 0, 0), (0, b_h, 0), (b_w, b_h, 0),
                              (0, 0, b_d), (b_w, 0, b_d), (0, b_h, b_d), (b_w, b_h, b_d)]
            # B's lo (relative to B's origin, in B's own frame) = (-b_w/2, 0, -b_d/2);
            # we already shifted to A.origin-to-B.origin, so world B.origin = (bx,by,bz).
            # B.lo world = (bx,by,bz) + r_world @ (-b_w/2, 0, -b_d/2)
            lo_local = (-b_w // 2, 0, -b_d // 2)
            lo_shift = _mat_vec(r_world, lo_local)
            world_corners = [(bx + lo_shift[0] + c[0], by + lo_shift[1] + c[1],
                                bz + lo_shift[2] + c[2]) for c in
                               [_mat_vec(r_world, cc) for cc in corners_local]]
            xs = [c[0] for c in world_corners]
            ys = [c[1] for c in world_corners]
            zs = [c[2] for c in world_corners]
            b_cells = [(min(xs), min(ys), min(zs), max(xs), max(ys), max(zs))]
        collide = False
        for bx0, by0, bz0, bx1, by1, bz1 in b_cells:
            # inflate by 1 LDU inward so we don't reject touching-but-not-overlapping
            margin = 2
            for q, qx0, qx1, qy0, qy1, qz0, qz1 in bboxes:
                if q is p:
                    continue
                if (bx0 + margin < qx1 and bx1 - margin > qx0
                        and by0 + margin < qy1 and by1 - margin > qy0
                        and bz0 + margin < qz1 and bz1 - margin > qz0):
                    collide = True
                    break
            if collide:
                break
        if collide:
            rejected_overlap += 1
            continue
        # Outward-facing check: dot((child - anchor), (anchor - COM)) must be
        # positive — the child moves AWAY from the model's centre of mass.
        a_to_com_x = parent_x - com[0]
        a_to_com_y = parent_y - com[1]
        a_to_com_z = parent_z - com[2]
        outward = wdx * a_to_com_x + wdy * a_to_com_y + wdz * a_to_com_z
        if outward < 0:
            rejected_inward += 1
            continue
        anchor_code = pal[p.color_idx].code
        b_colours = production.get(tmpl["b_ldraw"])
        if b_colours and anchor_code not in b_colours:
            # anchor's colour isn't made for B — pick the first colour B *is* made in
            b_colour = next(iter(b_colours), anchor_code)
        else:
            b_colour = anchor_code
        child = SnotChild(
            b_ldraw=tmpl["b_ldraw"],
            b_colour=b_colour,
            rot=tmpl["rot"],
            dy_LDU=dys,
            dx_LDU=dxs,
            dz_LDU=dzs,
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
