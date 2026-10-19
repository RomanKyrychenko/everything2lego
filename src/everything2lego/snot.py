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
        cat_by_ldraw = {p.ldraw: p for p in full_catalog()}
    except Exception:
        cat_by_ldraw = {}
    # Decoration allowlist for B: small shapes only (plate / tile / slope /
    # round, footprint area ≤ 4, height ≤ 1).  Keeps real wall-decoration
    # patterns (1x1 round, 1x2 plate sideways, 1x1 slope, 2x2 tile) and
    # drops minifig parts, bars, 1xN plates, hinge mechanisms etc. that
    # produce the "antenna sticking out of the top" look.
    _DECOR_SHAPES = frozenset(("plate", "tile", "slope", "slope_inv", "round"))

    def _is_decoration(part) -> bool:
        if part is None:
            return False
        if part.shape not in _DECOR_SHAPES:
            return False
        w, d = part.footprint
        return w * d <= 4 and part.height_units <= 1

    out = []
    for r in rows:
        rot = r.get("rot")
        if rot is None:
            continue                                                   # legacy schema; skip
        if r.get("face_a") != "U":
            continue                                                   # A sideways: pose not in sampler-placeable frame
        a_ldraw = f"{r['a']}.dat"
        b_ldraw = f"{r['b']}.dat"
        if cat_by_ldraw and (a_ldraw not in cat_by_ldraw or b_ldraw not in cat_by_ldraw):
            continue                                                   # primitive or non-catalogue
        if cat_by_ldraw and not _is_decoration(cat_by_ldraw.get(b_ldraw)):
            continue                                                   # B isn't a small decoration shape
        # Tight pose: B near A.  Templates with large horizontal / vertical
        # offsets place B far from the anchor and look like floating debris.
        # Keep only (|dx_s|, |dz_s|) ≤ 2 studs and dy_s ∈ [-1, 1] plates.
        if abs(int(r["dx"])) > 2 or abs(int(r["dz"])) > 2 or abs(int(r["dy"])) > 1:
            continue
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
    import numpy as _np
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
    # Occupancy grid at stud/plate resolution.  Used to (a) check all of B's
    # cells are empty before placing (fix 3), (b) check the anchor-adjacent
    # cell in the child's direction is empty — the "exit path" is clear (fix 2).
    if placements:
        y_lo = min(p.y_plate for p in placements)
        y_hi = max(p.y_plate + p.part.height_units for p in placements)
        x_lo = min(p.x_stud for p in placements)
        x_hi = max(p.x_stud + p.w_world for p in placements)
        z_lo = min(p.z_stud for p in placements)
        z_hi = max(p.z_stud + p.d_world for p in placements)
        H, X, Z = y_hi - y_lo + 1, x_hi - x_lo + 1, z_hi - z_lo + 1
        occ = _np.zeros((H, X, Z), dtype=bool)
        for p in placements:
            ys = p.y_plate - y_lo
            xs = p.x_stud - x_lo
            zs = p.z_stud - z_lo
            occ[ys:ys + p.part.height_units, xs:xs + p.w_world, zs:zs + p.d_world] = True
        grid_origin = (y_lo, x_lo, z_lo)
    else:
        occ = _np.zeros((1, 1, 1), dtype=bool)
        grid_origin = (0, 0, 0)
    # Accent colour: pick from the build's palette by MAX luminance contrast
    # against the primary (most-common) colour.  Prevents the "warm model gets
    # cool dark-gray accent" clash: a tan/brown primary pulls black or white
    # as the accent instead of grey.  Weighted toward colours actually used
    # in the build.
    from collections import Counter
    _col_counts = Counter(pp.color_idx for pp in placements)
    _top = _col_counts.most_common()
    def _lum(idx: int) -> float:
        r, g, b = pal[idx].rgb
        return 0.299 * r + 0.587 * g + 0.114 * b
    def _warm(idx: int) -> int:
        """+1 warm (red/orange/yellow/brown dominates), -1 cool
        (blue/cyan dominates), 0 neutral (greyscale within 20 RGB units)."""
        r, g, b = pal[idx].rgb
        if abs(r - b) < 20 and abs(g - (r + b) / 2) < 20:
            return 0
        return 1 if r > b else -1
    if _top:
        primary_idx = _top[0][0]
        primary_lum = _lum(primary_idx)
        primary_warm = _warm(primary_idx)
        # Prefer accent in the same hue family (warm/cool/neutral) as the
        # primary; within that family, pick max luminance contrast.  Falls back
        # to any-colour max-contrast if no same-family candidate exists.
        used = [(idx, cnt) for idx, cnt in _top if cnt >= 2 and idx != primary_idx]
        same_family = [(i, c) for i, c in used if _warm(i) == primary_warm]
        pool = same_family if same_family else used
        if pool:
            accent_colour_idx = max(pool, key=lambda ic: abs(_lum(ic[0]) - primary_lum))[0]
        else:
            accent_colour_idx = primary_idx
    else:
        accent_colour_idx = 0
    accent_code = pal[accent_colour_idx].code
    # Cap decorations per band so a long row doesn't turn into a mechanical
    # repeat.  5 feels enough to read as "pattern" without feeling stamped.
    BAND_CAP = 5
    band_count: dict[tuple, int] = {}
    anchor_band: dict = {}
    out: dict = {}
    rejected_overlap = 0
    rejected_inward = 0
    rejected_cells = 0
    rejected_interior = 0

    def _exterior_face(p) -> str | None:
        """Which face of the anchor is exterior (±x or ±z neighbour empty).
        Returns "+x", "-x", "+z", "-z", or None if buried.  First match wins
        so bands on the same face stay consistent."""
        ys = p.y_plate - grid_origin[0]
        xs = p.x_stud - grid_origin[1]
        zs = p.z_stud - grid_origin[2]
        y_mid = ys + max(0, p.part.height_units // 2)
        z0 = max(0, zs); z1 = min(occ.shape[2], zs + p.d_world)
        x0 = max(0, xs); x1 = min(occ.shape[1], xs + p.w_world)
        for dx, face in ((p.w_world, "+x"), (-1, "-x")):
            x = xs + dx
            if 0 <= x < occ.shape[1] and z0 < z1 and not occ[y_mid, x, z0:z1].all():
                return face
        for dz, face in ((p.d_world, "+z"), (-1, "-z")):
            z = zs + dz
            if 0 <= z < occ.shape[2] and x0 < x1 and not occ[y_mid, x0:x1, z].all():
                return face
        return None

    def _template_face_dir(tmpl) -> str | None:
        """Which face the template's offset points toward (dominant axis of
        A-frame offset dx_s, dz_s).  dy_s unused — bands are horizontal."""
        dx = int(tmpl["dx_s"]); dz = int(tmpl["dz_s"])
        if abs(dx) == 0 and abs(dz) == 0:
            return None                                       # ambiguous
        if abs(dx) >= abs(dz):
            return "+x" if dx > 0 else "-x"
        return "+z" if dz > 0 else "-z"

    # Band cache: (band_key) -> chosen template or None (skipped).  First
    # eligible anchor on a band picks the template; later anchors on the same
    # band reuse it, so contiguous same-face same-ldraw anchors get a repeating
    # decoration like real sets have.  Two band kinds:
    #   horizontal: ("h", face, y_plate, ldraw) — same row
    #   vertical:   ("v", face, orth_coord, ldraw) — same column
    # Horizontal tried first; vertical if horizontal has no face-matching
    # template.
    band_template: dict[tuple, dict] = {}

    def _h_key(p, face):
        return ("h", face, p.y_plate, p.part.ldraw)

    def _v_key(p, face):
        # orth coord: for ±x face, the vertical column runs along z;
        # for ±z face, along x.  Key by the perpendicular axis coord.
        orth = p.z_stud if face in ("+x", "-x") else p.x_stud
        return ("v", face, orth, p.part.ldraw)

    # Row index for gap-fill: (y_plate, ldraw) -> placements at that row.
    # After the main loop, we'll propagate decorations to adjacent same-ldraw
    # anchors (within 1 stud) that didn't make it through the main loop
    # (e.g. picked a different exterior face, or their band's rate roll skipped).
    row_index: dict[tuple, list] = {}
    for pp in placements:
        row_index.setdefault((pp.y_plate, pp.part.ldraw), []).append(pp)
    for p in placements:
        if p in out:
            continue
        tmpls = idx.get(p.part.ldraw)
        if not tmpls:
            continue
        face = _exterior_face(p)
        if face is None:
            rejected_interior += 1
            continue
        # Try horizontal band first, then vertical, then unoriented (fallback).
        tmpl = None
        hit_band_key = None
        for band_key, keyed_for_face in ((_h_key(p, face), True),
                                           (_v_key(p, face), True)):
            if band_key in band_template:
                tmpl = band_template[band_key]
                if tmpl is not None:
                    if band_count.get(band_key, 0) >= BAND_CAP:
                        tmpl = None
                        continue
                    hit_band_key = band_key
                    break
                continue
            # First anchor on this band: rate gate + template pick
            if rnd.random() >= rate:
                band_template[band_key] = None
                continue
            face_tmpls = [t for t in tmpls if _template_face_dir(t) == face]
            if not face_tmpls:
                face_tmpls = tmpls
            total = 0
            for t in face_tmpls:
                total += int(t.get("models", 1))
            if total <= 0:
                band_template[band_key] = None
                continue
            rr = rnd.random() * total
            acc = 0
            picked = face_tmpls[-1]
            for t in face_tmpls:
                acc += int(t.get("models", 1))
                if rr <= acc:
                    picked = t
                    break
            band_template[band_key] = picked
            tmpl = picked
            hit_band_key = band_key
            break
        if tmpl is None:
            continue
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
        # io_ldraw subtracts the anchor's own origin shift (rotated by m) from
        # its LDraw position; mirror that here so child offsets apply to the
        # same world point io_ldraw places the anchor at.  Zero for most parts.
        _ox, _oy, _oz = p.part.origin
        if _ox or _oy or _oz:
            parent_x -= m[0] * _ox + m[2] * _oz
            parent_z -= m[6] * _ox + m[8] * _oz
            parent_y -= _oy
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
        # Cell-level occupancy check (fix 3).  B's grid cell range in the
        # occupancy grid: compute from the world bbox we already have.
        if b_part is not None:
            # World bbox we already computed above in `b_cells`
            bx0, by0, bz0, bx1, by1, bz1 = b_cells[0]
            gy0 = int(round(-by1 / _LDU_PLATE)) - grid_origin[0]
            gy1 = int(round(-by0 / _LDU_PLATE)) - grid_origin[0]
            gx0 = int(round(bx0 / _LDU_STUD)) - grid_origin[1]
            gx1 = int(round(bx1 / _LDU_STUD)) - grid_origin[1]
            gz0 = int(round(bz0 / _LDU_STUD)) - grid_origin[2]
            gz1 = int(round(bz1 / _LDU_STUD)) - grid_origin[2]
            gy0 = max(0, gy0); gx0 = max(0, gx0); gz0 = max(0, gz0)
            gy1 = min(occ.shape[0], max(gy1, gy0 + 1))
            gx1 = min(occ.shape[1], max(gx1, gx0 + 1))
            gz1 = min(occ.shape[2], max(gz1, gz0 + 1))
            if gy0 < gy1 and gx0 < gx1 and gz0 < gz1:
                if occ[gy0:gy1, gx0:gx1, gz0:gz1].any():
                    rejected_cells += 1
                    continue
        # Exit-path check (fix 2): the stud/plate cell just outside the anchor,
        # in the dominant direction of the child's offset, should be EMPTY —
        # otherwise the "SNOT outward from COM but into another part" case.
        a_gy = p.y_plate - grid_origin[0] + p.part.height_units // 2
        a_gx = p.x_stud - grid_origin[1] + p.w_world // 2
        a_gz = p.z_stud - grid_origin[2] + p.d_world // 2
        step = (0, 0, 0)
        aw = abs(wdx) + abs(wdy) + abs(wdz)
        if aw > 0:
            if abs(wdx) >= abs(wdy) and abs(wdx) >= abs(wdz):
                step = (0, 1 if wdx > 0 else -1, 0)
            elif abs(wdz) >= abs(wdy):
                step = (0, 0, 1 if wdz > 0 else -1)
            else:
                step = (-1 if wdy > 0 else 1, 0, 0)             # LDraw y is DOWN
        exit_gy = a_gy + step[0]
        exit_gx = a_gx + step[1]
        exit_gz = a_gz + step[2]
        if (0 <= exit_gy < occ.shape[0] and 0 <= exit_gx < occ.shape[1]
                and 0 <= exit_gz < occ.shape[2]):
            if occ[exit_gy, exit_gx, exit_gz]:
                rejected_inward += 1
                continue
        # Attachment check: every grid cell B occupies must have at least one
        # solid 6-face neighbour (any placement, including the anchor).  Catches
        # the "flying detail" case where B's far cell sits in open air with
        # nothing to hold it.
        if b_part is not None:
            # grid cell range for B (already computed above for occupancy check)
            flying = False
            for gy in range(gy0, gy1):
                for gx in range(gx0, gx1):
                    for gz in range(gz0, gz1):
                        has_neighbour = False
                        for ny, nx, nz in ((gy-1,gx,gz),(gy+1,gx,gz),
                                            (gy,gx-1,gz),(gy,gx+1,gz),
                                            (gy,gx,gz-1),(gy,gx,gz+1)):
                            if (0 <= ny < occ.shape[0] and 0 <= nx < occ.shape[1]
                                    and 0 <= nz < occ.shape[2] and occ[ny, nx, nz]):
                                has_neighbour = True
                                break
                        if not has_neighbour:
                            flying = True
                            break
                    if flying:
                        break
                if flying:
                    break
            if flying:
                rejected_cells += 1
                continue
        anchor_code = pal[p.color_idx].code
        b_colours = production.get(tmpl["b_ldraw"])
        # Prefer the build's accent colour (second-most-common): looks
        # purposeful across many decorations.  Fall back to anchor's colour,
        # then to any colour B is made in.
        if b_colours and accent_code in b_colours:
            b_colour = accent_code
        elif b_colours and anchor_code in b_colours:
            b_colour = anchor_code
        elif b_colours:
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
        if hit_band_key is not None:
            band_count[hit_band_key] = band_count.get(hit_band_key, 0) + 1
            anchor_band[p] = hit_band_key
    # Gap-fill: for each decorated anchor, propagate its decoration to adjacent
    # same-row same-ldraw neighbours (within 1 stud ±x or ±z) that didn't pass
    # the main loop (e.g. _exterior_face picked a different side).  Keeps bands
    # continuous.  Reuses the SAME SnotChild so the offset vector is identical
    # in A's local frame; render_child applies each parent's yaw matrix.
    newly: dict = {}
    for p, kids in list(out.items()):
        ch = kids[0]
        bk = anchor_band.get(p)
        if bk is not None and band_count.get(bk, 0) >= BAND_CAP:
            continue                                           # band is full; don't gap-fill it
        for q in row_index.get((p.y_plate, p.part.ldraw), ()):
            if q is p or q in out or q in newly:
                continue
            dx = abs(q.x_stud - p.x_stud)
            dz = abs(q.z_stud - p.z_stud)
            if max(dx, dz) > p.w_world + 1:                   # too far: different wall
                continue
            if _exterior_face(q) is None:
                continue
            newly[q] = [ch]
            if bk is not None:
                band_count[bk] = band_count.get(bk, 0) + 1
                if band_count[bk] >= BAND_CAP:
                    break
    out.update(newly)
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
