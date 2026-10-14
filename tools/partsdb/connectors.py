"""Every connection point of an LDraw part, from the LDCad shadow library.

The shadow library (github.com/RolandMelkert/LDCadShadowLibrary, CC BY-SA
4.0) annotates LDraw files with `0 !LDCAD SNAP_*` metas: a stud primitive
is a male round cylinder, an anti-stud a female one, and so on.  A part's
connectors are its own metas plus everything its sub-files carry, placed
by the sub-file transforms — the same walk as the part's geometry.
Spec: https://www.melkert.net/LDCad/tech/meta

    conns = part_connectors("3001.dat")   # [Connector(kind, gender, pos, axis, ...)]

Coordinates are the part's own LDU (y down).  `axis` is the cylinder's
-Y (LDCad's convention: a stud rises along it, a hole runs up along it);
`facing(c)` is the side a connector is met from: up for a stud on top,
down for a socket underneath, sideways for SNOT.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

# The official LDraw library (ldraw.org complete.zip) first: it has the
# newer moulds; BrickLink Studio's older copy fills in what only it has.
LDRAW_OFFICIAL = Path(os.environ.get("LDRAW_OFFICIAL", "/Volumes/PRO-G40/lego_data/ldraw_official/ldraw"))
LDRAW = Path(os.environ.get("LDRAWDIR", "/Applications/Studio 2.0/ldraw"))
SHADOW = Path(os.environ.get("LDCAD_SHADOW", "/Volumes/PRO-G40/lego_data/LDCadShadowLibrary"))
_LD_DIRS = [d for root in (LDRAW_OFFICIAL, LDRAW)
            for d in (root / "parts", root / "p", root / "parts" / "s", root / "p" / "48",
                      root / "UnOfficial" / "parts", root / "UnOfficial" / "p")]
_SH_DIRS = [SHADOW / "parts", SHADOW / "p", SHADOW / "parts" / "s", SHADOW / "p" / "48"]


@dataclass
class Connector:
    kind: str            # stud, antistud, axle, axlehole, pin, pinhole, bar, barhole, clip,
                         # finger, ball, socket, generic
    gender: str          # "M" | "F" | ""
    pos: np.ndarray      # (3,) LDU in the part frame
    axis: np.ndarray     # (3,) unit: the cylinder's -Y in the part frame
    shape: str = ""      # the raw secs / bounding, for reference
    ident: str = ""


_META = re.compile(r"0\s+!LDCAD\s+(SNAP_[A-Z]+)\s*(.*)$")
_PARAM = re.compile(r"\[(\w+)=([^\]]*)\]")


def _find(dirs: list[Path], name: str) -> Path | None:
    name = name.replace("\\", "/").lower()
    for d in dirs:
        p = d / name
        if p.exists():
            return p
    return None


def _matrix(vals: list[float]) -> np.ndarray:
    x, y, z, a, b, c, d, e, f, g, h, i = vals
    return np.array([[a, b, c, x], [d, e, f, y], [g, h, i, z], [0, 0, 0, 1]], float)


def _params(text: str) -> dict[str, str]:
    return {k.lower(): v.strip() for k, v in _PARAM.findall(text)}


def _local(p: dict[str, str]) -> np.ndarray:
    """The meta's own placement (pos, ori) as a 4x4."""
    M = np.eye(4)
    if "ori" in p:
        M[:3, :3] = np.array(p["ori"].split(), float).reshape(3, 3)
    if "pos" in p:
        M[:3, 3] = np.array(p["pos"].split(), float)
    return M


def _grid(p: dict[str, str]) -> list[np.ndarray]:
    """Offsets (in the meta's frame) of a [grid=[C] xc [C] zc xs zs] repeat."""
    if "grid" not in p:
        return [np.zeros(3)]
    tok = p["grid"].split()
    vals, centred = [], []
    i = 0
    while i < len(tok) and len(vals) < 4:
        if tok[i].upper() == "C":
            centred.append(True)
            vals.append(float(tok[i + 1]))
            i += 2
        else:
            if len(vals) < 2:
                centred.append(False)
            vals.append(float(tok[i]))
            i += 1
    if len(vals) < 4:
        return [np.zeros(3)]
    xc, zc, xs, zs = int(vals[0]), int(vals[1]), vals[2], vals[3]
    cx, cz = (centred + [False, False])[:2]
    out = []
    for ix in range(xc):
        for iz in range(zc):
            x = (ix - (xc - 1) / 2) * xs if cx else ix * xs
            z = (iz - (zc - 1) / 2) * zs if cz else iz * zs
            out.append(np.array([x, 0.0, z]))
    return out


def _classify(meta: str, p: dict[str, str]) -> tuple[str, str]:
    g = p.get("gender", "").upper()
    if meta == "SNAP_CYL":
        secs = p.get("secs", "").split()
        shapes = secs[0::3]
        radii = [float(r) for r in secs[1::3]] if len(secs) >= 3 else [0.0]
        length = sum(float(l) for l in secs[2::3]) if len(secs) >= 3 else 0.0
        if "A" in shapes:
            return ("axle" if g == "M" else "axlehole"), g
        if max(radii) <= 4.5:
            return ("bar" if g == "M" else "barhole"), g
        if g == "M":
            return ("stud" if length <= 6 else "pin"), g
        # a female round 6 throughout (any length: a brick's socket runs
        # its height) or square takes a stud; collars (radius 8) make it a
        # Technic pin hole
        if "S" in shapes or all(abs(r - 6) < 0.5 for r in radii):
            return "antistud", g
        return "pinhole", g
    if meta == "SNAP_CLP":
        return "clip", "F"
    if meta == "SNAP_FGR":
        return "finger", p.get("genderofs", "").upper()
    if meta == "SNAP_SPH":
        return ("ball" if g == "M" else "socket"), g
    return "generic", g


@lru_cache(maxsize=None)
def _shadow_lines(name: str) -> tuple[tuple[str, dict], ...]:
    f = _find(_SH_DIRS, name)
    if f is None:
        return ()
    out = []
    for ln in f.read_text(errors="replace").splitlines():
        m = _META.match(ln.strip())
        if m:
            out.append((m.group(1), _params(m.group(2))))
    return tuple(out)


@lru_cache(maxsize=None)
def _ldraw_refs(name: str) -> tuple[tuple[str, tuple[float, ...]], ...]:
    f = _find(_LD_DIRS, name)
    if f is None:
        return ()
    out = []
    for ln in f.read_text(errors="replace").splitlines():
        t = ln.split()
        if len(t) >= 15 and t[0] == "1":
            out.append((" ".join(t[14:]).lower(), tuple(float(v) for v in t[2:14])))
    return tuple(out)


def _raw(name: str, depth: int = 0) -> list[tuple[str, str, np.ndarray, str, str]]:
    """(kind, gender, 4x4 placement, shape, id) in `name`'s own frame."""
    if depth > 14:
        return []
    metas = _shadow_lines(name)
    inherited: list = []
    for sub, vals in _ldraw_refs(name):
        M = _matrix(list(vals))
        for kind, g, P, shape, ident in _raw(sub, depth + 1):
            inherited.append((kind, g, M @ P, shape, ident))
    own: list = []
    for meta, p in metas:
        if meta == "SNAP_CLEAR":
            ident = p.get("id", "")
            inherited = [c for c in inherited if ident and c[4] != ident] if ident else []
            continue
        base = _local(p)
        if meta == "SNAP_INCL":
            ref = p.get("ref", "")
            for off in _grid(p):
                G = base.copy()
                G[:3, 3] += G[:3, :3] @ off
                for kind, g, P, shape, ident in _raw(ref, depth + 1):
                    own.append((kind, g, G @ P, shape, ident))
            continue
        kind, g = _classify(meta, p)
        shape = p.get("secs") or p.get("bounding") or p.get("seq") or ""
        for off in _grid(p):
            G = base.copy()
            G[:3, 3] += G[:3, :3] @ off
            own.append((kind, g, G, shape, p.get("id", "")))
    return inherited + own


def facing(c: Connector) -> np.ndarray:
    """The direction a connector meets its partner from: a stud points
    along its axis, a hole opens against it."""
    return c.axis if c.gender == "M" else -c.axis


@lru_cache(maxsize=None)
def part_connectors(name: str, *, height: float | None = None) -> tuple[Connector, ...]:
    """`height`: the part's body height in LDU (the bottom of its box, y
    down); with it, a System part gets the socket under each of its
    upward studs that the shadow library leaves implicit (1xN plates)."""
    out = []
    for kind, g, P, shape, ident in _raw(name.lower()):
        axis = -P[:3, 1]
        n = np.linalg.norm(axis)
        out.append(Connector(kind, g, P[:3, 3].copy(), axis / n if n else axis, shape, ident))
    if height is not None:
        down = [c for c in out if c.kind == "antistud" and facing(c)[1] > 0.9]
        for c in [c for c in out if c.kind == "stud" and facing(c)[1] < -0.9 and abs(c.pos[1]) < 1]:
            if not any(abs(d.pos[0] - c.pos[0]) < 2 and abs(d.pos[2] - c.pos[2]) < 2 for d in down):
                out.append(Connector("antistud", "F", np.array([c.pos[0], height, c.pos[2]]),
                                     np.array([0.0, -1.0, 0.0]), "implicit", "implicit"))
    return tuple(out)


def summary(conns) -> dict[str, int]:
    """Counts by kind and, for studs and anti-studs, by facing: up, down,
    side (studs not on top: SNOT)."""
    s: dict[str, int] = {}
    for c in conns:
        key = c.kind
        if c.kind in ("stud", "antistud"):
            ay = facing(c)[1]
            face = "up" if ay < -0.9 else "down" if ay > 0.9 else "side"
            key = f"{c.kind}_{face}"
        s[key] = s.get(key, 0) + 1
    return s
