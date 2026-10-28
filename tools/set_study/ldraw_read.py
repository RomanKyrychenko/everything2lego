"""Read official LDraw models (the OMR) into flat part lists, for studying
how real sets are built.

    parts = flatten("10037-1.mpd")   # [Part(name, colour, M 4x4, step, path)]
    lo, hi = part_bbox("3001.dat")   # LDU, in the part's own frame

Coordinates are LDraw's: LDU (a stud is 20, a plate 8), y points down.
Part sizes come from the parts library's own geometry (lines 3 and 4, the
triangles and quads), cached in `$SET_STUDY_CACHE/bbox.json`.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from functools import lru_cache

import numpy as np

LDRAW = Path(os.environ.get("LDRAWDIR", "/Applications/Studio 2.0/ldraw"))
# the official ldraw.org library first (newer moulds), Studio's copy after
LDRAW_OFFICIAL = Path(os.environ.get("LDRAW_OFFICIAL", "/Volumes/PRO-G40/lego_data/ldraw_official/ldraw"))
CACHE = Path(os.environ.get("SET_STUDY_CACHE", "/Volumes/PRO-G40/lego_data/cache"))
SEARCH = [d for root in (LDRAW_OFFICIAL, LDRAW)
          for d in (root / "parts", root / "p", root / "p" / "48", root / "parts" / "s",
                    root / "UnOfficial" / "parts", root / "UnOfficial" / "p")]


@dataclass
class Part:
    name: str           # e.g. "3001.dat", lower case
    colour: int
    M: np.ndarray       # 4x4 world transform, LDU
    step: int           # build step, counted across the whole model
    path: tuple         # sub-models it sits in, outermost first


def _matrix(t: list[str]) -> np.ndarray:
    x, y, z, a, b, c, d, e, f, g, h, i = map(float, t)
    return np.array([[a, b, c, x], [d, e, f, y], [g, h, i, z], [0, 0, 0, 1]])


def _read_mpd(path: Path) -> tuple[str, dict[str, list[str]]]:
    """Main model name and every sub-file's lines (an .ldr is one file)."""
    files: dict[str, list[str]] = {}
    cur, main = None, None
    for ln in path.read_text(errors="replace").splitlines():
        t = ln.strip()
        if t.lower().startswith("0 file "):
            cur = t[7:].strip().lower()
            files[cur] = []
            main = main or cur
            continue
        if t.lower().startswith("0 nofile"):
            cur = None
            continue
        if cur is None and not files:
            cur = main = path.name.lower()
            files[cur] = []
        if cur is not None:
            files[cur].append(t)
    return main, files


_EXPORT_PREFIX = re.compile(r"^.+? - ")
_LSYNTH = re.compile(r"^ls\d+[a-z]?\.dat$")                                 # LSynth hose/cable segments and constraints


def _title(lines: list[str]) -> str:
    return next((t[1:].strip() for t in lines if t.startswith("0")), "")


def _org(lines: list[str]) -> str | None:
    """Kind of an MPD-embedded file from its !LDRAW_ORG header: part/shortcut/subpart/primitive, None for models."""
    for t in lines[:20]:
        tok = t.lower().split()
        if len(tok) > 2 and tok[:2] == ["0", "!ldraw_org"]:
            kind = tok[2].removeprefix("unofficial_")
            return None if kind == "model" else kind
    return None


@lru_cache(maxsize=None)
def _is_fragment(name: str) -> bool:
    """Primitive or subpart reference (p/, p/48, parts/s), or a "~" part (hose / cable / tube segment that only
    draws a flexible element): real parts live flat in parts/ and their title doesn't start with "~"."""
    if "/" in name:
        return True
    lib = [root / d / name for root in (LDRAW_OFFICIAL, LDRAW) for d in ("parts", "UnOfficial/parts")]
    f = next((q for q in lib if q.exists()), None)
    if f is None:
        return any((root / d / name).exists() for root in (LDRAW_OFFICIAL, LDRAW) for d in ("p", "UnOfficial/p"))
    with f.open(errors="replace") as fh:
        title = fh.readline().strip()
    if title.lower().startswith("0 ~moved to "):                         # renamed part: judge by its new name
        return _is_fragment(title.split()[-1].lower().removesuffix(".dat") + ".dat")
    return title.startswith("0 ~")


def _find(name: str) -> Path | None:
    name = name.replace("\\", "/")
    for d in SEARCH:
        p = d / name
        if p.exists():
            return p
        p = d / name.lower()
        if p.exists():
            return p
    return None


def flatten(path: str | Path) -> list[Part]:
    main, files = _read_mpd(Path(path))
    out: list[Part] = []
    step = [0]

    def walk(name: str, M: np.ndarray, colour: int, trail: tuple) -> None:
        for t in files[name]:
            tok = t.split()
            if not tok:
                continue
            if tok[0] == "0" and len(tok) > 1 and tok[1].upper() in ("STEP", "ROTSTEP"):
                step[0] += 1
            if tok[0] != "1" or len(tok) < 15:
                continue
            c = int(tok[1]) if tok[1].lstrip("-").isdigit() else 16
            c = colour if c == 16 else c
            sub = " ".join(tok[14:]).lower().replace("\\", "/")
            W = M @ _matrix(tok[2:14])
            org = _org(files[sub]) if sub in files else None
            if sub in files and org is None:                               # a sub-model: open it
                walk(sub, W, c, trail + (sub,))
            elif org in ("part", "shortcut"):                                 # a part embedded by the exporter
                base = _EXPORT_PREFIX.sub("", sub)                              # Studio: "42131 - 61332.dat" -> "61332.dat"
                if not (_LSYNTH.match(base) or _title(files[sub]).startswith("~")):
                    out.append(Part(base, c, W, step[0], trail))
            elif org is None and not _is_fragment(sub):
                out.append(Part(sub, c, W, step[0], trail))                # a library part
            # else a primitive / subpart drawn straight into a model (strings, hoses): geometry, not a part

    walk(main, np.eye(4), 16, ())
    return out


_bbox: dict[str, list] | None = None


def _geometry_bbox(name: str, depth: int = 0) -> tuple[np.ndarray, np.ndarray] | None:
    f = _find(name)
    if f is None or depth > 12:
        return None
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for ln in f.read_text(errors="replace").splitlines():
        tok = ln.split()
        if not tok:
            continue
        if tok[0] in ("3", "4"):
            n = int(tok[0])
            pts = np.array(tok[2:2 + 3 * n], float).reshape(n, 3)
            lo, hi = np.minimum(lo, pts.min(0)), np.maximum(hi, pts.max(0))
        elif tok[0] == "1" and len(tok) >= 15:
            sub = _geometry_bbox(" ".join(tok[14:]), depth + 1)
            if sub is None:
                continue
            M = _matrix(tok[2:14])
            corners = np.array([[x, y, z, 1] for x in (sub[0][0], sub[1][0])
                                for y in (sub[0][1], sub[1][1]) for z in (sub[0][2], sub[1][2])])
            w = (M @ corners.T).T[:, :3]
            lo, hi = np.minimum(lo, w.min(0)), np.maximum(hi, w.max(0))
    if not np.isfinite(lo).all():
        return None
    return lo, hi


def part_bbox(name: str) -> tuple[np.ndarray, np.ndarray] | None:
    """The part's bounding box in its own frame, LDU (None if not in the library)."""
    global _bbox
    if _bbox is None:
        f = CACHE / "bbox.json"
        _bbox = json.loads(f.read_text()) if f.exists() else {}
    key = name.lower()
    if key not in _bbox:
        b = _geometry_bbox(key)
        _bbox[key] = None if b is None else [b[0].tolist(), b[1].tolist()]
    b = _bbox[key]
    return None if b is None else (np.array(b[0]), np.array(b[1]))


def save_cache() -> None:
    if _bbox is not None:
        CACHE.mkdir(parents=True, exist_ok=True)
        (CACHE / "bbox.json").write_text(json.dumps(_bbox))
