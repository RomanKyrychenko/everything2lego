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
from dataclasses import dataclass
from pathlib import Path

import numpy as np

LDRAW = Path(os.environ.get("LDRAWDIR", "/Applications/Studio 2.0/ldraw"))
CACHE = Path(os.environ.get("SET_STUDY_CACHE", "/Volumes/PRO-G40/lego_data/cache"))
SEARCH = [LDRAW / "parts", LDRAW / "p", LDRAW / "p" / "48", LDRAW / "parts" / "s",
          LDRAW / "UnOfficial" / "parts", LDRAW / "UnOfficial" / "p"]


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
            if sub in files:
                walk(sub, W, c, trail + (sub,))
            else:
                out.append(Part(sub, c, W, step[0], trail))

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
