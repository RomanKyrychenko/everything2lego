"""Sideways (SNOT) options for the sampler: small plates and tiles turned on their side, held by a side stud.

A sideways option is an ordinary Orient, so both engines treat it like any other: its shape is drawn into the
4x4x4 sub-cell masks (a plate is 8 LDU thick: two of the four 5-LDU samples across), its anti-studs become side
connectors facing the anchor (sampler.CONNECTORS "antistud:F"), so `side_held` supports it when they meet a side
stud ("stud:M"); it has no top / bottom studs.  The part's bottom edge sits 4 LDU above the box floor, which puts
its first hole 14 LDU up -- where the common side-stud bricks (87087, 30414: hy = 4 half plates) have their stud.
How often real sets mount each part sideways (data/snot_share.json) is the option's prior.

    python -m everything2lego.snot_options          # rebuild data/snot_share.json from the OMR sets
"""
from __future__ import annotations

import dataclasses
import json
import math
from functools import lru_cache
from importlib.resources import files
from pathlib import Path

import numpy as np

from .sampler import CONNECTORS, SUB, Orient, _halo, pack, popcount

Y0 = 4                     # LDU from the box floor to the part's bottom edge
THICK = 8                  # plate / tile thickness, LDU
MAX_SIDE = 4               # footprint sides up to 4 studs
STUD_UP = 14               # side studs of 87087, 30414, 11211, 4733, 26604, 4070: 10 LDU under the top = 14 over the base
SHARE = "snot_share.json"


@lru_cache(maxsize=1)
def sideways_share() -> dict:
    """ldraw -> share of its real-set placements that are mounted sideways."""
    f = files("everything2lego.data").joinpath(SHARE)
    return json.loads(f.read_text())["share"] if f.is_file() else {}


def _orient(a: int, b: int, face: tuple[int, int], vertical_first: bool) -> Orient:
    """A footprint (a x b studs; a along the part's local x) turned so its anti-studs face `face` (dx, dz);
    vertical_first: local x goes up (else local x lies along the face, local z goes up)."""
    fx, fz = face
    vert, horiz = (a, b) if vertical_first else (b, a)
    h = math.ceil((Y0 + 20 * vert) / 8)
    W, D = (1, horiz) if fx else (horiz, 1)
    fine = np.zeros((h * SUB, W * SUB, D * SUB), bool)
    ys = slice(Y0 // 2, (Y0 + 20 * vert) // 2)                                    # 2 LDU per sample upward
    t = round(THICK / 5)                                                           # 5 LDU per sample across
    if fx: fine[ys, (W * SUB - t if fx > 0 else 0):(W * SUB if fx > 0 else t), :] = True
    else: fine[ys, :, (D * SUB - t if fz > 0 else 0):(D * SUB if fz > 0 else t)] = True
    bits = pack(fine, (h, W, D)); cells = np.argwhere(bits != 0); cb = bits[tuple(cells.T)]
    hole = CONNECTORS["antistud:F"]; sides = []
    for i in range(vert):                                                          # a hole per stud of the footprint
        if (Y0 + 10 + 20 * i - STUD_UP) % 8:
            continue                                   # holes 20 LDU apart vs studs 8k apart: only these can meet one exactly
        hy = round((Y0 + 10 + 20 * i) / 4)                                         # half plates up the box
        for j in range(horiz):
            if fx: sides.append((2 * W if fx > 0 else 0, hy, 2 * j + 1, fx, 0, hole))
            else: sides.append((2 * j + 1, hy, 2 * D if fz > 0 else 0, 0, fz, hole))
    # LDraw pose: local +y (towards the anti-studs) -> face; local x -> up or along the face
    f = np.array([fx, 0, fz], float); up = np.array([0, -1, 0], float); along = np.array([1, 0, 0] if fz else [0, 0, 1], float)
    ex = up if vertical_first else along; ez = np.cross(ex, f)
    R = np.stack([ex, f, ez], 1)                                                   # columns: images of local x, y, z
    # origin = centre of the top face (the outward one), box-local LDU (x right, y UP, z)
    lx = 10 * W if not fx else (20 * W - THICK if fx > 0 else THICK)
    lz = 10 * D if not fz else (20 * D - THICK if fz > 0 else THICK)
    snot = (tuple(float(v) for v in R.ravel()), (float(lx), float(Y0 + 10 * vert), float(lz)))
    pix = (np.argwhere(fine.any(axis=1)), np.argwhere(fine.any(axis=0)), np.argwhere(fine.any(axis=2)))
    rot = {(1, 0): 1001, (-1, 0): 1002, (0, 1): 1003, (0, -1): 1004}[face] + (10 if vertical_first else 0)   # never a yaw
    return Orient(rot, (h, W, D), cells, cb, popcount(cb) / SUB ** 3, tuple(int(v) for v in cells[0]),
                  frozenset(), frozenset(), tuple(sides), _halo(cells, (h, W, D)), pix, snot)


def sideways_orients(vp) -> tuple:
    """The sideways options of a small plain plate or tile (none for anything else)."""
    if vp.box not in ("plate", "tile"): return ()
    a, b = vp.part.footprint
    if max(a, b) > MAX_SIDE: return ()
    out = []
    for face in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        out.append(_orient(a, b, face, True))
        if a != b: out.append(_orient(a, b, face, False))
    return tuple(out)


def pair_rows(opts: list) -> np.ndarray:
    """(n, 7) int32 rows for the engine's sideways move: (plain brick option, side-stud anchor option with the same
    box and cells, sideways option, dy, dx, dz of its box from the brick's, anchor stud index).  Offsets solve the
    connector meeting rule (sampler.sides_join): facing each other, same half-stud point, height within half a plate.
    Per anchor stud the larger sideways parts come first (the engine takes the first that fits)."""
    stud = CONNECTORS["stud:M"]; hole = CONNECTORS["antistud:F"]
    sig = lambda o: (o.size, tuple(sorted(map(tuple, o.cells.tolist()))))              # noqa: E731
    anchors: dict = {}
    for k, (v, o) in enumerate(opts):
        if o.snot is None and any(s[5] == stud for s in o.sides):
            anchors.setdefault(sig(o), []).append(k)
    side = [(k, o) for k, (v, o) in enumerate(opts) if o.snot is not None]
    side.sort(key=lambda ko: -len(ko[1].sides))                                     # more holes = larger part first
    rows = []
    for k, (v, o) in enumerate(opts):
        if v.box != "brick" or o.snot is not None:
            continue
        for ak in anchors.get(sig(o), ()):
            ao = opts[ak][1]
            for si, (hx, hy, hz, dx, dz, kind) in enumerate(ao.sides):
                if kind != stud:
                    continue
                for tk, to in side:
                    for (gx, gy, gz, ex, ez, kb) in to.sides:
                        if kb != hole or ex != -dx or ez != -dz or (hx - gx) % 2 or (hz - gz) % 2:
                            continue
                        for ty in {(hy - gy) // 2, (hy - gy + 1) // 2}:
                            if abs(2 * ty + gy - hy) <= 1:
                                rows.append((k, ak, tk, ty, (hx - gx) // 2, (hz - gz) // 2, si))
    return np.array(rows, np.int32).reshape(-1, 7)


def with_sideways(vocab: tuple) -> tuple:
    """The vocabulary with sideways options added to its small plates and tiles."""
    return tuple(dataclasses.replace(v, orients=v.orients + sideways_orients(v)) if sideways_orients(v) else v for v in vocab)


def build(out: Path) -> None:
    import sys
    from collections import Counter
    root = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(root / "tools" / "set_study")]
    from ldraw_read import flatten
    tot, side = Counter(), Counter()
    fs = sorted(Path("/Volumes/PRO-G40/lego_data/omr/files").glob("*.mpd"))
    for k, f in enumerate(fs):
        try: ps = flatten(f)
        except Exception: continue                                                 # noqa: BLE001
        for p in ps:
            tot[p.name] += 1; side[p.name] += abs(p.M[1, 1]) < 0.5
        if k % 400 == 0: print(f"  {k}/{len(fs)}", flush=True)
    share = {n: side[n] / tot[n] for n in tot if tot[n] >= 20}
    out.write_text(json.dumps({"_source": "snot_options.build: share of OMR placements with the part's y axis horizontal", "share": share}))
    print(f"wrote {out}: {len(share)} parts; e.g. 3069b {share.get('3069b.dat', 0):.3f}  3070b {share.get('3070b.dat', 0):.3f}  3023 {share.get('3023.dat', 0):.3f}")


if __name__ == "__main__":
    build(Path(__file__).resolve().parent / "data" / SHARE)
