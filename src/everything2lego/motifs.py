"""Building patterns mined from real sets (tools/learn/motifs.py), as rows the native sampler places in one move."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DATA = Path(__file__).parent / "data" / "motifs.json"


def tables(opts, kind: str | None = "building"):
    """(start (n + 1,) int32, rows (m, 4) int32 of option, dy, dx, dz, weight (n,) = sets that use it, type (n,)
    0 anywhere / 1 on a pitched roof) for the patterns whose parts are all among these options (upright, as mined)."""
    none = np.zeros(1, np.int32), np.zeros((0, 4), np.int32), np.zeros(0), np.zeros(0, np.uint8)
    if not DATA.exists():
        return none
    d = json.loads(DATA.read_text())
    if kind is not None and d.get("kind") != kind:
        return none
    index = {(v.part.ldraw.lower(), o.rot): k for k, (v, o) in enumerate(opts)
             if getattr(o, "snot", None) is None and getattr(o, "stack", None) is None}
    start, rows, w, ty = [0], [], [], []
    for m in d["motifs"]:
        ks = [index.get((ld, rot)) for ld, rot, *_ in m["parts"]]
        if any(k is None for k in ks):
            continue
        rows += [(k, dy, dx, dz) for k, (_, _, dy, dx, dz) in zip(ks, m["parts"])]
        start.append(len(rows)); w.append(float(m["sets"])); ty.append(m.get("type", 0))
    return np.array(start, np.int32), np.array(rows, np.int32).reshape(-1, 4), np.array(w), np.array(ty, np.uint8)
