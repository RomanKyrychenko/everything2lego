"""Building patterns mined from real sets (tools/learn/motifs.py), as rows the native sampler places in one move."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DATA = Path(__file__).parent / "data" / "motifs.json"


def tables(opts, kind: str | None = "building"):
    """(start (n + 1,) int32, rows (m, 4) int32 of option, dy, dx, dz, weight (n,) = sets that use it) for the
    patterns whose parts are all among these options (upright, as mined)."""
    if not DATA.exists():
        return np.zeros(1, np.int32), np.zeros((0, 4), np.int32), np.zeros(0)
    d = json.loads(DATA.read_text())
    if kind is not None and d.get("kind") != kind:
        return np.zeros(1, np.int32), np.zeros((0, 4), np.int32), np.zeros(0)
    index = {(v.part.ldraw.lower(), o.rot): k for k, (v, o) in enumerate(opts)
             if getattr(o, "snot", None) is None and getattr(o, "stack", None) is None}
    start, rows, w = [0], [], []
    for m in d["motifs"]:
        ks = [index.get((ld, rot)) for ld, rot, *_ in m["parts"]]
        if any(k is None for k in ks):
            continue
        rows += [(k, dy, dx, dz) for k, (_, _, dy, dx, dz) in zip(ks, m["parts"])]
        start.append(len(rows)); w.append(float(m["sets"]))
    return np.array(start, np.int32), np.array(rows, np.int32).reshape(-1, 4), np.array(w)
