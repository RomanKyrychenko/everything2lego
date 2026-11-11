"""Merge neighbouring plain parts into the one real part their union is: a final clean-up of a build.

Repeatedly joins two upright plain bricks / plates / tiles on the same layer, of the same kind and height, whose
union is a rectangle a catalogue part has (in that colour) -- so rows and blocks of 3, 4 or more end up as one part --
and three stacked plates of one footprint and colour into the brick they make.  --colour-tol (Lab units) lets
neighbours of different colour merge: the larger part's colour wins (0 = same colour only, as the sampler's own merge).
Sideways and special parts are written back unchanged.

    .venv-arm/bin/python tools/merge_parts.py in.ldr out.ldr [--colour-tol 30]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("ldr", type=Path); ap.add_argument("out", type=Path)
    ap.add_argument("--colour-tol", type=float, default=0.0); ap.add_argument("--kind", default="building")
    a = ap.parse_args()
    from everything2lego.colors import _palette_lab, palette
    from everything2lego.io_ldraw import placement_to_line
    from everything2lego.pack import Placement
    from everything2lego.sampler import _boxes, select_vocabulary
    V = select_vocabulary(a.kind, 160); by = {v.part.ldraw: v for v in V}; boxes = _boxes(V)
    pal = palette(); ci = {c.code: i for i, c in enumerate(pal)}; lab = _palette_lab()
    keep, items = [], []
    for ln in a.ldr.read_text().splitlines():
        t = ln.split()
        v = by.get(t[-1]) if ln.startswith("1 ") else None
        M = np.array(t[5:14], float).reshape(3, 3) if v else None
        if v is None or not v.box or abs(M[1, 1] - 1) > 1e-3 or int(t[1]) not in ci:
            if ln.startswith("1 "): keep.append(ln)
            continue
        w, d = v.part.footprint; h = v.part.height_units
        if abs(M[0, 2]) > 0.5: w, d = d, w
        x, y, z = map(float, t[2:5])
        items.append(dict(box=v.box, h=h, c=int(t[1]), x=int(round((x / 10 + 1 - w) / 2)), z=int(round((z / 10 + 1 - d) / 2)),
                          y=int(round(-y / 8 - h)), w=w, d=d, src=ln))
    n0 = len(items)

    def part_for(box, h, w, d, c):
        """(vpart, orient) of a catalogue part with this box, in colour c, either way round."""
        for key in ((box, h, w, d), (box, h, d, w)):
            g = boxes.get(key)
            if g and c in g[0].colours: return g
        return None

    def close(c1, c2): return np.linalg.norm(lab[ci[c1]] - lab[ci[c2]]) <= a.colour_tol
    merges = {"side by side": 0, "stacked plates": 0}; changed = True
    while changed:
        changed = False; layer = {}
        for k, it in enumerate(items): layer.setdefault((it["box"], it["h"], it["y"]), []).append(k)
        used, new = set(), []
        for ks in layer.values():
            for i in ks:
                if i in used: continue
                A = items[i]
                for j in ks:
                    if j == i or j in used or not close(A["c"], items[j]["c"]): continue
                    B = items[j]; x0, z0 = min(A["x"], B["x"]), min(A["z"], B["z"])
                    x1, z1 = max(A["x"] + A["w"], B["x"] + B["w"]), max(A["z"] + A["d"], B["z"] + B["d"])
                    if (x1 - x0) * (z1 - z0) != A["w"] * A["d"] + B["w"] * B["d"]: continue
                    c = A["c"] if A["w"] * A["d"] >= B["w"] * B["d"] else B["c"]
                    if part_for(A["box"], A["h"], x1 - x0, z1 - z0, c):
                        new.append(dict(A, x=x0, z=z0, w=x1 - x0, d=z1 - z0, c=c, src=None)); used |= {i, j}
                        merges["side by side"] += 1; changed = True; break
        items = [it for k, it in enumerate(items) if k not in used] + new
        col = {}
        for k, it in enumerate(items):
            if it["box"] == "plate" and it["h"] == 1: col.setdefault((it["x"], it["z"], it["w"], it["d"]), []).append(k)
        used, new = set(), []
        for ks in col.values():
            ks = sorted(ks, key=lambda k: items[k]["y"])
            for s in range(len(ks) - 2):
                trio = [items[k] for k in ks[s:s + 3]]
                if any(k in used for k in ks[s:s + 3]) or [t["y"] - trio[0]["y"] for t in trio] != [0, 1, 2]: continue
                c = trio[0]["c"]
                if all(close(c, t["c"]) for t in trio) and part_for("brick", 3, trio[0]["w"], trio[0]["d"], c):
                    new.append(dict(trio[0], box="brick", h=3, src=None)); used |= set(ks[s:s + 3])
                    merges["stacked plates"] += 1; changed = True
        items = [it for k, it in enumerate(items) if k not in used] + new
    out = list(keep)
    for it in items:
        if it["src"] is not None: out.append(it["src"]); continue
        vp, o = part_for(it["box"], it["h"], it["w"], it["d"], it["c"])
        W, D = o.size[1], o.size[2]
        rot = o.rot if (W, D) == (it["w"], it["d"]) else next(q.rot for q in vp.orients if (q.size[1], q.size[2]) == (it["w"], it["d"]))
        out.append(placement_to_line(Placement(vp.part, ci[it["c"]], it["x"], it["z"], it["y"], it["w"], it["d"], rot)))
    a.out.write_text("0 merged by tools/merge_parts.py\n" + "\n".join(out) + "\n")
    print(f"{a.ldr.name}: {n0} plain parts -> {len(items)}  ({merges})  total parts {len(out)}  -> {a.out}")


if __name__ == "__main__":
    main()
