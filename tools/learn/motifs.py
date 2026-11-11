"""Building patterns: groups of 2-3 touching parts that recur, at the same relative places, in real building sets.

Each real part is mapped onto the sampler's grid as one of its options (part + quarter turn) at an integer cell, by
inverting the sampler's own LDraw writer (io_ldraw.placement_to_line): an option written at the origin gives its
LDraw rotation and offset, so a real part with that rotation sits at (its offset - that offset) / cell size.  Parts
off the grid (half-stud jumpers, sideways, hinged) are left out.  Two parts touch when one sits on the other with
overlapping footprints, or they share a side face at overlapping heights; a pattern is a part with one or two of the
parts it touches, kept by its parts' relative cells and turns, counted by how many sets have it.

Bigger patterns grow level by level (Apriori, as in frequent-subgraph mining): a k-part group is a frequent (k-1)-part
group's instance plus a part one of its members touches, and only the groups in at least --min-sets sets are grown
further; from four parts up a group of one part alike (a tiled floor, a column of 1x1 bricks) is not kept -- the
sampler's merge / retile moves make those.  The parts' cells fix the geometry, so the relative-cell key is canonical and no graph matching is needed.

    .venv-arm/bin/python tools/learn/motifs.py [--n 150 --min-sets 8 --max-size 6 --top 1500]
        -> src/everything2lego/data/motifs.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE.parent / "set_study"), str(REPO / "src")]
DATA = Path("/Volumes/PRO-G40/lego_data")
OUT = REPO / "src/everything2lego/data/motifs.json"


def option_poses(kind: str = "building"):
    """(ldraw, rotation row-major rounded) -> [(rot, (h, w, d), origin LDU)]: every way an option is written."""
    from everything2lego.io_ldraw import _ROT
    from everything2lego.pack import Placement
    from everything2lego.io_ldraw import placement_to_line
    from everything2lego.sampler import select_vocabulary
    poses = defaultdict(list)
    for v in select_vocabulary(kind, 100000):
        ups = [o for o in v.orients if o.snot is None and o.stack is None]
        if not ups:
            continue
        for r in (0, 90, 180, 270):
            # a turn the sampler does not keep (the part looks the same turned) maps onto a kept one of that footprint
            o0 = ups[0]
            foot = (o0.size[1], o0.size[2]) if (r - o0.rot) % 180 == 0 else (o0.size[2], o0.size[1])
            same = [o for o in ups if o.rot == r] or [o for o in ups if (o.size[1], o.size[2]) == foot]
            if not same:
                continue
            o = same[0]
            p = Placement(v.part, 0, 0, 0, 0, o.size[1], o.size[2], r)
            f = placement_to_line(p).split("\n")[0].split()
            t = np.array([float(f[2]), float(f[3]), float(f[4])])
            R = tuple(int(round(float(x))) for x in f[5:14])
            poses[(v.part.ldraw.lower(), R)].append((o.rot, tuple(o.size), t))
    return poses


def grid_items(path: Path, poses) -> list[tuple]:
    """A set's parts on the grid: (ldraw, rot, (y, x, z), (h, w, d))."""
    from ldraw_read import flatten
    raw = []
    for p in flatten(path):
        R = tuple(int(round(x)) for x in p.M[:3, :3].reshape(-1))
        cand = poses.get((p.name.lower(), R))
        if not cand:
            continue
        rot, size, t0 = cand[0]
        t = p.M[:3, 3]
        raw.append((p.name.lower(), rot, np.array([(t0[1] - t[1]) / 8.0, (t[0] - t0[0]) / 20.0, (t[2] - t0[2]) / 20.0]), size))
    if not raw:
        return []
    pos = np.array([r[2] for r in raw])
    shift = []
    for a in range(3):                                   # the set's own grid: the commonest fractional offset per axis
        fr = np.round((pos[:, a] % 1.0) * 20) / 20 % 1.0
        shift.append(Counter(fr.tolist()).most_common(1)[0][0])
    pos = pos - np.array(shift)
    on = np.all(np.abs(pos - np.round(pos)) < 0.06, axis=1)
    return [(r[0], r[1], tuple(int(v) for v in np.round(q)), r[3]) for r, q, ok in zip(raw, pos, on) if ok]


def touching(items) -> list[set]:
    """Per item, the items it touches (on / under it with overlapping footprints, or face to face beside it)."""
    by_cell = defaultdict(list)
    for i, (_, _, (y, x, z), (h, w, d)) in enumerate(items):
        for yy in range(y, y + h):
            for xx in range(x, x + w):
                for zz in range(z, z + d):
                    by_cell[(yy, xx, zz)].append(i)
    nb = [set() for _ in items]
    for i, (_, _, (y, x, z), (h, w, d)) in enumerate(items):
        probe = [(y + h, xx, zz) for xx in range(x, x + w) for zz in range(z, z + d)]            # above
        probe += [(yy, x - 1, zz) for yy in range(y, y + h) for zz in range(z, z + d)]          # beside
        probe += [(yy, x + w, zz) for yy in range(y, y + h) for zz in range(z, z + d)]
        probe += [(yy, xx, z - 1) for yy in range(y, y + h) for xx in range(x, x + w)]
        probe += [(yy, xx, z + d) for yy in range(y, y + h) for xx in range(x, x + w)]
        for c in probe:
            for j in by_cell.get(c, ()):
                if j != i:
                    nb[i].add(j); nb[j].add(i)
    return nb


def key(items, members) -> tuple:
    """The group by its parts' turns and cells relative to the group's lowest corner."""
    y0 = min(items[m][2][0] for m in members); x0 = min(items[m][2][1] for m in members); z0 = min(items[m][2][2] for m in members)
    return tuple(sorted((items[m][0], items[m][1], items[m][2][0] - y0, items[m][2][1] - x0, items[m][2][2] - z0) for m in members))


def overlaps(items, members) -> bool:
    cells = set()
    for m in members:
        (y, x, z), (h, w, d) = items[m][2], items[m][3]
        c = {(yy, xx, zz) for yy in range(y, y + h) for xx in range(x, x + w) for zz in range(z, z + d)}
        if cells & c:
            return True
        cells |= c
    return False


def mine(sets: list, min_sets: int, max_size: int, cap: int = 40) -> dict:
    """{size: Counter(key -> sets)} of groups of 2..max_size touching parts in at least `min_sets` of `sets`
    ((items, touching, on): only parts with on[i] count).  At most `cap` instances of a group per set are grown."""
    import random
    rnd = random.Random(0)
    out = {}
    inst = []                                            # per set: key -> member tuples
    for items, nb, on in sets:
        g = defaultdict(set)
        for i in range(len(items)):
            if on[i]:
                for j in nb[i]:
                    if j > i and on[j]:
                        g[key(items, (i, j))].add((i, j))
        inst.append(g)
    for k in range(2, max_size + 1):
        n = Counter(kk for g in inst for kk in g
                    if k < 4 or len({p[:2] for p in kk}) > 1)       # 4+ alike parts: a plain floor / column, retile's job
        keep = {kk for kk, c in n.items() if c >= min_sets}
        out[k] = Counter({kk: n[kk] for kk in keep})
        print(f"  size {k}: {len(n)} groups, {len(keep)} in >= {min_sets} sets", flush=True)
        if k == max_size or not keep:
            break
        nxt = []
        for (items, nb, on), g in zip(sets, inst):
            h = defaultdict(set)
            for kk in keep & g.keys():
                ms = list(g[kk])
                for m in (rnd.sample(ms, cap) if len(ms) > cap else ms):
                    for j in {j for i in m for j in nb[i]} - set(m):
                        if on[j]:
                            mm = tuple(sorted(m + (j,)))
                            if not overlaps(items, mm):
                                h[key(items, mm)].add(mm)
            nxt.append(h)
        inst = nxt
    return out


def load_sets(kind: str, n: int, min_parts: int, poses) -> list:
    """Up to n real sets of this kind with at least min_parts parts on the grid: (file, items, touching)."""
    kinds = {r["id"]: r["kind"] for r in csv.DictReader(open(DATA / "analysis" / "set_kinds.csv"))}
    files = [f for f in sorted((DATA / "omr" / "files").glob("*.mpd")) if kinds.get(f.stem.split("_")[0]) == kind]
    out = []
    for f in files:
        try:
            items = grid_items(f, poses)
        except Exception:                                                     # noqa: BLE001
            continue
        if len(items) >= min_parts:
            out.append((f, items, touching(items)))
        if len(out) >= n:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="building"); ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--min-parts", type=int, default=200); ap.add_argument("--min-sets", type=int, default=8)
    ap.add_argument("--max-size", type=int, default=6); ap.add_argument("--top", type=int, default=1500, help="per size")
    a = ap.parse_args()
    poses = option_poses(a.kind)
    sets = load_sets(a.kind, a.n, a.min_parts, poses)
    found = mine([(items, nb, [True] * len(items)) for _, items, nb in sets], a.min_sets, a.max_size)
    keep = [(kk, c) for k in sorted(found) for kk, c in found[k].most_common(a.top)]
    old = json.loads(OUT.read_text()) if OUT.exists() else {}
    typed = [m for m in old.get("motifs", []) if m.get("type", 0) != 0]          # roof / flat-top patterns (roof_mix.py)
    OUT.write_text(json.dumps({**{k: v for k, v in old.items() if k != "motifs"},
                               "source": "tools/learn/motifs.py", "kind": a.kind, "sets": len(sets),
                               "motifs": [{"parts": [list(p) for p in kk], "sets": c} for kk, c in keep] + typed}))
    print(f"{len(sets)} {a.kind} sets, {np.mean([len(i) for _, i, _ in sets]):.0f} grid parts each; "
          f"{len(keep)} patterns kept -> {OUT}")
    for k in sorted(found):
        for kk, c in found[k].most_common(3):
            print(f"  {c:4d} sets  " + "  ".join(f"{p[0]}@{p[1]}+{p[2:]}" for p in kk))

if __name__ == "__main__":
    main()
