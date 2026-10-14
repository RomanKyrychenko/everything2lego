"""Mine 3-D neighbourhood statistics from the official LDraw models: for
every part, which parts sit around it, where (in the part's own frame) and
at what relative rotation — the "market basket" of a part's surroundings,
for any kind of model.

    python tools/partsdb/neighbours.py            # writes tables into the parts DB

Each part is a token at two levels: its Rebrickable part number, and its
shape class "<category>:<w>x<d>x<h>" (studs, studs, plates; w >= d) — so a
rare part borrows the statistics of its class.  Two parts are neighbours
when their boxes touch or overlap on the stud x plate grid (a 1-cell
dilation).  The relation of B to A is

    (dx, dy, dz)  B's box centre minus A's, in half studs / half plates,
                  rotated into A's frame (A's yaw), y up
    rot           B's yaw relative to A's (0-3) and B's up axis in A's
                  frame (U up, D down, S sideways)

Tables (in $PARTSDB):
    part_class(part_num, class)
    nbr_class(a, b, dx, dy, dz, rot, n)     counts over the training models
    nbr_part(a, b, dx, dy, dz, rot, n)      the same for exact parts (n >= 3)
    nbr_total(level, token, n)              how often each token occurs as A
    nbr_split(model, held_out)              10% of models held out (for scoring)
"""
from __future__ import annotations

import hashlib
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "set_study"))
from build import DB, OMR, _ld_to_part  # noqa: E402

STUD, PLATE = 20.0, 8.0
HELD_OUT = 0.1


def held_out(model: str) -> bool:
    """A fixed 10% of models, by name, kept out of the statistics."""
    return int(hashlib.md5(model.encode()).hexdigest(), 16) % 1000 < HELD_OUT * 1000


def part_classes(con: sqlite3.Connection) -> dict[str, str]:
    """Rebrickable part -> shape class, from its category and size."""
    g = pd.read_sql("""SELECT p.part_num, p.category, g.w_studs, g.d_studs, g.h_plates
                       FROM part p JOIN (SELECT part_num, w_studs, d_studs, h_plates FROM geometry
                                         GROUP BY part_num) g ON g.part_num = p.part_num""", con)
    out = {}
    for pn, cat, w, d, h in g.values:
        a, b = sorted((int(round(w or 0)), int(round(d or 0))), reverse=True)
        out[pn] = f"{cat}:{a}x{b}x{int(round(h or 0))}"
    return out


def _frame(M: np.ndarray) -> tuple[int, str]:
    """(yaw 0-3, up 'U'/'D'/'S') of a part placement; y down in LDraw."""
    R = M[:3, :3] / (np.linalg.norm(M[:3, :3], axis=0) + 1e-9)
    up = R @ np.array([0.0, -1.0, 0.0])                 # the part's own up, in world (y down)
    face = "U" if up[1] < -0.9 else "D" if up[1] > 0.9 else "S"
    xa = R @ np.array([1.0, 0.0, 0.0])
    yaw = int(round(np.degrees(np.arctan2(-xa[2], xa[0])) / 90.0)) % 4
    return yaw, face


def _rot2(dx: float, dz: float, yaw: int) -> tuple[float, float]:
    """(dx, dz) seen from a frame turned `yaw` quarter turns."""
    for _ in range(yaw % 4):
        dx, dz = dz, -dx
    return dx, dz


def model_parts(f: Path, ldmap: dict, classes: dict):
    """[(part_num, class, lo(3 grid), hi(3 grid), yaw, face)] of one model,
    in its own frame: x, z in studs, y in plates, y up."""
    from ldraw_read import flatten, part_bbox
    from patterns import to_model_frame
    parts = [p for p in flatten(f) if part_bbox(p.name) is not None]
    if not parts:
        return []
    parts = to_model_frame(parts)
    out = []
    for p in parts:
        lo, hi = part_bbox(p.name)
        lo = lo.copy()
        if hi[1] > 0:
            lo[1] = max(lo[1], 0.0)                     # studs above the body don't count
        c = np.array([[x, y, z, 1] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        w = (p.M @ c.T).T[:, :3]
        a, b = w.min(0), w.max(0)
        glo = np.array([a[0] / STUD, -b[1] / PLATE, a[2] / STUD])
        ghi = np.array([b[0] / STUD, -a[1] / PLATE, b[2] / STUD])
        pn = ldmap.get(p.name, p.name[:-4])
        yaw, face = _frame(p.M)
        out.append((pn, classes.get(pn, "?"), glo, ghi, yaw, face))
    return out


def neighbour_pairs(items):
    """Directed (i, j) pairs of parts whose grid boxes touch or overlap."""
    cells = defaultdict(list)
    boxes = []
    for i, (_, _, lo, hi, _, _) in enumerate(items):
        i0 = np.floor(lo + 0.25).astype(int)
        i1 = np.maximum(np.ceil(hi - 0.25).astype(int), i0 + 1)
        boxes.append((i0, i1))
        if np.prod(i1 - i0) > 4096:                     # a baseplate: index its top only
            i0 = np.array([i0[0], i1[1] - 1, i0[2]])
        for x in range(i0[0], i1[0]):
            for y in range(i0[1], i1[1]):
                for z in range(i0[2], i1[2]):
                    cells[(x, y, z)].append(i)
    pairs = set()
    for i, (i0, i1) in enumerate(boxes):
        found = set()
        for x in range(i0[0] - 1, i1[0] + 1):
            for z in range(i0[2] - 1, i1[2] + 1):
                for y in (i0[1] - 1, i1[1]):            # the layers just below and above
                    found.update(cells.get((x, y, z), ()))
                if x in (i0[0] - 1, i1[0]) or z in (i0[2] - 1, i1[2]):
                    for y in range(i0[1], i1[1]):       # the ring beside it
                        found.update(cells.get((x, y, z), ()))
        found.discard(i)
        pairs.update((i, j) for j in found)
    return pairs


def relation(A, B) -> tuple[int, int, int, str]:
    """(dx, dy, dz, rot) of B relative to A, half-units, A's frame."""
    ca = (A[2] + A[3]) / 2
    cb = (B[2] + B[3]) / 2
    d = cb - ca
    dx, dz = _rot2(d[0], d[2], A[4])
    rot = f"{(B[4] - A[4]) % 4}{B[5] if A[5] == 'U' else A[5] + B[5]}"
    return int(round(2 * dx)), int(round(2 * d[1])), int(round(2 * dz)), rot


def main() -> None:
    from ldraw_read import save_cache
    con = sqlite3.connect(DB, timeout=900)
    ldmap = _ld_to_part(con)
    classes = part_classes(con)
    pd.DataFrame(list(classes.items()), columns=["part_num", "class"]) \
        .to_sql("part_class", con, if_exists="replace", index=False)
    by_class: Counter = Counter()
    by_part: Counter = Counter()
    tot_class: Counter = Counter()
    tot_part: Counter = Counter()
    split = []
    files = sorted(OMR.glob("*.mpd")) + sorted(OMR.glob("*.ldr"))
    for k, f in enumerate(files):
        ho = held_out(f.stem)
        split.append((f.stem, ho))
        if ho:
            continue
        try:
            items = model_parts(f, ldmap, classes)
        except Exception:
            continue
        for it in items:
            tot_class[it[1]] += 1
            tot_part[it[0]] += 1
        for i, j in neighbour_pairs(items):
            A, B = items[i], items[j]
            rel = relation(A, B)
            by_class[(A[1], B[1]) + rel] += 1
            by_part[(A[0], B[0]) + rel] += 1
        if k % 100 == 0:
            print(f"  {k}/{len(files)}: {len(by_class)} class relations", flush=True)
            save_cache()
    save_cache()
    pd.DataFrame([k + (n,) for k, n in by_class.items() if n >= 2],
                 columns=["a", "b", "dx", "dy", "dz", "rot", "n"]).to_sql("nbr_class", con, if_exists="replace", index=False)
    pd.DataFrame([k + (n,) for k, n in by_part.items() if n >= 3],
                 columns=["a", "b", "dx", "dy", "dz", "rot", "n"]).to_sql("nbr_part", con, if_exists="replace", index=False)
    pd.DataFrame([("class", t, n) for t, n in tot_class.items()] + [("part", t, n) for t, n in tot_part.items()],
                 columns=["level", "token", "n"]).to_sql("nbr_total", con, if_exists="replace", index=False)
    pd.DataFrame(split, columns=["model", "held_out"]).to_sql("nbr_split", con, if_exists="replace", index=False)
    con.execute("CREATE INDEX IF NOT EXISTS ix_nc ON nbr_class(a)")
    con.execute("CREATE INDEX IF NOT EXISTS ix_np ON nbr_part(a)")
    con.commit()
    print(f"neighbours: {sum(by_class.values())} pair observations, {len(by_class)} class relations, "
          f"{len(by_part)} part relations; {sum(h for _, h in split)} models held out")


if __name__ == "__main__":
    main()
