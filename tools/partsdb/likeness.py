"""How much a model's part neighbourhoods look like the official models'.

For every pair of neighbouring parts (A, B) in a model — B at a relation
(position, rotation) in A's frame, see neighbours.py — the pointwise
mutual information

    PMI = log P(B, rel | A) - log P(B, rel)

from the official models' neighbourhood counts: how much more often B sits
there next to an A than B sits there next to anything.  Exact parts are
used where A is common enough (`min_part`), else A's and B's shape classes.
P(B, rel | A) is smoothed toward P(B, rel).  The model's likeness is the
mean PMI over its neighbour pairs; pairs whose A the statistics never saw
count 0 (no evidence either way).

    python tools/partsdb/likeness.py model.ldr [more.ldr ...]
    python tools/partsdb/likeness.py --validate     # held-out official models vs shuffled vs ours
"""
from __future__ import annotations

import math
import random
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "set_study"))
from build import DB, OMR, _ld_to_part  # noqa: E402
from neighbours import model_parts, neighbour_pairs, part_classes, relation  # noqa: E402


class Likeness:
    def __init__(self, con: sqlite3.Connection, *, alpha: float = 20.0, min_part: int = 200):
        self.alpha, self.min_part = alpha, min_part
        self.ldmap = _ld_to_part(con)
        self.classes = part_classes(con)
        self.levels = {}
        for level, table in (("class", "nbr_class"), ("part", "nbr_part")):
            df = pd.read_sql(f"SELECT a, b, dx, dy, dz, rot, n FROM {table}", con)
            joint = {(a, b, dx, dy, dz, r): n for a, b, dx, dy, dz, r, n in df.itertuples(index=False)}
            na = df.groupby("a").n.sum().to_dict()
            nb = df.groupby(["b", "dx", "dy", "dz", "rot"]).n.sum()
            nb = {k: v for k, v in nb.items()}
            self.levels[level] = (joint, na, nb, float(df.n.sum()), len(nb))

    def pmi(self, level: str, a: str, b: str, rel: tuple) -> float | None:
        joint, na, nb, total, vocab = self.levels[level]
        n_a = na.get(a, 0)
        if n_a == 0:
            return None
        p0 = (nb.get((b,) + rel, 0) + 1.0) / (total + vocab)
        p = (joint.get((a, b) + rel, 0) + self.alpha * p0) / (n_a + self.alpha)
        return math.log(p) - math.log(p0)

    def pair(self, A, B) -> float | None:
        rel = relation(A, B)
        if self.levels["part"][1].get(A[0], 0) >= self.min_part:
            v = self.pmi("part", A[0], B[0], rel)
            if v is not None:
                return v
        return self.pmi("class", A[1], B[1], rel)

    def score_items(self, items) -> dict:
        vals, unknown = [], 0
        for i, j in neighbour_pairs(items):
            v = self.pair(items[i], items[j])
            if v is None:
                unknown += 1
            else:
                vals.append(v)
        n = len(vals) + unknown
        return {"likeness": sum(vals) / max(1, n), "pairs": n,
                "unknown_pct": 100.0 * unknown / max(1, n)}

    def score_file(self, path: Path) -> dict:
        return self.score_items(model_parts(path, self.ldmap, self.classes))


def shuffled(items, seed: int = 0):
    """The same boxes, the parts dealt among boxes of the same size at
    random: geometry kept, arrangement destroyed."""
    rnd = random.Random(seed)
    groups = defaultdict(list)
    for i, it in enumerate(items):
        groups[tuple((it[3] - it[2]).round(2))].append(i)
    out = list(items)
    for idx in groups.values():
        toks = [(items[i][0], items[i][1], items[i][4], items[i][5]) for i in idx]
        rnd.shuffle(toks)
        for i, (pn, cl, yaw, face) in zip(idx, toks):
            out[i] = (pn, cl, items[i][2], items[i][3], yaw, face)
    return out


def validate(lk: Likeness, con: sqlite3.Connection, ours: list[Path]) -> None:
    held = pd.read_sql("SELECT model FROM nbr_split WHERE held_out", con).model.tolist()
    rows = []
    for m in held:
        f = OMR / f"{m}.mpd"
        if not f.exists():
            f = OMR / f"{m}.ldr"
        try:
            items = model_parts(f, lk.ldmap, lk.classes)
        except Exception:
            continue
        if len(items) < 50:
            continue
        a = lk.score_items(items)["likeness"]
        b = lk.score_items(shuffled(items))["likeness"]
        rows.append((m, len(items), a, b))
    df = pd.DataFrame(rows, columns=["model", "parts", "official", "shuffled"])
    print(f"held-out official models: {len(df)}")
    print(f"  official  mean {df.official.mean():.3f}  median {df.official.median():.3f}")
    print(f"  shuffled  mean {df.shuffled.mean():.3f}  median {df.shuffled.median():.3f}")
    print(f"  official > shuffled in {100 * (df.official > df.shuffled).mean():.0f}% of models")
    for p in ours:
        s = lk.score_file(p)
        print(f"  ours {p}: {s['likeness']:.3f} over {s['pairs']} pairs ({s['unknown_pct']:.1f}% unknown)")


if __name__ == "__main__":
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    lk = Likeness(con)
    args = sys.argv[1:]
    if args and args[0] == "--validate":
        validate(lk, con, [Path(a) for a in args[1:]])
    else:
        for a in args:
            print(a, lk.score_file(Path(a)))
