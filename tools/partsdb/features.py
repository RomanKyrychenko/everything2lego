"""Per-model neighbourhood features of the official LDraw models, for the
statistics a generator matches: every part as a token (kind + size), every
pair of touching parts as a pattern (token A, token B, where B is: up,
down or level), and the kind of model (building, vehicle, figure, other)
read from what its parts are used for in official models.

    python tools/partsdb/features.py        # -> $LEGO_DATA/analysis/features.pkl
    python tools/partsdb/features.py paths  # adds each model's paths (needs the first)

Tokens are coarse on purpose: exact parts at exact offsets are too sparse
to compare models by (most of a real model's exact patterns occur nowhere
else); kind + size patterns are dense enough to have a distribution.
"""
from __future__ import annotations

import os
import pickle
import re
import sqlite3
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "set_study"))
from build import DATA, DB, OMR, _ld_to_part  # noqa: E402
from neighbours import held_out, model_parts, neighbour_pairs, part_classes  # noqa: E402

sys.path.insert(0, str(HERE.parents[1] / "src"))
from everything2lego.style import middle_weight, token_degree  # noqa: E402

OUT = DATA / "analysis" / "features.pkl"

KIND = {"Plates": "plate", "Plates Special": "plateS", "Bricks": "brick", "Bricks Special": "brickS",
        "Tiles": "tile", "Tiles Special": "tileS", "Bricks Curved": "curve", "Tiles Round and Curved": "tileR",
        "Plates Round Curved and Dishes": "plateR", "Bricks Round and Cones": "brickR",
        "Windows and Doors": "window", "Panels": "panel", "Plates Wedged": "wedgeP",
        "Bricks Wedged": "wedgeB", "Hinges, Arms and Turntables": "hinge", "Plants & Trees": "plant",
        "Bars, Ladders and Fences": "bar", "Wheels and Tyres": "wheel", "Technic Beams": "tbeam",
        "Technic Bricks": "tbrick", "Technic Pins": "tpin", "Technic Axles": "taxle",
        "Technic Connectors": "tconn"}
BUILDING = {"roof", "wall", "window", "door", "floor", "chimney", "stairs", "tower", "fence", "base"}
VEHICLE = {"wheel", "vehicle_body", "cockpit", "wing"}
FIGURE = {"figure"}


def token(category: str, name: str, w: float, d: float, h: float) -> str:
    """kind + footprint + height: 'plate1x2h1', 'slope45-2x2h3', 'brick1x1h3'."""
    kind = KIND.get(category, "other")
    if category == "Bricks Sloped":
        m = re.search(r"(\d+)°", name or "")
        inv = "inv" if "Inverted" in (name or "") else ""
        kind = f"slope{inv}{m.group(1) if m else ''}"
    a, b = sorted((int(round(w)), int(round(d))), reverse=True)
    a, b = min(a, 8), min(b, 8)                         # 8+ studs is "large"
    return f"{kind}{a}x{b}h{min(int(round(h)), 9)}"


def vertical(A, B) -> str:
    """Where B is relative to A: on top, underneath, or level (beside)."""
    if B[2][1] >= A[3][1] - 0.1:
        return "up"
    if B[3][1] <= A[2][1] + 0.1:
        return "down"
    return "level"


def shape_descriptor(extent, volume: float):
    """A model's overall shape, comparable between an official model and a
    target: log width, depth (studs; width >= depth), log height (studs, a
    stud is 2.5 plates), and how full its bounding box is."""
    import numpy as np
    w, d = sorted((float(extent[0]), float(extent[2])), reverse=True)
    h = float(extent[1]) / 2.5
    fill = volume / max(1e-6, w * d * h * 2.5)
    return np.array([np.log(w), np.log(d), np.log(max(h, 0.4)), min(1.0, fill)])


_CTX: dict = {}


def _init():
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    _CTX["ldmap"] = _ld_to_part(con)
    _CTX["classes"] = part_classes(con)
    g = pd.read_sql("""SELECT p.part_num, p.category, p.name, g.w_studs, g.d_studs, g.h_plates
                       FROM part p JOIN (SELECT part_num, w_studs, d_studs, h_plates FROM geometry
                                         GROUP BY part_num) g ON g.part_num = p.part_num""", con)
    _CTX["token"] = {pn: token(c, n, w or 0, d or 0, h or 0) for pn, c, n, w, d, h in g.values}
    roles = pd.read_sql("SELECT part_num, role, share FROM part_role", con)
    _CTX["roles"] = {pn: dict(zip(gr.role, gr.share)) for pn, gr in roles.groupby("part_num")}


def role_mass(part_nums: list[str]) -> Counter:
    """The model's parts' role shares (official models' uses), summed per
    kind of model: building, vehicle, figure, other."""
    s = Counter()
    for pn in part_nums:
        for r, sh in _CTX["roles"].get(pn, {}).items():
            s["building" if r in BUILDING else "vehicle" if r in VEHICLE else
              "figure" if r in FIGURE else "other"] += sh
    return s


def assign_kinds(rows: list[dict], min_lift: float = 1.2) -> None:
    """Each model's kind: the one whose share of its role mass most exceeds
    that kind's share over all models (lift), when it does by `min_lift`;
    else "other".  Common roles (every model has some vehicle-ish parts)
    don't decide by being common."""
    tot = Counter()
    for r in rows:
        n = sum(r["roles"].values()) or 1
        for k, v in r["roles"].items():
            tot[k] += v / n
    base = {k: v / len(rows) for k, v in tot.items()}
    for r in rows:
        n = sum(r["roles"].values())
        if not n:
            r["kind"] = "other"
            continue
        lifts = {k: (v / n) / base[k] for k, v in r["roles"].items() if base.get(k)}
        k, lift = max(lifts.items(), key=lambda kv: kv[1])
        r["kind"] = k if lift >= min_lift else "other"


def features(f: Path):
    try:
        items = model_parts(f, _CTX["ldmap"], _CTX["classes"])
    except Exception:
        return None
    if len(items) < 20:
        return None
    toks = [_CTX["token"].get(it[0], "other0x0h0") for it in items]
    import numpy as np
    lo = np.min([it[2] for it in items], axis=0)
    hi = np.max([it[3] for it in items], axis=0)
    ext = np.maximum(hi - lo, 1e-3)                      # studs, plates, studs
    vol = sum(float(np.prod(np.maximum(it[3] - it[2], 0))) for it in items)
    pats = Counter()
    for i, j in neighbour_pairs(items):
        pats[(toks[i], toks[j], vertical(items[i], items[j]))] += 1
    return {"model": f.stem, "held_out": held_out(f.stem), "roles": role_mass([it[0] for it in items]),
            "n_parts": len(items), "tokens": Counter(toks), "patterns": pats,
            "shape": shape_descriptor(ext, vol)}


def paths(items, toks, adj, degree: dict[str, float]) -> Counter:
    """The second scale of the patterns: walks a - b - c through a shared
    neighbour b (a != c), as (token a, token c, where c is from a), each
    weighted by b's middle_weight.  Pairs are what touches; paths are what
    sits around a part's neighbours — the larger receptive field, as style
    transfer matches feature correlations at more than one layer."""
    import numpy as np
    vocab = sorted(set(toks))
    index = {t: i for i, t in enumerate(vocab)}
    tid = np.array([index[t] for t in toks])
    lo = np.array([it[2][1] for it in items])
    hi = np.array([it[3][1] for it in items])
    V = len(vocab)
    acc = np.zeros(V * V * 3)
    for b, nb in adj.items():
        if len(nb) < 2:
            continue
        nb = np.fromiter(nb, int)
        A, Cc = nb[:, None], nb[None, :]
        # where c is from a (vertical): 0 up, 1 down, 2 level
        d = np.where(lo[Cc] >= hi[A] - 0.1, 0, np.where(hi[Cc] <= lo[A] + 0.1, 1, 2))
        key = (tid[A] * V + tid[Cc]) * 3 + d
        off = ~np.eye(len(nb), dtype=bool)
        acc += np.bincount(key[off], minlength=V * V * 3) * middle_weight(degree, toks[b])
    out = Counter()
    names = ("up", "down", "level")
    for k in np.flatnonzero(acc):
        a, rest = divmod(int(k), V * 3)
        c, d = divmod(rest, 3)
        out[(vocab[a], vocab[c], names[d])] = float(acc[k])
    return out


def model_paths(f: Path):
    try:
        items = model_parts(f, _CTX["ldmap"], _CTX["classes"])
    except Exception:
        return None
    toks = [_CTX["token"].get(it[0], "other0x0h0") for it in items]
    adj: dict[int, set] = {}
    for i, j in neighbour_pairs(items):
        adj.setdefault(i, set()).add(j)
    return f.stem, paths(items, toks, adj, _CTX["degree"])


def _init_paths(degree):
    _init()
    _CTX["degree"] = degree


def add_paths() -> None:
    """Second stage: every model's paths, weighted by the token degrees of
    the training models (the sampler computes the same weights)."""
    rows = pickle.load(open(OUT, "rb"))
    degree = token_degree([r for r in rows if not r["held_out"]])
    by_model = {r["model"]: r for r in rows}
    files = [f for f in sorted(OMR.glob("*.mpd")) + sorted(OMR.glob("*.ldr")) if f.stem in by_model]
    with Pool(max(1, (os.cpu_count() or 2) - 1), initializer=_init_paths, initargs=(degree,),
              maxtasksperchild=50) as pool:
        for k, got in enumerate(pool.imap_unordered(model_paths, files, chunksize=2)):
            if got:
                by_model[got[0]]["paths"] = got[1]
            if k % 200 == 0:
                print(f"  paths {k}/{len(files)}", flush=True)
    save(rows)
    print(f"paths: {sum('paths' in r for r in rows)} models -> {OUT}")


def save(rows: list[dict]) -> None:
    """Write through a temporary file: an interrupted write (Ctrl-C, the
    external disk gone) must not leave a broken file that took 40 minutes."""
    tmp = OUT.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(rows, fh)
    os.replace(tmp, OUT)


def main() -> None:
    files = sorted(OMR.glob("*.mpd")) + sorted(OMR.glob("*.ldr"))
    rows = []
    with Pool(max(1, (os.cpu_count() or 2) - 1), initializer=_init, maxtasksperchild=50) as pool:
        for k, r in enumerate(pool.imap_unordered(features, files, chunksize=2)):
            if r:
                rows.append(r)
            if k % 200 == 0:
                print(f"  {k}/{len(files)}", flush=True)
    assign_kinds(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    save(rows)
    kinds = Counter(r["kind"] for r in rows)
    print(f"features: {len(rows)} models -> {OUT}; kinds {dict(kinds)}")


if __name__ == "__main__":
    add_paths() if sys.argv[1:] == ["paths"] else main()
