"""Per-model neighbourhood features of the official LDraw models, for the
statistics a generator matches: every part as a token (kind + size), every
pair of touching parts as a pattern (token A, token B, where B is: up,
down or level), and the kind of model (building, vehicle, figure, other)
read from what its parts are used for in official models.

    python tools/partsdb/features.py        # -> $LEGO_DATA/analysis/features.pkl
    python tools/partsdb/features.py paths  # adds each model's paths (needs the first)
    python tools/partsdb/features.py paths4 # adds each model's 4-walks (needs paths)

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
from everything2lego.style import kind_of, middle_weight, token_degree  # noqa: E402

PATHS4_BUCKETS = 8                                                       # middle-kind pair hash buckets

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


def paths4_bucket(kind_a: str, kind_b: str) -> int:
    """8-bucket hash of a sorted pair of middle kinds.  Keeps paths4 keys dense:
    without this the key space would be O(n_tok^4), with it it's O(n_tok^2 * 3 * B)
    like the paths term plus a factor B."""
    pair = (kind_a, kind_b) if kind_a <= kind_b else (kind_b, kind_a)
    return hash(pair) % PATHS4_BUCKETS


def paths4(items, toks, adj, degree: dict[str, float]) -> Counter:
    """4-walks a - b - c - d (a != c, d != b, a may equal d) through two
    successive adjacencies.  Keyed by (token_a, token_d, where_d_from_a:bucket)
    where bucket is an 8-way hash of the sorted pair of middle kinds
    (kind_of(b), kind_of(c)) folded into the direction string.  Weight per walk
    = middle_w(b) * middle_w(c) (product of inner-degree normalisations)."""
    import numpy as np
    lo = np.array([it[2][1] for it in items])
    hi = np.array([it[3][1] for it in items])
    kind_tok: dict[str, str] = {}
    def kind_of_tok(t):
        v = kind_tok.get(t)
        if v is None:
            v = kind_tok[t] = kind_of(t)
        return v
    # cache middle_weight per token
    mw_cache: dict[str, float] = {}
    def mw(t):
        v = mw_cache.get(t)
        if v is None:
            v = mw_cache[t] = middle_weight(degree, t)
        return v
    names = ("up", "down", "level")
    out: Counter = Counter()
    # iterate directed middle edges (b, c): each walk a-b-c-d counted once
    for b, nb_b in adj.items():
        tb = toks[b]
        wb = mw(tb)
        kb = kind_of_tok(tb)
        a_pool = list(nb_b)
        if not a_pool:
            continue
        hib = hi[b]; lob = lo[b]
        for c in nb_b:
            if c == b:
                continue
            nb_c = adj.get(c)
            if nb_c is None:
                continue
            tc = toks[c]
            w = wb * mw(tc)
            bucket = paths4_bucket(kb, kind_of_tok(tc))
            # emit over (a, d) with a != c and d != b
            for a in a_pool:
                if a == c:
                    continue
                ta = toks[a]
                hi_a = hi[a]; lo_a = lo[a]
                for d in nb_c:
                    if d == b:
                        continue
                    td = toks[d]
                    hi_d = hi[d]; lo_d = lo[d]
                    # vertical(a, d): d relative to a's grid y-range
                    if lo_d >= hi_a - 0.1:
                        direction = 0
                    elif hi_d <= lo_a + 0.1:
                        direction = 1
                    else:
                        direction = 2
                    out[(ta, td, f"{names[direction]}:{bucket}")] += w
    return out


def triangles(items, toks, adj, degree: dict[str, float]) -> Counter:
    """3-cliques: parts (i, j, k) that are pairwise adjacent.  Keyed by the
    sorted token triple (as a 3-element tuple folded into the Reference's
    expected 3-tuple schema: (token_lo, token_mid, "tri:token_hi")).  Weight
    per triangle = 1 / (middle_w(j) * middle_w(k)) — but simpler: unit weight,
    since triangles are rare and we count unique 3-cliques per model.

    Walked once per triangle via i < j < k ordering to avoid 6x overcount."""
    out: Counter = Counter()
    for i, nbi in adj.items():
        nbi_up = {j for j in nbi if j > i}
        for j in nbi_up:
            nbj = adj.get(j)
            if nbj is None:
                continue
            for k in nbi_up:
                if k > j and k in nbj:
                    a, b, c = sorted((toks[i], toks[j], toks[k]))
                    out[(a, b, f"tri:{c}")] += 1
    return out


def model_triangles(f: Path):
    try:
        items = model_parts(f, _CTX["ldmap"], _CTX["classes"])
    except Exception:
        return None
    toks = [_CTX["token"].get(it[0], "other0x0h0") for it in items]
    adj: dict[int, set] = {}
    for i, j in neighbour_pairs(items):
        adj.setdefault(i, set()).add(j)
    return f.stem, triangles(items, toks, adj, _CTX["degree"])


def add_triangles(limit: int | None = None, max_parts: int = 2000, workers: int = 4) -> None:
    """Precompute per-model triangles.  Same pool infrastructure as add_paths4."""
    print(f"loading {OUT}...", flush=True)
    rows = pickle.load(open(OUT, "rb"))
    print(f"loaded {len(rows)} rows; collecting degree + files", flush=True)
    degree = token_degree([r for r in rows if not r["held_out"]])
    by_model = {r["model"]: r for r in rows}
    files = [f for f in sorted(OMR.glob("*.mpd")) + sorted(OMR.glob("*.ldr")) if f.stem in by_model]
    files = [f for f in files if by_model[f.stem].get("n_parts", 0) <= max_parts
             and "triangles" not in by_model[f.stem]]
    if limit:
        files = files[:limit]
    print(f"{len(files)} files to process (max_parts={max_parts}, limit={limit}, workers={workers})", flush=True)
    import time
    t0 = time.time()
    with Pool(workers, initializer=_init_paths, initargs=(degree,), maxtasksperchild=20) as pool:
        for k, got in enumerate(pool.imap_unordered(model_triangles, files, chunksize=1)):
            if got:
                by_model[got[0]]["triangles"] = got[1]
            if (k + 1) % 50 == 0:
                elapsed = time.time() - t0
                rate = (k + 1) / max(1, elapsed)
                eta = (len(files) - k - 1) / max(0.1, rate)
                print(f"  triangles {k+1}/{len(files)}  ({elapsed:.0f}s, {rate:.2f}/s, ETA {eta:.0f}s)", flush=True)
            if (k + 1) % 500 == 0:
                save(rows)
                print(f"    [checkpoint saved]", flush=True)
    save(rows)
    print(f"triangles: {sum('triangles' in r for r in rows)} models -> {OUT}")


def model_paths4(f: Path):
    try:
        items = model_parts(f, _CTX["ldmap"], _CTX["classes"])
    except Exception:
        return None
    toks = [_CTX["token"].get(it[0], "other0x0h0") for it in items]
    adj: dict[int, set] = {}
    for i, j in neighbour_pairs(items):
        adj.setdefault(i, set()).add(j)
    return f.stem, paths4(items, toks, adj, _CTX["degree"])


def add_paths4(limit: int | None = None, max_parts: int = 2000, workers: int = 4) -> None:
    """Third stage (optional): every model's 4-walks, keyed by (token_a, token_d,
    where, mid_bucket).  Requires `paths` precomputed first (shares the degree
    weighting).  Store as `paths4` key in the per-model row.

    Uses a Pool with conservative settings (small worker count, chunksize=1,
    maxtasksperchild=20) to balance speed against the pool-crash sensitivity
    we saw on earlier runs.  Checkpoints to disk every 500 completed models.

    `limit`: if set, only process the first N models.
    `max_parts`: skip models with more than this many parts.  The 4-walk count
    scales as deg^3 per part, so baseplate-heavy mega-models dominate time
    without adding signal to the reference.
    `workers`: pool size.
    """
    print(f"loading {OUT}...", flush=True)
    rows = pickle.load(open(OUT, "rb"))
    print(f"loaded {len(rows)} rows; collecting degree + files", flush=True)
    degree = token_degree([r for r in rows if not r["held_out"]])
    by_model = {r["model"]: r for r in rows}
    files = [f for f in sorted(OMR.glob("*.mpd")) + sorted(OMR.glob("*.ldr")) if f.stem in by_model]
    files = [f for f in files if by_model[f.stem].get("n_parts", 0) <= max_parts
             and "paths4" not in by_model[f.stem]]                     # resume: skip already-done
    if limit:
        files = files[:limit]
    print(f"{len(files)} files to process (max_parts={max_parts}, limit={limit}, workers={workers})", flush=True)
    import time
    t0 = time.time()
    with Pool(workers, initializer=_init_paths, initargs=(degree,), maxtasksperchild=20) as pool:
        for k, got in enumerate(pool.imap_unordered(model_paths4, files, chunksize=1)):
            if got:
                by_model[got[0]]["paths4"] = got[1]
            if (k + 1) % 50 == 0:
                elapsed = time.time() - t0
                rate = (k + 1) / max(1, elapsed)
                eta = (len(files) - k - 1) / max(0.1, rate)
                print(f"  paths4 {k+1}/{len(files)}  ({elapsed:.0f}s, {rate:.2f}/s, ETA {eta:.0f}s)", flush=True)
            if (k + 1) % 500 == 0:
                save(rows)
                print(f"    [checkpoint saved]", flush=True)
    save(rows)
    print(f"paths4: {sum('paths4' in r for r in rows)} models -> {OUT}")


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
    argv = sys.argv[1:]
    if argv == ["paths"]:
        add_paths()
    elif argv and argv[0] == "triangles":
        limit = None; max_parts = 2000; workers = 4
        i = 1
        while i < len(argv):
            if argv[i] == "--limit" and i + 1 < len(argv):
                limit = int(argv[i + 1]); i += 2
            elif argv[i] == "--max-parts" and i + 1 < len(argv):
                max_parts = int(argv[i + 1]); i += 2
            elif argv[i] == "--workers" and i + 1 < len(argv):
                workers = int(argv[i + 1]); i += 2
            else:
                i += 1
        add_triangles(limit=limit, max_parts=max_parts, workers=workers)
    elif argv and argv[0] == "paths4":
        # optional second positional argument: --limit N
        limit = None
        max_parts = 2000
        i = 1
        workers = 4
        while i < len(argv):
            if argv[i] == "--limit" and i + 1 < len(argv):
                limit = int(argv[i + 1]); i += 2
            elif argv[i] == "--max-parts" and i + 1 < len(argv):
                max_parts = int(argv[i + 1]); i += 2
            elif argv[i] == "--workers" and i + 1 < len(argv):
                workers = int(argv[i + 1]); i += 2
            else:
                i += 1
        add_paths4(limit=limit, max_parts=max_parts, workers=workers)
    else:
        main()
