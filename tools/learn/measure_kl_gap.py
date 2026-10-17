"""Measure KL gap at pattern orders 2 (pairs), 3 (3-walks), 4 (4-walks) —
on held-out OMR models' own placements vs. the current sampler's builds for
the same voxel targets.  Reports median gap per order and the rank correlation
between orders.  Decides whether 4-walks (paths4) add signal over 3-walks.

    python tools/learn/measure_kl_gap.py [--n 30] [--iters 20000] [--kind building]

The reference is built from the non-held-out models' statistics (as the sampler
sees them at runtime).  Compared statistic per model and order:
  gap_order = kl(sampler_build_counter, Reference) - kl(official_counter, Reference)

Lower is better.  If 4-hop median gap >= 0.5 * 3-hop median AND rank correlation
with 3-hop < 0.6 → GO (4-hop adds signal), else pivot to triangles.
"""
from __future__ import annotations

import argparse
import pickle
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "tools" / "partsdb"))
sys.path.insert(0, str(REPO / "tools" / "set_study"))
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))

import common as C  # noqa: E402
import features as F  # noqa: E402
from neighbours import neighbour_pairs  # noqa: E402
from sampler_demo import load_models  # noqa: E402
from everything2lego import sampler as S  # noqa: E402
from everything2lego.style import Reference, kl, token_degree  # noqa: E402

DEGRADE = 0.15


def placements_to_items(ps):
    """Convert sampler Placements into (part_num, class, lo, hi, yaw, face)
    items in the same format as neighbours.model_parts: y in plates, x/z in
    studs.  class/yaw/face unused by pair/path counters so we just leave None."""
    items = []
    toks = []
    for p in ps:
        h = p.part.height_units                                        # part's height in plates
        lo = np.array([float(p.x_stud), float(p.y_plate), float(p.z_stud)])
        hi = np.array([float(p.x_stud + p.w_world), float(p.y_plate + h), float(p.z_stud + p.d_world)])
        items.append((p.part.ldraw, None, lo, hi, 0, 0))
        toks.append(F._CTX["token"].get(p.part.ldraw, "other0x0h0"))
    return items, toks


def counters_from_build(ps, degree):
    """Compute (pair, p3, p4, triangles) counters from a list of sampler Placements."""
    items, toks = placements_to_items(ps)
    pairs_set = neighbour_pairs(items)
    adj: dict[int, set] = {}
    for i, j in pairs_set:
        adj.setdefault(i, set()).add(j)
    pats = Counter()
    for i, j in pairs_set:
        pats[(toks[i], toks[j], F.vertical(items[i], items[j]))] += 1
    p3 = F.paths(items, toks, adj, degree)
    p4 = F.paths4(items, toks, adj, degree)
    tri = F.triangles(items, toks, adj, degree)
    return pats, p3, p4, tri


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="held-out models to evaluate")
    ap.add_argument("--iters", type=int, default=20000, help="sampler iters per build")
    ap.add_argument("--kind", default=None, help="restrict to one kind")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    F._init()
    rows = pickle.load(open(F.OUT, "rb"))
    # tag every row with its sampler kind (building, aircraft, ...) from set_kinds.csv
    from sampler_demo import KINDS as KINDS_CSV
    if KINDS_CSV.exists():
        import pandas as pd
        k_df = pd.read_csv(KINDS_CSV).drop_duplicates("id", keep="last")
        kmap = dict(zip(k_df.id, k_df.kind))
        for r in rows:
            r["object_kind"] = kmap.get(r["model"]) or kmap.get(r["model"].split("_")[0])
    need = ("paths4", "triangles")
    rows_train = [r for r in rows if not r["held_out"] and all(k in r for k in need)]
    rows_hold = [r for r in rows if r["held_out"] and 100 <= r.get("n_parts", 0) <= 1500
                  and (a.kind is None or r.get("object_kind") == a.kind)
                  and all(k in r for k in need)]
    print(f"train rows (with paths4): {len(rows_train)}  held-out pool: {len(rows_hold)}", flush=True)
    if not rows_train or not rows_hold:
        print("ERROR: need both train and held-out rows with paths4", flush=True)
        return

    print("building references...", flush=True)
    ref_pair = Reference.from_models(rows_train, key="patterns")
    ref_p3 = Reference.from_models(rows_train, key="paths")
    ref_p4 = Reference.from_models(rows_train, key="paths4")
    ref_tri = Reference.from_models(rows_train, key="triangles")
    degree = token_degree(rows_train)

    rnd = random.Random(a.seed)
    chosen = rnd.sample(rows_hold, min(a.n, len(rows_hold)))
    models = load_models()

    gaps = {"pair": [], "p3": [], "p4": [], "tri": []}
    for ri, r in enumerate(chosen):
        try:
            got = C.target(r["model"], r.get("object_kind") or r.get("kind") or "building",
                            degrade=DEGRADE, seed=a.seed)
            if got is None:
                continue
            target = got[0]
            t0 = time.time()
            ps, _info = S.sample(target, models, iters=a.iters, seed=a.seed)
            pats, p3, p4, tri = counters_from_build(ps, degree)
            # official counters live in the row
            off_pair, off_p3, off_p4, off_tri = r["patterns"], r["paths"], r["paths4"], r["triangles"]
            g_pair = kl(pats, ref_pair) - kl(off_pair, ref_pair)
            g_p3 = kl(p3, ref_p3) - kl(off_p3, ref_p3)
            g_p4 = kl(p4, ref_p4) - kl(off_p4, ref_p4)
            g_tri = kl(tri, ref_tri) - kl(off_tri, ref_tri)
            gaps["pair"].append(g_pair)
            gaps["p3"].append(g_p3)
            gaps["p4"].append(g_p4)
            gaps["tri"].append(g_tri)
            print(f"  [{ri+1}/{len(chosen)}] {r['model']}  pair{g_pair:+.3f}  p3{g_p3:+.3f}  p4{g_p4:+.3f}  tri{g_tri:+.3f}  ({time.time()-t0:.1f}s)", flush=True)
        except Exception as e:
            print(f"  [{ri+1}/{len(chosen)}] {r['model']}: FAILED {type(e).__name__}: {e}", flush=True)

    print("\n== medians (lower = closer to real) ==")
    for k in ("pair", "p3", "p4", "tri"):
        v = gaps[k]
        if v:
            print(f"  {k:4s}: median {np.median(v):+.3f}  mean {np.mean(v):+.3f}  n={len(v)}")

    def rho(a, b):
        try:
            import scipy.stats as ss
            return ss.spearmanr(a, b).correlation
        except Exception:
            a = np.argsort(np.argsort(a)); b = np.argsort(np.argsort(b))
            return float(np.corrcoef(a, b)[0, 1])

    print("\n== rank correlations between orders ==")
    for a_k, b_k in (("pair", "p3"), ("pair", "p4"), ("pair", "tri"),
                       ("p3", "p4"), ("p3", "tri"), ("p4", "tri")):
        if gaps[a_k] and gaps[b_k]:
            n = min(len(gaps[a_k]), len(gaps[b_k]))
            print(f"  {a_k:4s} vs {b_k:4s}: rho = {rho(gaps[a_k][:n], gaps[b_k][:n]):+.2f}")

    print("\n== verdict (M1 go/no-go) ==")
    for cand in ("p4", "tri"):
        if gaps["p3"] and gaps[cand]:
            med3 = float(np.median(gaps["p3"]))
            medc = float(np.median(gaps[cand]))
            r = rho(gaps["p3"], gaps[cand])
            ratio = medc / med3 if med3 != 0 else float("inf")
            ok = abs(medc) >= 0.5 * abs(med3) and abs(r) < 0.6
            print(f"  {cand}: median {medc:+.3f} (p3 {med3:+.3f}, ratio {ratio:+.2f}), rank-corr {r:+.2f}  → {'GO' if ok else 'NO-GO'}")


if __name__ == "__main__":
    main()
