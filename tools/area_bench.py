"""How well a build's area labels rank each area's true parts, per way of scoring an area from its cells: held-out
folds of cell_bench's generated sets, each cell's P(part) as cell_labels computes it on a build (cell_scorer.cell_probs:
no picture), areas as cell_labels makes them (colour-only grid regions, partition), area score = mean P(part) over its
cells / (mean over the target) ** lift, for each --lifts.  Reports the share of labelled area cells whose true part
(or a substitute, cell_scorer's twin) is in its area's top 5 / top 30 (cell_labels --top), for all, special and
window/door cells, and the mean over part categories of each one's share (macro: every category counts the same,
the criterion for choosing --lift).

--fit-thresholds F: also each area's mean P(category) (summed over the category's parts) against whether most of its
labelled cells are of that category, and per category the lowest threshold whose held-out precision is >= --precision
(at least 5 true areas above it; else none), written as JSON.  Result (40 sets, 8717 areas, no picture): no category reaches precision 0.5 (best: bars 0.17,
round bricks 0.09, windows/doors 0.02; area AUC windows 0.45, round 0.49, panels 0.73) -- without the picture the
features rank parts within a cell but do not tell which areas hold a category.

    tools/timed.sh area-bench .venv-arm/bin/python tools/area_bench.py /Volumes/PRO-G40/lego_data/analysis/cell_bench --extra '*.ldeg.npz'
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]


def main() -> None:
    from cell_scorer import FEATS, cell_probs, cell_rows, geometry, parts_table, surface_rows, train
    from grid_regions import partition, regions
    from part_match_bench import BASIC
    ap = argparse.ArgumentParser()
    ap.add_argument("bench", type=Path); ap.add_argument("--cells", type=int, default=300)
    ap.add_argument("--folds", type=int, default=5); ap.add_argument("--extra", default="")
    ap.add_argument("--lifts", default="0,0.125,0.25,0.375,0.5,0.625,0.75,0.875,1"); ap.add_argument("--area-max", type=int, default=30)
    ap.add_argument("--fit-thresholds", type=Path, default=None); ap.add_argument("--precision", type=float, default=0.5)
    a = ap.parse_args()
    t0 = time.time()
    P = parts_table(); rng = np.random.default_rng(0)
    load = lambda f: dict(np.load(f, allow_pickle=True))
    sets = []
    for f in sorted(a.bench.glob("[!.]*.gen.npz")):
        side = lambda x: a.bench / f.name.replace(".gen.npz", x)
        if not side(".evidence.npz").exists():
            continue
        r = load(f)
        if "uv" not in r:
            continue
        geo = geometry(f, r)
        d = cell_rows(r, load(side(".evidence.npz")), P, a.cells, rng,
                      load(side(".regions.npz")) if side(".regions.npz").exists() else None, geo,
                      load(side(".patch.npz")) if side(".patch.npz").exists() else None)
        if d is not None:
            sets.append((f.name.split(".")[0], d, load(side(".evidence.npz")), r, geo))
    extra = []
    for f in sorted(a.bench.glob(a.extra)) if a.extra else []:
        r = load(f); d = cell_rows(r, None, P, 0, rng, geo=geometry(f, r))
        if d is not None:
            del d["nbr"], d["cells"]
            extra.append((f.name.split(".")[0], d))
    print(f"{len(sets)} sets, {len(extra)} extra; load {time.time() - t0:.0f} s", flush=True)
    family = lambda n: n.split("_")[0].split("-")[0]
    fam = [family(x[0]) for x in sets]; ids = {f: i for i, f in enumerate(dict.fromkeys(fam))}
    fold = np.array([ids[f] % a.folds for f in fam])
    use = list(range(len(FEATS)))
    lifts = [float(x) for x in a.lifts.split(",")]
    rows = []                                          # (lift, k, special, window/door, hit, category) per area cell
    nc = P["cats"].max() + 1; cs, ct = [], []           # per area: mean P(category), majority category (-1 none)
    for k in range(a.folds):
        tr = [x[:3] for x, q in zip(sets, fold) if q != k]
        tf = {family(x[0]) for x, q in zip(sets, fold) if q == k}
        T, _, w1, w = train(P, tr, [x for x in extra if family(x[0]) not in tf], use, P["twin"])
        for (_, d, _, r, geo), q in zip(sets, fold):
            if q != k:
                continue
            sd = surface_rows(r, P, d["lab"], geo)
            p = cell_probs(sd, P, T, w1, use, w)
            cells, cands = regions(r, 0.0, 0.0)
            lut = np.full(r["want"].shape, -1); lut[tuple(sd["cells"].T)] = np.arange(len(sd["cells"]))
            row = lut[tuple(cells.T)]; p = p[row]; lab = sd["lab"][row]
            base = p.mean(0)
            for c in partition(cells, cands, a.area_max):
                t = lab[c]; t = t[t >= 0]
                if not len(t):
                    continue
                m = p[c].mean(0)
                cs.append(np.bincount(P["cats"], m, nc)); cc = np.bincount(P["cats"][t], minlength=nc)
                ct.append(cc.argmax() if cc.max() >= 0.5 * len(t) else -1)
                sp = ~np.isin(P["cat"][t], list(BASIC)); wd = P["cat"][t] == "Windows and Doors"
                for L in lifts:
                    order = np.argsort(-(m / base ** L))[:30]
                    hit = P["twin"][t[:, None], order[None]]              # (cells, 30)
                    for kk in (5, 30):
                        rows += [(L, kk, s_, w_, h, c_) for s_, w_, h, c_ in zip(sp, wd, hit[:, :kk].any(1), P["cats"][t])]
        print(f"fold {k}: {time.time() - t0:.0f} s", flush=True)
    if a.fit_thresholds:
        import json
        cs, ct = np.array(cs), np.array(ct); names = np.unique(P["cat"]); th = {}
        np.savez_compressed(a.fit_thresholds.with_suffix(".npz"), score=cs, true=ct, names=names)
        for i in range(nc):
            if names[i] in BASIC:
                continue
            o = np.argsort(-cs[:, i]); y = ct[o] == i
            tp = np.cumsum(y); prec = tp / np.arange(1, len(y) + 1)
            ok = np.flatnonzero((prec >= a.precision) & (tp >= 5))
            th[str(names[i])] = float(cs[o[ok[-1]], i]) if len(ok) else None
            if y.sum():
                best = prec[np.flatnonzero(tp >= 5)].max() if (tp >= 5).any() else 0
                print(f"  {names[i][:26]:26s} true areas {y.sum():4d} of {len(y)}  best precision {best:.2f}  threshold {th[str(names[i])]}  "
                      f"recall {tp[ok[-1]] / y.sum() if len(ok) else 0:.2f}")
        a.fit_thresholds.write_text(json.dumps(th, indent=1))
    R = np.array(rows, float)
    print(f"{'lift':>5s} {'k':>3s} {'all':>6s} {'special':>8s} {'win/door':>9s} {'macro':>6s}   (share of labelled area cells with a true part in the area's top k)")
    for L in lifts:
        for kk in (5, 30):
            q = (R[:, 0] == L) & (R[:, 1] == kk)
            macro = np.mean([R[q & (R[:, 5] == c), 4].mean() for c in np.unique(R[q, 5]) if (q & (R[:, 5] == c)).sum() >= 50])
            print(f"{L:5.3f} {kk:3d} {R[q, 4].mean():6.3f} {R[q & (R[:, 2] > 0), 4].mean():8.3f} {R[q & (R[:, 3] > 0), 4].mean():9.3f} {macro:6.3f}")
    print(f"cells: all {int(((R[:, 0] == lifts[0]) & (R[:, 1] == 5)).sum())}, special {int(((R[:, 0] == lifts[0]) & (R[:, 1] == 5) & (R[:, 2] > 0)).sum())}, "
          f"win/door {int(((R[:, 0] == lifts[0]) & (R[:, 1] == 5) & (R[:, 3] > 0)).sum())}; total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
