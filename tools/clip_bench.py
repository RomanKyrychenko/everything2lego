"""How well semantic_labels.py's CLIP finds windows, doors, wheels, fences and plants: measured on real sets, where
every face knows its part.

Real building sets (OMR, set_kinds 'building', --min-parts..--max-parts parts, holding at least one part of a semantic
category) are turned into meshes (ldr_to_mesh.build_mesh, each face tagged with its LDraw part) and run through
semantic_labels.classify exactly as a mesh is (same --studs, region finder, renders, CLIP, keep rule; every region also
seen alone).  A region's truth is the Rebrickable category holding >= half its area (its faces matched to the set's
faces by nearest centroid, both placed as the target places them), else none; 'door' is a Windows and Doors part
named a door.

Per category: regions that are it, precision / recall of the keep rule, and average precision of the CLIP score
(the category's summed probability in context, and times its probability alone) -- threshold-free, so a better model
or prompt shows as a higher AP before any threshold is tuned.  Per-set results are cached (<out>/<set>.npz).

    .venv-arm/bin/python tools/clip_bench.py --n 20 --out /Volumes/PRO-G40/lego_data/analysis/clip_bench/base
"""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools"), str(ROOT / "tools" / "set_study")]
DATA = Path("/Volumes/PRO-G40/lego_data")

from semantic_labels import CLASSES, DEFAULT_MODEL, classify, parser  # noqa: E402

SEM = sorted({c for _, _, c in CLASSES if c})
KINDS = SEM + ["door"]                                   # 'door': a Windows and Doors part named a door


def part_info(names) -> dict:
    """LDraw name -> (Rebrickable category, part name)."""
    db = sqlite3.connect(str(DATA / "parts.sqlite")); out = {}
    for n in set(names):
        r = db.execute("select p.category, p.name from ldraw_name l join part p on p.part_num = l.part_num where l.ldraw = ?",
                       (n,)).fetchone()
        out[n] = (r[0] or "", r[1] or "") if r else ("", "")
    return out


def sets(lo: int, hi: int) -> list[Path]:
    from ldraw_read import flatten
    kinds = {r["id"]: r["kind"] for r in csv.DictReader(open(DATA / "analysis" / "set_kinds.csv"))}
    for f in sorted((DATA / "omr" / "files").glob("*.mpd")):
        if kinds.get(f.stem.split("_")[0]) != "building":
            continue
        try:
            ps = flatten(f)
        except Exception:                                # noqa: BLE001
            continue
        info = part_info(p.name for p in ps)
        if lo <= len(ps) <= hi and any(info[p.name][0] in SEM for p in ps):
            yield f


def placed(v: np.ndarray) -> np.ndarray:
    """Vertices centred on the box, scaled to max horizontal extent 1, ground 0."""
    lo, hi = v.min(0), v.max(0)
    return (v - np.array([(lo[0] + hi[0]) / 2, lo[1], (lo[2] + hi[2]) / 2])) / max(hi[0] - lo[0], hi[2] - lo[2])


def face_truth(src, dst) -> tuple[np.ndarray, np.ndarray]:
    """For each face of `dst` (the target's processed, aligned mesh), by the nearest `src` face centroid (of the four
    turns about y the one matching best): (F_dst, len(KINDS)) bool, it is of each kind; (F_dst,) its part's index."""
    from scipy.spatial import cKDTree
    info = part_info(src.metadata["face_part"])
    names = src.metadata["face_part"]
    kind = np.zeros((len(names), len(KINDS)), bool)
    for i, n in enumerate(names):
        c, nm = info[n]
        if c in SEM:
            kind[i, SEM.index(c)] = True
            kind[i, -1] = c == "Windows and Doors" and "door" in nm.lower()
    a = placed(src.triangles_center); b = placed(dst.triangles_center)
    tree, best = cKDTree(a), None
    for k in range(4):
        th = k * np.pi / 2; R = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
        d, j = tree.query(b @ R.T)
        if best is None or np.median(d) < best[0]:
            best = (np.median(d), j)
    return kind[best[1]], np.asarray(src.metadata["face_inst"])[best[1]]


def run_set(f: Path, studs: int, neg: int, extra: list[str], gen: Path | None = None) -> dict:
    """Every region touching a true feature is scored, and `neg` of the rest at random; weight = 1 / its sampling
    rate, so precision and AP are those of all regions.  inst (n, KINDS): the part instance holding most of the
    region's area of that kind (-1 none); part / part_share: the LDraw part (any) holding most of its area, and that
    share; emb / emb_alone: its image embeddings (views averaged), for matching it to parts (part_embed.py).
    gen: a generated mesh of the set (gen_bench_meshes.py) labelled instead of the set's own geometry."""
    from ldr_to_mesh import build_mesh
    src = build_mesh(f)
    got = {}

    def pick(mesh, regs):
        ft, fi = face_truth(src, mesh); fa = mesh.area_faces
        frac = np.array([(fa[r["faces"], None] * ft[r["faces"]]).sum(0) / fa[r["faces"]].sum() for r in regs])
        pos = np.flatnonzero(frac.max(1) > 0); rest = np.setdiff1d(np.arange(len(regs)), pos)
        take = np.random.default_rng(0).permutation(rest)[:neg]
        w = np.ones(len(regs)); w[take] = len(rest) / max(1, len(take))
        idx = np.sort(np.r_[pos, take])
        inst = -np.ones((len(idx), len(KINDS)), int)
        for n, i in enumerate(idx):
            g = regs[i]["faces"]
            for k in range(len(KINDS)):
                m = ft[g, k]
                if m.any():
                    u, ui = np.unique(fi[g[m]], return_inverse=True)
                    inst[n, k] = u[np.bincount(ui, fa[g[m]]).argmax()]
        names = np.asarray(src.metadata["face_part"]); sinst = np.asarray(src.metadata["face_inst"])
        first = {int(q): names[np.flatnonzero(sinst == q)[0]] for q in np.unique(sinst)}
        part, share = [], []                             # the part (any part) holding most of each region's area
        for i in idx:
            g = regs[i]["faces"]; u, ui = np.unique(fi[g], return_inverse=True); ar = np.bincount(ui, fa[g])
            part.append(first[int(u[ar.argmax()])]); share.append(ar.max() / ar.sum())
        got.update(frac=frac[idx], weight=w[idx], inst=inst, part=np.array(part), part_share=np.array(share))
        return [regs[i] for i in idx]
    with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
        src.export(tmp.name)
    try:
        c = classify(parser().parse_args([str(gen or tmp.name), "--studs", str(studs), "--out", "-", *extra]),
                     alone_all=True, pick=pick)
    finally:
        Path(tmp.name).unlink()
    regs = c["regs"]
    return dict(prob=np.array([r["prob"] for r in regs]), alone=np.array([r["prob_alone"] for r in regs]),
                emb=np.array([r["emb"] for r in regs]), emb_alone=np.array([r["emb_alone"] for r in regs]),
                keep=np.array([r["keep"] for r in regs]), cls=np.array([r["cls"] for r in regs]),
                area=np.array([r["area"] for r in regs]), faces=len(c["mesh"].faces), **got)


CCAT = np.array([c or "" for _, _, c in CLASSES]); CNAME = np.array([n for n, _, _ in CLASSES])


def scores(r: dict, c: str, score: str) -> np.ndarray:
    """A region's score for category c: P(c | in context), times P(c | alone) for score 'ctx*alone'."""
    s = r["prob"][:, CCAT == c].sum(1)
    return s * r["alone"][:, CCAT == c].sum(1) if score == "ctx*alone" else s


def fitted_keep(r: dict, fitted: dict) -> tuple[np.ndarray, np.ndarray]:
    """semantic_labels' fitted rule on cached scores: -> (keep, cls) per region."""
    tau = {c: v for c, v in fitted["thresholds"].items() if v is not None}
    sc = np.stack([scores(r, c, fitted["score"]) for c in SEM], 1)
    sc[:, [c not in tau for c in SEM]] = -1
    best = sc.argmax(1); keep = sc[np.arange(len(sc)), best] >= np.array([tau.get(c, np.inf) for c in SEM])[best]
    cls = np.array([CNAME[np.flatnonzero(CCAT == SEM[b])[np.argmax(p[CCAT == SEM[b]])]] for b, p in zip(best, r["prob"])])
    return keep, cls


def rows(res: list[dict], keep_cls=None) -> dict:
    """Per kind: true regions, kept, precision, recall (regions), instance recall (parts with a region mostly of them
    found by a kept one), AP of the scores."""
    from sklearn.metrics import average_precision_score
    cat = lambda k: [r[k] for r in res]
    prob, alone, frac, w = (np.concatenate(cat(k)) for k in ("prob", "alone", "frac", "weight"))
    kc = [keep_cls(r) if keep_cls else (r["keep"], r["cls"]) for r in res]
    keep = np.concatenate([k for k, _ in kc]); cls = np.concatenate([c for _, c in kc])
    setid = np.concatenate([np.full(len(r["prob"]), i) for i, r in enumerate(res)])
    inst = np.concatenate(cat("inst"))
    out = {}
    for i, k in enumerate(KINDS):
        y = frac[:, i] >= 0.5
        sel = CCAT == k if k != "door" else CNAME == "door"
        s, s2 = prob[:, sel].sum(1), prob[:, sel].sum(1) * alone[:, sel].sum(1)
        kept = keep & (cls == "door") if k == "door" else keep & np.isin(cls, CNAME[CCAT == k])
        tp = (w * (kept & y)).sum()
        every = {(a, b) for a, b in zip(setid[y], inst[y, i])}
        found = {(a, b) for a, b in zip(setid[kept & y], inst[kept & y, i])}
        out[k] = dict(true=int(y.sum()), kept=int(kept.sum()), precision=float(tp / max(1e-9, (w * kept).sum())),
                      recall=float(tp / max(1, y.sum())), parts=len(every), inst_recall=len(found) / max(1, len(every)),
                      ap=float(average_precision_score(y, s, sample_weight=w)) if y.any() else None,
                      ap_alone=float(average_precision_score(y, s2, sample_weight=w)) if y.any() else None,
                      base=float((w * y).sum() / w.sum()))
    return out


def show(title: str, out: dict) -> None:
    f = lambda v: f"{v:.3f}" if v is not None else "  -  "
    print(f"-- {title}")
    print(f"{'kind':26s} {'true':>5s} {'kept':>5s} {'prec':>5s} {'recall':>6s} {'parts':>5s} {'found':>6s} {'AP ctx':>6s} {'AP ctx*alone':>12s} {'base':>5s}")
    for k, r in out.items():
        print(f"{k:26s} {r['true']:5d} {r['kept']:5d} {f(r['precision'])} {f(r['recall']):>6s} {r['parts']:5d} {f(r['inst_recall']):>6s}"
              f" {f(r['ap']):>6s} {f(r['ap_alone']):>12s} {r['base']:.3f}")


def fit(res: list[dict], score: str) -> dict:
    """Per category the threshold on its score with the best F1 of precision (kept regions that are it,
    weighted) and part recall (its parts with a kept region mostly of them): the build needs each window found once,
    not every sliver of its frame."""
    tau = {}
    for c in SEM:
        i = SEM.index(c)
        y = np.concatenate([r["frac"][:, i] >= 0.5 for r in res]); w = np.concatenate([r["weight"] for r in res])
        s = np.concatenate([scores(r, c, score) for r in res])
        part = np.concatenate([[f"{j}:{q}" for q in r["inst"][:, i]] for j, r in enumerate(res)])
        if len({q for q in part[y]}) < 5:
            tau[c] = None                                # too few parts here to measure it: semantic_labels keeps its hand rule
            continue
        best = {}                                        # each part's best score among the regions that are it
        for q, v in zip(part[y], s[y]):
            best[q] = max(best.get(q, -1.0), v)
        ts = np.unique(s[y])
        bv = np.array(sorted(best.values()))
        prec = np.array([(w * (s >= t) * y).sum() / (w * (s >= t)).sum() for t in ts])
        rec = len(bv) - np.searchsorted(bv, ts, side="left")
        f1 = 2 * prec * (rec / len(bv)) / np.maximum(1e-9, prec + rec / len(bv))
        f1[prec < 0.5] = -1                              # a label is given only where it is right more often than wrong
        if f1.max() > 0:
            tau[c] = float(ts[f1.argmax()])
    return {"score": score, "thresholds": tau}


def f1_held_out(out: dict) -> float:
    """Mean over the fitted categories' kinds of F1(precision, part recall)."""
    v = [2 * o["precision"] * o["inst_recall"] / max(1e-9, o["precision"] + o["inst_recall"]) for o in out.values() if o["parts"]]
    return float(np.mean(v)) if v else 0.0


def main() -> None:
    from semantic_labels import THRESHOLDS, setting
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--gen", type=Path, default=None, help="label each set's generated mesh <gen>/<set>.glb "
                    "(gen_bench_meshes.py) instead of its own geometry; sets without one are skipped")
    ap.add_argument("--skip", type=int, default=0, help="start after this many sets (a second worker, same --out)")
    ap.add_argument("--studs", type=int, default=32)
    ap.add_argument("--clip", default=DEFAULT_MODEL, help="image-text model, as semantic_labels.py --clip")
    ap.add_argument("--views", type=int, default=3); ap.add_argument("--context-dim", type=float, default=0.4)
    ap.add_argument("--neg", type=int, default=600, help="regions touching no feature scored per set (at random)")
    ap.add_argument("--min-parts", type=int, default=150); ap.add_argument("--max-parts", type=int, default=700)
    ap.add_argument("--fit", action="store_true", help="fit the thresholds (two-fold check), write them to "
                    "tools/clip_thresholds.json for this model and render setting")
    a = ap.parse_args()
    extra = ["--clip", a.clip, "--views", str(a.views), "--context-dim", str(a.context_dim)]
    a.out.mkdir(parents=True, exist_ok=True)
    res = []
    for k, f in enumerate(sets(a.min_parts, a.max_parts)):
        if k < a.skip:
            continue
        if len(res) >= a.n:
            break
        cache = a.out / f"{f.stem}.npz"
        claim = cache.with_suffix(".claim")             # workers sharing --out split the sets
        if a.gen and not (a.gen / f"{f.stem}.glb").exists():
            continue
        if not cache.exists():
            try:
                claim.touch(exist_ok=False)
            except FileExistsError:
                continue
            print(f"== {f.name}", flush=True)
            try:
                r = run_set(f, a.studs, a.neg, extra, a.gen / f"{f.stem}.glb" if a.gen else None)
            except Exception as e:                       # noqa: BLE001
                print(f"   skipped: {e}", flush=True); continue
            np.savez_compressed(cache, **r); claim.unlink()
        r = dict(np.load(cache, allow_pickle=True))
        if len(r["prob"]):
            res.append(r)
    w = np.concatenate([r["weight"] for r in res])
    print(f"{len(res)} sets, {int(sum(len(r['prob']) for r in res))} regions scored, standing for {w.sum():.0f}")
    rep = {"sets": len(res), "keep_rule": rows(res)}
    show("keep rule as run", rep["keep_rule"])
    if a.fit:                                            # fit on one half of the sets, measure on the other, both ways
        halves = [res[0::2], res[1::2]]
        held = [r for h in (0, 1) for r in halves[1 - h]]
        best = None
        for score in ("ctx*alone", "ctx"):              # the score whose fitted rule does better on held-out sets
            fits = [fit(halves[h], score) for h in (0, 1)]
            lookup = {id(r): fits[h] for h in (0, 1) for r in halves[1 - h]}
            out = rows(held, lambda r: fitted_keep(r, lookup[id(r)]))
            show(f"fitted thresholds on {score}, measured on the held-out half", out)
            if best is None or f1_held_out(out) > best[0]:
                best = (f1_held_out(out), score, out, fits)
        rep["two_fold"], rep["score"] = best[2], best[1]
        tau = fit(res, best[1])                          # a category only if each half alone also gave it a threshold
        tau["thresholds"] = {c: v for c, v in tau["thresholds"].items()
                             if v is None or all(f["thresholds"].get(c) is not None for f in best[3])}
        rep["thresholds"] = tau
        print(f"score: {best[1]} (held-out F1 {best[0]:.3f})")
        key = setting(parser().parse_args(["-", "--out", "-", *extra]))
        allt = json.loads(THRESHOLDS.read_text()) if THRESHOLDS.exists() else {}
        allt[key] = tau; THRESHOLDS.write_text(json.dumps(allt, indent=1) + "\n")
        print(f"thresholds {key}: {tau['score']} " + ", ".join(f"{c} {'unmeasured' if v is None else f'{v:.3f}'}" for c, v in tau["thresholds"].items()) + f" -> {THRESHOLDS}")
    (a.out / "report.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
