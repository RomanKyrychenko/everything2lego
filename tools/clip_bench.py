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

from semantic_labels import CLASSES, CLIP_ID, classify, parser  # noqa: E402

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


def face_truth(src, dst) -> np.ndarray:
    """(F_dst, len(KINDS)) bool: each face of `dst` (the target's processed, aligned mesh) is of each kind, by the
    nearest `src` face centroid; of the four turns about y the one matching best."""
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
    return kind[best[1]]


def run_set(f: Path, studs: int, neg: int, clip: str) -> dict:
    """Every region touching a true feature is scored, and `neg` of the rest at random; weight = 1 / its sampling
    rate, so precision and AP are those of all regions."""
    from ldr_to_mesh import build_mesh
    src = build_mesh(f)
    got = {}

    def pick(mesh, regs):
        ft = face_truth(src, mesh); fa = mesh.area_faces
        frac = np.array([(fa[r["faces"], None] * ft[r["faces"]]).sum(0) / fa[r["faces"]].sum() for r in regs])
        pos = np.flatnonzero(frac.max(1) > 0); rest = np.setdiff1d(np.arange(len(regs)), pos)
        take = np.random.default_rng(0).permutation(rest)[:neg]
        w = np.ones(len(regs)); w[take] = len(rest) / max(1, len(take))
        idx = np.sort(np.r_[pos, take]); got.update(frac=frac[idx], weight=w[idx])
        return [regs[i] for i in idx]
    with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
        src.export(tmp.name)
    try:
        c = classify(parser().parse_args([tmp.name, "--studs", str(studs), "--out", "-", "--clip", clip]), alone_all=True, pick=pick)
    finally:
        Path(tmp.name).unlink()
    regs = c["regs"]
    return dict(prob=np.array([r["prob"] for r in regs]), alone=np.array([r["prob_alone"] for r in regs]),
                keep=np.array([r["keep"] for r in regs]), cls=np.array([r["cls"] for r in regs]),
                area=np.array([r["area"] for r in regs]), faces=len(c["mesh"].faces), **got)


def report(res: list[dict]) -> dict:
    from sklearn.metrics import average_precision_score
    prob = np.concatenate([r["prob"] for r in res]); alone = np.concatenate([r["alone"] for r in res])
    frac = np.concatenate([r["frac"] for r in res]); keep = np.concatenate([r["keep"] for r in res])
    cls = np.concatenate([r["cls"] for r in res]); w = np.concatenate([r["weight"] for r in res])
    ccat = np.array([c or "" for _, _, c in CLASSES]); cname = np.array([n for n, _, _ in CLASSES])
    out = {"sets": len(res), "regions": int(len(prob))}
    print(f"{len(res)} sets, {len(prob)} regions scored, standing for {w.sum():.0f}")
    print(f"{'kind':26s} {'true':>5s} {'kept':>5s} {'prec':>5s} {'recall':>6s} {'AP ctx':>6s} {'AP ctx*alone':>12s} {'base':>5s}")
    for i, k in enumerate(KINDS):
        y = frac[:, i] >= 0.5
        sel = ccat == k if k != "door" else cname == "door"
        s, s2 = prob[:, sel].sum(1), prob[:, sel].sum(1) * alone[:, sel].sum(1)
        if k == "door":
            kept = keep & (cls == "door")
        else:
            kept = keep & np.isin(cls, cname[ccat == k])
        tp = (w * (kept & y)).sum()
        row = dict(true=int(y.sum()), kept=int(kept.sum()), precision=tp / max(1e-9, (w * kept).sum()), recall=tp / max(1, y.sum()),
                   ap=float(average_precision_score(y, s, sample_weight=w)) if y.any() else None,
                   ap_alone=float(average_precision_score(y, s2, sample_weight=w)) if y.any() else None,
                   base=float((w * y).sum() / w.sum()))
        out[k] = row
        f = lambda v: f"{v:.3f}" if v is not None else "  -  "
        print(f"{k:26s} {row['true']:5d} {row['kept']:5d} {f(row['precision'])} {f(row['recall']):>6s} {f(row['ap']):>6s}"
              f" {f(row['ap_alone']):>12s} {row['base']:.3f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--studs", type=int, default=32)
    ap.add_argument("--clip", default=CLIP_ID, help="image-text model, as semantic_labels.py --clip")
    ap.add_argument("--neg", type=int, default=600, help="regions touching no feature scored per set (at random)")
    ap.add_argument("--min-parts", type=int, default=150); ap.add_argument("--max-parts", type=int, default=700)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    res = []
    for f in sets(a.min_parts, a.max_parts):
        if len(res) >= a.n:
            break
        cache = a.out / f"{f.stem}.npz"
        if not cache.exists():
            print(f"== {f.name}", flush=True)
            try:
                r = run_set(f, a.studs, a.neg, a.clip)
            except Exception as e:                       # noqa: BLE001
                print(f"   skipped: {e}", flush=True); continue
            np.savez_compressed(cache, **r)
        r = dict(np.load(cache, allow_pickle=True))
        if len(r["prob"]):
            res.append(r)
    (a.out / "report.json").write_text(json.dumps(report(res), indent=1))


if __name__ == "__main__":
    main()
