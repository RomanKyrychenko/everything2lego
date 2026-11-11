"""Does matching a mesh region to LEGO parts through the real-life objects each part depicts (part_objects.py) beat
matching it to the parts' own renders?  On clip_bench's cached regions (real sets, each region's true part):

  - look:    region image embedding . part look embedding (part_embed.py), both centred
  - objects: region image embedding . the part's object phrases ("a photo of a <object>", SigLIP text side), the
             best phrase per part, centred over parts
  - proto:   region . the part's prototype from the other sets' regions (part_match_bench.py), as a reference

Every region mostly one vocabulary part, ranked among all parts; look twins (centred look cos >= 0.9) count as hits.
Per score: top1 / top5 / top20 and the top part's category right, all regions and special (not plain) parts.

    .venv-arm/bin/python tools/part_objects_bench.py /Volumes/PRO-G40/lego_data/analysis/clip_bench/sg_gen
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools"), str(ROOT / "tools" / "set_study")]
from part_match_bench import BASIC, EMB, nz  # noqa: E402

OBJ = Path("/Volumes/PRO-G40/lego_data/analysis/part_objects/objects.jsonl")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bench", type=Path, nargs="+"); ap.add_argument("--share", type=float, default=0.5)
    ap.add_argument("--model", default="google/siglip2-so400m-patch14-384")
    ap.add_argument("--field", default="emb", help="region embedding: emb (in context) or emb_alone")
    a = ap.parse_args()
    import semantic_labels as S
    S.MODEL = a.model
    E = np.load(EMB / (a.model.replace("/", "__") + ".npz"))
    lab = {d["ldraw"]: d["objects"] for d in map(json.loads, open(OBJ)) if d["objects"]}
    keep = np.array([n in lab for n in E["ldraw"]])         # parts labelled so far
    ld, cat, look = E["ldraw"][keep], E["category"][keep], E["look"][keep].astype(np.float32)
    idx = {n: i for i, n in enumerate(ld)}
    phr = [(i, f"a photo of a {o}") for i, n in enumerate(ld) for o in lab[n]]
    T = S.embed_texts([p for _, p in phr]).astype(np.float32); own = np.array([i for i, _ in phr])
    R = [dict(np.load(f, allow_pickle=True)) for b in a.bench for f in sorted(b.glob("[!.]*.npz"))]
    R = [r for r in R if a.field in r]
    x = np.concatenate([r[a.field] for r in R]).astype(np.float32)
    part = np.concatenate([r["part"] for r in R]); share = np.concatenate([r["part_share"] for r in R])
    ok = (share >= a.share) & np.array([p in idx for p in part])
    x, t = x[ok], np.array([idx[p] for p in part[ok]])
    special = ~np.isin(cat[t], list(BASIC))
    L = nz(look); tw = (L @ L.T) >= 0.9
    so = x @ T.T                                            # (n, phrases) -> best phrase per part
    obj = np.full((len(x), len(ld)), -np.inf)
    for i in range(len(ld)):
        obj[:, i] = so[:, own == i].max(1)
    obj = obj - obj.mean(0)                                 # centred over regions: no hub part
    scores = {"look": nz(x) @ L.T, "objects": obj}
    print(f"{len(R)} sets, {len(x)} regions mostly one of {len(ld)} labelled parts ({special.sum()} special)")
    print(f"{'score':10s} {'set':8s} {'top1':>6s} {'top5':>6s} {'top20':>6s} {'cat@1':>6s}")
    for name, s in scores.items():
        for row, sel in (("all", np.ones(len(t), bool)), ("special", special)):
            rk = np.argsort(-s[sel], 1); tt = t[sel]
            hit = lambda k: np.mean([tw[q, rk[j, :k]].any() for j, q in enumerate(tt)])
            print(f"{name:10s} {row:8s} {hit(1):6.3f} {hit(5):6.3f} {hit(20):6.3f} {np.mean(cat[rk[:, 0]] == cat[tt]):6.3f}")


if __name__ == "__main__":
    main()
