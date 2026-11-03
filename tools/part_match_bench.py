"""How well a mesh region is matched to its own LEGO part, part by part (no categories): clip_bench.py's cached
regions (real sets; each region's true part, its image embeddings) against part_embed.py's per-part embeddings.

A region counts when one part holds >= --share of its area and that part is in the vocabulary.  Scores, each a cosine:
look (region in context . part look), look-alone (region alone . part look), name (region . part name), and the sum
of look-alone and name z-scored per region; 'centred' (each side minus its mean -- the raw cosines are ruled by a few
hub parts); 'ridge' (a linear map region -> its part's look, fitted on the other sets, so it must carry to parts and
buildings it has not seen; 'unseen' rows are regions whose part no training set held).  Per score: share of regions whose true part ranks 1st / in the top 5 /
top 20 of all parts, and whose best part has the true part's category -- for all counted regions and for those of
special parts (not plain bricks, plates, tiles, slopes).

    .venv-arm/bin/python tools/part_match_bench.py /Volumes/PRO-G40/lego_data/analysis/clip_bench/sg_emb
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

EMB = Path("/Volumes/PRO-G40/lego_data/analysis/part_embed")
BASIC = {"Bricks", "Plates", "Tiles", "Bricks Sloped", "Bricks Curved", "Tiles Special", "Plates Special", "Bricks Special"}


def z(s: np.ndarray) -> np.ndarray:
    return (s - s.mean(1, keepdims=True)) / (s.std(1, keepdims=True) + 1e-9)


def nz(x: np.ndarray, m=None) -> np.ndarray:
    x = x - (x.mean(0) if m is None else m)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bench", type=Path); ap.add_argument("--share", type=float, default=0.5)
    ap.add_argument("--ridge", type=float, default=1.0)
    ap.add_argument("--model", default="google/siglip2-so400m-patch14-384")
    a = ap.parse_args()
    E = np.load(EMB / (a.model.replace("/", "__") + ".npz"))
    idx = {n: i for i, n in enumerate(E["ldraw"])}
    look, text, cat = E["look"].astype(np.float32), E["text"].astype(np.float32), E["category"]
    R = [dict(np.load(f, allow_pickle=True)) for f in sorted(a.bench.glob("*.npz"))]
    R = [r for r in R if "emb" in r]
    emb = np.concatenate([r["emb"] for r in R]).astype(np.float32)
    alone = np.concatenate([r["emb_alone"] for r in R]).astype(np.float32)
    part = np.concatenate([r["part"] for r in R]); share = np.concatenate([r["part_share"] for r in R])
    w = np.concatenate([r["weight"] for r in R])
    st = np.concatenate([np.full(len(r["part"]), i) for i, r in enumerate(R)])
    ok = (share >= a.share) & np.array([p in idx for p in part])
    t = np.array([idx.get(p, -1) for p in part])[ok]
    emb, alone, w, st = emb[ok], alone[ok], w[ok], st[ok]
    special = ~np.isin(cat[t], list(BASIC))
    print(f"{len(R)} sets; {ok.sum()} regions mostly one vocabulary part ({special.sum()} special), {len(idx)} parts")
    scores = {"look": emb @ look.T, "look-alone": alone @ look.T, "name": emb @ text.T}
    scores["look-alone+name"] = z(scores["look-alone"]) + z(scores["name"])
    L = nz(look)
    scores["centred"] = nz(alone) @ L.T
    ridge, unseen = np.zeros_like(scores["look"]), np.zeros(len(t), bool)
    for i in range(len(R)):                             # leave one set out
        tr, te = st != i, st == i
        m = alone[tr].mean(0); A = nz(alone[tr], m)
        W = np.linalg.solve(A.T @ A + a.ridge * np.eye(A.shape[1]), A.T @ L[t[tr]])
        ridge[te] = nz(nz(alone[te], m) @ W, 0) @ L.T
        unseen[te] = ~np.isin(t[te], t[tr])
    scores["ridge"] = ridge
    print(f"{'score':16s} {'set':8s} {'top1':>6s} {'top5':>6s} {'top20':>6s} {'cat@1':>6s}")
    for k, s in scores.items():
        rank = (s > s[np.arange(len(s)), t][:, None]).sum(1)
        cat1 = cat[s.argmax(1)] == cat[t]
        for nm, m in (("all", np.ones(len(t), bool)), ("special", special), ("unseen", unseen)):
            if not m.any():
                continue
            f = lambda x: (w[m] * x[m]).sum() / w[m].sum()
            print(f"{k:16s} {nm:8s} {f(rank == 0):6.3f} {f(rank < 5):6.3f} {f(rank < 20):6.3f} {f(cat1):6.3f}")


if __name__ == "__main__":
    main()
