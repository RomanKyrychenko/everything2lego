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


def write_protos(a, E, alone, t, special, held, tw, cand, unseen) -> None:
    """Prototypes of all sets' regions; T = the temperature whose softmax over special parts gives the held-out true
    part (or a twin) the most likelihood; q = the share of special regions whose part no other set held."""
    m = alone.mean(0); A = nz(alone, m); j = np.unique(t)
    P = nz(np.stack([A[t == p].mean(0) for p in j]), 0)
    ref = A @ P.T                                       # each part on its own scale: a common part's prototype is
    mu, sd = ref.mean(0), ref.std(0) + 1e-9             # near everything (hub) -- its score standardised
    sel = special & ~unseen
    hz = held.copy(); hz[:, j] = (held[:, j] - mu) / sd
    s = np.where(cand & np.isfinite(hz[sel]), hz[sel], -np.inf); hit = tw[t[sel]] & np.isfinite(s)
    def ll(T):
        x = s / T; mx = x.max(1, keepdims=True)
        lse = lambda y: mx[:, 0] + np.log(np.exp(y - mx).sum(1))
        return (lse(np.where(hit, x, -np.inf)) - lse(x)).mean()
    Ts = np.geomspace(0.03, 30, 61); T = float(Ts[np.argmax([ll(T) for T in Ts])])
    q = float(unseen[special].mean())
    f = EMB / (a.model.replace("/", "__") + ".protos.npz")
    np.savez_compressed(f, ldraw=E["ldraw"][j], proto=P.astype(np.float16), mean=m.astype(np.float32), T=T, q=q,
                        mu=mu.astype(np.float32), sd=sd.astype(np.float32))
    raw = np.where(cand & np.isfinite(held[sel]), held[sel], -np.inf)
    print(f"   held-out top1, special seen regions: raw {tw[t[sel], raw.argmax(1)].mean():.3f}, standardised {tw[t[sel], s.argmax(1)].mean():.3f}")
    print(f"-> {f}: {len(j)} parts, T {T:.4f} (held-out log-lik {ll(T):.3f}), unseen share {q:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bench", type=Path); ap.add_argument("--share", type=float, default=0.5)
    ap.add_argument("--ridge", type=float, default=1.0)
    ap.add_argument("--twin", type=float, default=0.9)
    ap.add_argument("--write", action="store_true", help="write the prototypes of all sets (alone, centred) and the "
                    "softmax temperature fitted on the held-out scores to " + str(EMB) + "/<model>.protos.npz")
    ap.add_argument("--model", default="google/siglip2-so400m-patch14-384")
    a = ap.parse_args()
    E = np.load(EMB / (a.model.replace("/", "__") + ".npz"))
    idx = {n: i for i, n in enumerate(E["ldraw"])}
    look, text, cat = E["look"].astype(np.float32), E["text"].astype(np.float32), E["category"]
    R = [dict(np.load(f, allow_pickle=True)) for f in sorted(a.bench.glob("[!.]*.npz"))]   # not macOS ._ files
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
    L = nz(look)
    tw = L @ L.T >= a.twin                              # look twins: no picture tells them apart
    scores = {"look": emb @ look.T, "name": emb @ text.T, "centred": nz(alone) @ L.T}
    ridge, unseen = np.zeros_like(scores["look"]), np.zeros(len(t), bool)
    proto = {k: np.full_like(ridge, -np.inf) for k in ("alone", "ctx")}
    filled = np.zeros_like(ridge); zonly = np.zeros_like(ridge)
    for i in range(len(R)):                             # leave one set out
        tr, te = st != i, st == i
        unseen[te] = ~np.isin(t[te], t[tr])
        m = alone[tr].mean(0); A = nz(alone[tr], m)
        W = np.linalg.solve(A.T @ A + a.ridge * np.eye(A.shape[1]), A.T @ L[t[tr]])
        ridge[te] = nz(nz(alone[te], m) @ W, 0) @ L.T
        # a part's own embedding = the mean of its regions in the other sets (same renderer, same kind of mesh)
        for k, X in (("alone", alone), ("ctx", emb)):
            m = X[tr].mean(0); A = nz(X[tr], m); j = np.unique(t[tr])
            Pj = nz(np.stack([A[t[tr] == p].mean(0) for p in j]), 0)
            proto[k][np.ix_(te, j)] = nz(X[te], m) @ Pj.T
            if k == "ctx":                              # parts no set holds: their look, mapped to where prototypes lie
                V = np.linalg.solve(L[j].T @ L[j] + a.ridge * np.eye(L.shape[1]), L[j].T @ Pj)
                P = nz(L @ V, 0); P[j] = Pj
                ref = A @ P.T                           # each part scored on its own scale (training regions)
                filled[te] = (nz(X[te], m) @ P.T - ref.mean(0)) / (ref.std(0) + 1e-9)
                zp = filled[te].copy(); zp[:, np.setdiff1d(np.arange(len(L)), j)] = -np.inf; zonly[te] = zp
    scores["ridge"] = ridge
    if a.write:
        write_protos(a, E, alone, t, special, proto["alone"], tw, ~np.isin(cat, list(BASIC)), unseen)
    scores["proto-alone"], scores["proto-ctx"] = proto["alone"], proto["ctx"]
    scores["proto-ctx+look"] = filled; scores["proto-ctx z"] = zonly
    cand_sp = ~np.isin(cat, list(BASIC))
    print(f"hit = the true part or a look twin (centred look cos >= {a.twin}); 'special' ranks special regions "
          f"among the {cand_sp.sum()} special parts only")
    print(f"{'score':16s} {'set':8s} {'top1':>6s} {'top5':>6s} {'top20':>6s} {'cat@1':>6s}")
    for k, s in scores.items():
        for nm, m, c in (("all", np.ones(len(t), bool), None), ("special", special, cand_sp),
                         ("unseen", unseen & special, cand_sp)):
            if not m.any():
                continue
            sc = s[m] if c is None else np.where(c, s[m], -np.inf)
            top = np.argsort(-sc, 1)[:, :20]
            hit = tw[t[m][:, None], top]
            f = lambda x: (w[m] * x).sum() / w[m].sum()
            print(f"{k:16s} {nm:8s} {f(hit[:, 0]):6.3f} {f(hit[:, :5].any(1)):6.3f} {f(hit.any(1)):6.3f} "
                  f"{f(cat[top[:, 0]] == cat[t[m]]):6.3f}")

if __name__ == "__main__":
    main()
