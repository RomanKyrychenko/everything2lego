"""Per-cell part evidence: for each surface cell of a target, a weight over every vocabulary part from what the builder
cannot see itself -- the picture and real sets -- with shape left to the builder's own fit.  Measured on cell_bench.py
(generated meshes of real sets, the true part of each cell), the weights fitted with each fold of sets held out.

Features of (cell, part), each centred over parts:
  look    the cell's picture tile (picture_evidence.py; the 8x8 tile its pixel is in, and the 4x4 one) . the part's
          look (part_embed.py), on cells the picture sees (seen >= 0.3; else 0)
  desc    the tile . the part's described objects (part_objects_embed.py, idf-weighted vector), seen cells
  name    which of the object's named parts the tile shows (softmax over the names) times how well each name matches
          the part's described objects (best phrase, idf-weighted), seen cells
  setname the object's names against the part (best name), the same for every cell of the object
  prior   log P(part | cell's height band, facing) and log P(part's category | same), from the other folds' cells
          (counts + 1), and log n_sets
  colour  the part is made in the cell's colour (vocabulary colours)

A cell counts when one part holds >= half its samples and that part is in the vocabulary; a hit is the true part or
a substitute: a look twin (centred look cos >= 0.9) or the same category with the same stud-scale box (stud_shape).

    tools/timed.sh cell-scorer .venv-arm/bin/python tools/cell_scorer.py /Volumes/PRO-G40/lego_data/analysis/cell_bench
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]
EMB = Path("/Volumes/PRO-G40/lego_data/analysis/part_embed/google__siglip2-so400m-patch14-384.npz")
OBJ = Path("/Volumes/PRO-G40/lego_data/analysis/part_objects/google__siglip2-so400m-patch14-384.objects.npz")
FEATS = ("look", "desc", "name", "setname", "prior_part", "prior_cat", "n_sets", "colour")
HB, FB = 5, 5                                            # height bands, facing bands


def unit(x):
    x = np.asarray(x, np.float32); return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def parts_table():
    from everything2lego.colors import palette
    from everything2lego.sampler import vocabulary
    from everything2lego.stud_shape import part_descriptor
    from part_match_bench import nz
    E = np.load(EMB); O = np.load(OBJ)
    ld = E["ldraw"]; idx = {n: i for i, n in enumerate(ld)}
    V = {}
    for v in vocabulary():
        V.setdefault(str(getattr(v.part, "ldraw", v.part)), v)
    code_of = np.array([c.code for c in palette()])
    made = np.zeros((len(ld), len(code_of)), bool); nsets = np.ones(len(ld)); box = np.zeros((len(ld), 3), int)
    for i, n in enumerate(ld):
        v = V.get(str(n))
        if v is None:
            continue
        made[i] = np.isin(code_of, list(v.colours)); nsets[i] = max(1, v.n_sets)
        box[i] = part_descriptor(v.orients[0])[:3]
    look = nz(E["look"].astype(np.float32))
    oi = np.array([idx[n] for n in O["ldraw"]])
    desc = np.zeros_like(look); desc[oi] = unit(O["part_vec"])
    ph_part = oi[O["phrase_part"]]; ph_vec = unit(O["phrase_vec"]); idf = O["idf"]
    cat = E["category"]
    twin = (look @ look.T >= 0.9) | ((cat[:, None] == cat[None]) & (box[:, None] == box[None]).all(-1))
    return dict(ld=ld, idx=idx, look=look, desc=desc, ph_part=ph_part, ph_vec=ph_vec, idf=idf, cat=cat,
                cats=np.unique(cat, return_inverse=True)[1], made=made, nsets=nsets, twin=twin)


def cell_rows(r: dict, ev: dict, P: dict, n_max: int, rng) -> dict | None:
    """The scored cells of one set: their bands, colour, tile embeddings and true part."""
    s = r["want"] & r["visible"] & (r["inst"] >= 0) & (r["share"] >= 0.5)
    c = np.argwhere(s)
    part = r["inst_part"][r["inst"][tuple(c.T)]]
    ok = np.array([p in P["idx"] for p in part])
    c, part = c[ok], part[ok]
    if not len(c):
        return None
    c = c[rng.permutation(len(c))[:n_max]]; part = r["inst_part"][r["inst"][tuple(c.T)]]
    H = r["want"].shape[0]
    nrm = r["normal"].reshape(*r["want"].shape, 3).astype(np.float32)[tuple(c.T)]
    hb = np.minimum((c[:, 0] / max(1, H - 1) * HB).astype(int), HB - 1)
    fb = np.digitize(nrm[:, 0], [-0.8, -0.3, 0.3, 0.8])           # down .. up (normal's y)
    uv = r["uv"][tuple(c.T)].astype(float); seen = r["seen"][tuple(c.T)].astype(float) >= 0.3
    box, sc, te = ev["tile_box"], ev["tile_scale"], unit(ev["tile_emb"])
    emb = np.zeros((len(c), te.shape[1]), np.float32)
    for n in np.unique(sc):                                  # each scale: the tile whose centre is nearest the pixel
        k = np.flatnonzero(sc == n); cen = (box[k, :2] + box[k, 2:]) / 2
        j = k[np.argmin(((uv[:, None] - cen[None]) ** 2).sum(-1), 1)]
        emb += te[j]
    return dict(true=np.array([P["idx"][p] for p in part]), band=hb * FB + fb, colour=r["colour"][tuple(c.T)],
                tile=unit(emb), seen=seen & (uv[:, 0] >= 0))


def features(d: dict, ev: dict, P: dict, prior_p: np.ndarray, prior_c: np.ndarray) -> np.ndarray:
    """(n_cells, n_parts, len(FEATS)) for one set."""
    n, m = len(d["true"]), len(P["ld"])
    F = np.zeros((n, m, len(FEATS)), np.float32)
    sw = d["seen"][:, None].astype(np.float32)
    F[..., 0] = d["tile"] @ P["look"].T * sw
    F[..., 1] = d["tile"] @ P["desc"].T * sw
    names = unit(ev["name_emb"]) if len(ev["names"]) else np.zeros((0, P["ph_vec"].shape[1]))
    if len(names):
        s = names @ P["ph_vec"].T * P["idf"][None]                  # name x phrase
        nm = np.full((len(names), m), 0.0, np.float32)
        for k in range(len(names)):                                 # best phrase per part
            np.maximum.at(nm[k], P["ph_part"], s[k])
        a = d["tile"] @ names.T * 30; a = np.exp(a - a.max(1, keepdims=True)); a /= a.sum(1, keepdims=True)
        F[..., 2] = (a @ nm) * sw
        F[..., 3] = nm.max(0)[None]
    F[..., 4] = prior_p[d["band"]]
    F[..., 5] = prior_c[d["band"]][:, P["cats"]]
    F[..., 6] = np.log(P["nsets"])[None]
    F[..., 7] = P["made"][:, np.maximum(d["colour"], 0)].T
    F -= F.mean(1, keepdims=True)                                   # centred over parts: no hub part
    return F


def fit(Fs: list, ts: list, iters: int = 200) -> np.ndarray:
    """Weights of a softmax over parts (score = F . w) maximising the true part's (or a twin's... the true part's)
    log-likelihood: L-BFGS on the held-in cells."""
    from scipy.optimize import minimize
    F = np.concatenate(Fs); t = np.concatenate(ts)

    def f(w):
        s = F @ w; mx = s.max(1, keepdims=True); e = np.exp(s - mx); Z = e.sum(1, keepdims=True); p = e / Z
        ll = (s[np.arange(len(t)), t] - mx[:, 0] - np.log(Z[:, 0])).mean()
        g = (F[np.arange(len(t)), t] - np.einsum("np,npf->nf", p, F)).mean(0)
        return -ll + 1e-3 * w @ w, -g + 2e-3 * w
    return minimize(f, np.zeros(F.shape[-1]), jac=True, method="L-BFGS-B", options={"maxiter": iters}).x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bench", type=Path); ap.add_argument("--cells", type=int, default=300)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--only", default="", help="comma list of features to use (default all)")
    ap.add_argument("--random", action="store_true", help="score parts at random: the chance level of this metric")
    ap.add_argument("--save", type=Path, default=None, help="also fit on every set and write the model (priors, weights) here, for cell_labels.py")
    a = ap.parse_args()
    t0 = time.time()
    P = parts_table(); rng = np.random.default_rng(0)
    sets = []
    for f in sorted(a.bench.glob("[!.]*.gen.npz")):
        ev = a.bench / f.name.replace(".gen.npz", ".evidence.npz")
        r = dict(np.load(f, allow_pickle=True))
        if "uv" not in r or not ev.exists():
            continue
        ev = dict(np.load(ev, allow_pickle=True)); d = cell_rows(r, ev, P, a.cells, rng)
        if d is not None:
            sets.append((f.name.split(".")[0], d, ev))
    print(f"{len(sets)} sets, {sum(len(d['true']) for _, d, _ in sets)} cells, {len(P['ld'])} parts; load {time.time() - t0:.0f} s")
    use = [FEATS.index(x) for x in a.only.split(",")] if a.only else list(range(len(FEATS)))
    m, nb = len(P["ld"]), HB * FB
    fam = [n.split("_")[0].split("-")[0] for n, _, _ in sets]          # a model's cars / variants: one fold
    ids = {f: i for i, f in enumerate(dict.fromkeys(fam))}
    fold = np.array([ids[f] % a.folds for f in fam])
    ranks, cats, seen_all, W = [], [], [], []
    for k in range(a.folds):
        tr = [s for s, q in zip(sets, fold) if q != k]; te = [s for s, q in zip(sets, fold) if q == k]
        cp = np.ones((nb, m)); cc = np.ones((nb, P["cats"].max() + 1))
        for _, d, _ in tr:                                          # priors from the held-in sets only
            np.add.at(cp, (d["band"], d["true"]), 1); np.add.at(cc, (d["band"], P["cats"][d["true"]]), 1)
        prior_p = np.log(cp / cp.sum(1, keepdims=True)); prior_c = np.log(cc / cc.sum(1, keepdims=True))
        Ftr = [features(d, ev, P, prior_p, prior_c)[..., use] for _, d, ev in tr]
        w = fit(Ftr, [d["true"] for _, d, _ in tr]); W.append(w)
        for _, d, ev in te:
            s = features(d, ev, P, prior_p, prior_c)[..., use] @ w
            if a.random:
                s = rng.random(s.shape)
            order = np.argsort(-s, 1)
            hit = P["twin"][d["true"][:, None], order]               # (n, m): rank positions holding a substitute
            ranks.append(np.argmax(hit, 1)); cats.append(P["cat"][d["true"]]); seen_all.append(d["seen"])
    r, c, sn = np.concatenate(ranks), np.concatenate(cats), np.concatenate(seen_all)
    from part_match_bench import BASIC
    sp = ~np.isin(c, list(BASIC))
    print("weights (mean over folds): " + ", ".join(f"{FEATS[i]} {x:+.2f}" for i, x in zip(use, np.mean(W, 0))))
    print(f"{'cells':16s} {'n':>6s} {'top1':>6s} {'top5':>6s} {'top20':>6s} {'top50':>6s}")
    for name, sel in (("all", np.ones(len(r), bool)), ("plain", ~sp), ("special", sp), ("special seen", sp & sn),
                      ("special unseen", sp & ~sn)):
        q = r[sel]
        print(f"{name:16s} {sel.sum():6d} " + " ".join(f"{(q < k).mean():6.3f}" for k in (1, 5, 20, 50)))
    if a.save:                                                      # the model from all sets, for building
        cp = np.ones((nb, m)); cc = np.ones((nb, P["cats"].max() + 1))
        for _, d, _ in sets:
            np.add.at(cp, (d["band"], d["true"]), 1); np.add.at(cc, (d["band"], P["cats"][d["true"]]), 1)
        prior_p = np.log(cp / cp.sum(1, keepdims=True)); prior_c = np.log(cc / cc.sum(1, keepdims=True))
        w = fit([features(d, ev, P, prior_p, prior_c)[..., use] for _, d, ev in sets], [d["true"] for _, d, _ in sets])
        np.savez_compressed(a.save, prior_p=prior_p, prior_c=prior_c, w=w, use=np.array(use), ld=P["ld"])
        print(f"model -> {a.save}: " + ", ".join(f"{FEATS[i]} {x:+.2f}" for i, x in zip(use, w)))
    print(f"total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
