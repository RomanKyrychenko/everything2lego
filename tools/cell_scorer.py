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
  spot_look  the best, over the regions holding the cell that the picture shows (region_evidence.py: the labeller's
             candidates at every scale, cropped from the picture), of region . the part's look
  spot_desc  ... of the region against the part's described objects: its best phrase, times the phrase's idf
  spot_cls   ... of log P(the part's category | region) from semantic_labels' classes (window, door, fence, ... and
             the plain ones summed for every other category): the labeller's category evidence
  spot_keep  1 if a region holding the cell passes the labeller's fitted keep rule for the part's category
             (clip_thresholds.json: P(category | region) >= its threshold, fitted for precision >= 0.5)

  prior_part_rec, prior_cat_rec  log P(part | facing, recess) and of its category: how far the cell sits behind the
             surface around it, along its outward normal (recess(); an opening is set back from its wall)
  prior_part_ext, prior_cat_ext  ... | facing, extent): the height and long side of the colour area holding the cell
             (grid_regions' colour-only areas, partition(); a window pane is a dark 2 x 6 patch)
  fit        how close that area's box (height in plates, long and short side in studs) is to the part's own box
  ctx        the neighbours' parts: log mean over the cell's neighbouring cells of sum_q p1(q) A[q, part], p1 a first
             pass of the features above (no picture), A = P(neighbour's part | part) counted on the training cells
  x_prior_part, x_prior_part_rec, x_prior_part_ext  the part priors counted on the --extra sets alone (other
             models, degraded LDraw meshes): their own features, so the fit weighs that source against the benchmark's
  knn        appearance votes: the cell's DINOv2 patch (patch_evidence.py) against every labelled seen cell of the
             training sets, log share of the K most alike holding the part (seen cells; else 0)

--extra GLOB: more sets for the count tables only (the x_ priors, and A): cell_bench --extra's degraded official models.
Count tables use every labelled cell of the training sets; the fit and the scores, --cells sampled per set.

A cell counts when one part holds >= half its samples and that part is in the vocabulary; a hit is the true part or
a substitute: a look twin (centred look cos >= 0.9) or the same category with the same stud-scale box (stud_shape).

    tools/timed.sh cell-scorer .venv-arm/bin/python tools/cell_scorer.py /Volumes/PRO-G40/lego_data/analysis/cell_bench
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]
EMB = Path("/Volumes/PRO-G40/lego_data/analysis/part_embed/google__siglip2-so400m-patch14-384.npz")
OBJ = Path("/Volumes/PRO-G40/lego_data/analysis/part_objects/google__siglip2-so400m-patch14-384.objects.npz")
FEATS = ("look", "desc", "name", "setname", "prior_part", "prior_cat", "n_sets", "colour", "spot_look", "spot_desc", "spot_cls", "spot_keep",
         "prior_part_rec", "prior_cat_rec", "prior_part_ext", "prior_cat_ext", "fit", "ctx", "knn",
         "x_prior_part", "x_prior_part_rec", "x_prior_part_ext")
BASE = ("prior_part", "prior_cat", "n_sets", "colour", "prior_part_rec", "prior_cat_rec", "prior_part_ext", "prior_cat_ext", "fit",
        "x_prior_part", "x_prior_part_rec", "x_prior_part_ext")
HB, FB = 5, 5                                            # height bands, facing bands
NREC, NEXT = 4, 17                                       # recess bins (proud, flush, 1, 2+), extent bins (4 x 4 + none)
NBR = np.array([(a, b, c) for a in range(-3, 4) for b in (-1, 0, 1) for c in (-1, 0, 1) if (a, b, c) != (0, 0, 0)])
K_NN = 32


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
                cats=np.unique(cat, return_inverse=True)[1], made=made, nsets=nsets, twin=twin, box=np.maximum(box, 1))


def recess(want: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """(H, X, Z) how many cells each cell sits behind the median first-hit depth of the 9 x 5 (plates x studs; 5 x 5
    from above) around it, seen from the side its normal's main axis faces: 0 flush, > 0 set back, < 0 proud."""
    from scipy.ndimage import median_filter
    n = normal.reshape(*want.shape, 3).astype(np.float32)
    ax = np.abs(n).argmax(-1); pos = np.take_along_axis(n, ax[..., None], -1)[..., 0] > 0
    out = np.zeros(want.shape, np.int16)
    idx = np.indices(want.shape)
    for a in range(3):
        w = np.moveaxis(want, a, 0); L = w.shape[0]
        for up in (True, False):
            hit = w[::-1] if up else w
            D = np.where(w.any(0), np.argmax(hit, 0), L)            # first occupied depth from that side
            med = median_filter(D, size=(5, 5) if a == 0 else (9, 5), mode="nearest")
            q = (ax == a) & (pos == up)
            i = idx[a][q]; plane = tuple(np.delete(idx, a, 0)[:, q])
            out[q] = ((L - 1 - i) if up else i) - med[plane]
    return out


def extent(r: dict) -> np.ndarray:
    """(H, X, Z, 3) the box of the colour area holding each surface cell: height (plates), long, short side (studs);
    0 outside every area (grid_regions colour-only candidates, partition)."""
    from grid_regions import partition, regions
    cells, cands = regions(r, 0.0, 0.0)
    E = np.zeros((*r["want"].shape, 3), np.int16)
    for c in partition(cells, cands):
        q = cells[c]; h, w, d = q.max(0) - q.min(0) + 1
        E[tuple(q.T)] = (h, max(w, d), min(w, d))
    return E


def geometry(f: Path, r: dict) -> tuple[np.ndarray, np.ndarray]:
    """recess and extent of a cell_bench file, cached beside it (<f>.geo.npz)."""
    g = f.with_suffix(".geo.npz")
    if g.exists():
        z = np.load(g); return z["rec"], z["ext"]
    rec, ext = recess(r["want"], r["normal"]), extent(r)
    np.savez_compressed(g, rec=rec, ext=ext)
    return rec, ext


def surface_rows(r: dict, P: dict, lab: np.ndarray | None = None, geo: tuple | None = None) -> dict:
    """A target's surface cells (wanted, visible) with each one's keys (band, colour, recess, extent), its neighbour
    pairs and its true part `lab` (-1 none; default none): what the features without the picture need."""
    W = r["want"]; H = W.shape[0]
    r["normal"] = r["normal"].reshape(*W.shape, 3)
    c = np.argwhere(W & r["visible"]); at = tuple(c.T)
    lab = np.full(len(c), -1) if lab is None else lab
    nrm = r["normal"].astype(np.float32)[at]
    hb = np.minimum((c[:, 0] / max(1, H - 1) * HB).astype(int), HB - 1)
    fb = np.digitize(nrm[:, 0], [-0.8, -0.3, 0.3, 0.8])           # down .. up (normal's y)
    d = dict(cells=c, lab=lab, band_all=hb * FB + fb, colour_all=r["colour"][at])
    if geo is not None:
        rec, E = geo[0][at], geo[1][at]
        eb = np.where(E[:, 0] > 0, np.digitize(E[:, 0], [3, 6, 10]) * 4 + np.digitize(E[:, 1], [2, 3, 5]), NEXT - 1)
        d.update(rec_all=fb * NREC + np.digitize(rec, [0, 1, 2]), ext_all=fb * NEXT + eb, box_all=E)
    lut = np.full(W.shape, -1); lut[at] = np.arange(len(c))
    I, J = [], []
    for o in NBR:                                                    # neighbour pairs (i, j), both surface cells
        q = c + o; inb = ((q >= 0) & (q < W.shape)).all(1)
        j = np.full(len(c), -1); j[inb] = lut[tuple(q[inb].T)]
        I.append(np.flatnonzero(j >= 0)); J.append(j[j >= 0])
    d["nbr"] = (I := np.concatenate(I), J := np.concatenate(J))
    a_, b_ = lab[I], lab[J]; q = (a_ >= 0) & (b_ >= 0)
    d["pairlab"] = np.unique(a_[q].astype(np.int64) * len(P["ld"]) + b_[q], return_counts=True)   # for A, kept small
    return d


def cell_rows(r: dict, ev: dict | None, P: dict, n_max: int, rng, areas: dict | None = None,
              geo: tuple | None = None, patch: dict | None = None) -> dict | None:
    """One set's surface cells (surface_rows) with their true parts; and `sel`, n_max labelled cells sampled for
    fitting and scoring, with their picture evidence (a generated set's: uv in r, ev), regions (areas) and patches."""
    c = np.argwhere(r["want"] & r["visible"]); at = tuple(c.T)
    inst = r["inst"][at]; ok = (inst >= 0) & (r["share"][at] >= 0.5)
    lab = np.array([P["idx"].get(p, -1) for p in np.where(ok, r["inst_part"][np.maximum(inst, 0)], "")])
    if not (lab >= 0).any():
        return None
    d = surface_rows(r, P, lab, geo)
    sel = rng.permutation(np.flatnonzero(lab >= 0))[:n_max]
    d.update(sel=sel, true=lab[sel], band=d["band_all"][sel], colour=d["colour_all"][sel])
    n = len(sel); cs = c[sel]
    d.update(tile=np.zeros((n, P["look"].shape[1]), np.float32), seen=np.zeros(n, bool),
             pairs=np.zeros((0, 2), int), reg=np.zeros((0, P["look"].shape[1]), np.float32))
    if "uv" not in r or ev is None:
        return d
    uv_all = r["uv"][at].astype(float); seen_all = (r["seen"][at].astype(float) >= 0.3) & (uv_all[:, 0] >= 0)
    uv = uv_all[sel]; d["seen"] = seen_all[sel]
    box, sc, te = ev["tile_box"], ev["tile_scale"], unit(ev["tile_emb"])
    emb = np.zeros((n, te.shape[1]), np.float32)
    for k_ in np.unique(sc):                                  # each scale: the tile whose centre is nearest the pixel
        k = np.flatnonzero(sc == k_); cen = (box[k, :2] + box[k, 2:]) / 2
        j = k[np.argmin(((uv[:, None] - cen[None]) ** 2).sum(-1), 1)]
        emb += te[j]
    d["tile"] = unit(emb)
    if areas is not None:                                    # the cell's regions the picture shows (region_evidence.py)
        lk = {tuple(x): i for i, x in enumerate(cs)}
        row = np.array([lk.get(tuple(x), -1) for x in areas["cells"]])
        m = areas["member"]; m = m[(row[m[:, 0]] >= 0) & areas["seen"][m[:, 1]]]
        u, inv = np.unique(m[:, 1], return_inverse=True)
        d["pairs"], d["reg"] = np.stack([row[m[:, 0]], inv.ravel()], 1), unit(areas["emb"][u].astype(np.float32))
    if patch is not None:                                    # each seen cell's DINOv2 patch
        g = patch["feat"].shape[0]
        rc = np.clip(((uv_all[:, ::-1] + patch["off"][::-1]) * g / patch["side"]).astype(int), 0, g - 1)
        d["patch_all"] = np.where(seen_all[:, None], patch["feat"][rc[:, 0], rc[:, 1]], 0).astype(np.float16)
        d["pseen_all"] = seen_all
    return d


def tables(ds: list, P: dict) -> dict:
    """Count tables from every labelled cell of the sets `ds`: log P(part | key) and log P(category | key) for the
    band, recess and extent keys (counts + 1); A = P(neighbour's part | part) (counts + 5 x the neighbour marginal)."""
    m, nc = len(P["ld"]), P["cats"].max() + 1
    T = {}
    for k, nk in (("band", HB * FB), ("rec", FB * NREC), ("ext", FB * NEXT)):
        cp = np.ones((nk, m)); cc = np.ones((nk, nc))
        for d in ds:
            if f"{k}_all" in d:
                q = d["lab"] >= 0; key, t = d[f"{k}_all"][q], d["lab"][q]
                np.add.at(cp, (key, t), 1); np.add.at(cc, (key, P["cats"][t]), 1)
        T[k] = (np.log(cp / cp.sum(1, keepdims=True)), np.log(cc / cc.sum(1, keepdims=True)))
    C = np.zeros(m * m, np.float32)
    for d in ds:
        np.add.at(C, *d["pairlab"])
    C = C.reshape(m, m)
    marg = C.sum(0) + 1; marg /= marg.sum()
    T["A"] = (C + 5 * marg) / (C.sum(1, keepdims=True) + 5)
    return T


def base_feats(d: dict, P: dict, T: dict, rows: np.ndarray) -> np.ndarray:
    """(len(rows), m, len(BASE)) the features without the picture, of cells `rows` (indices into d["cells"])."""
    m = len(P["ld"]); F = np.zeros((len(rows), m, len(BASE)), np.float32)
    F[..., 0] = T["band"][0][d["band_all"][rows]]
    F[..., 1] = T["band"][1][d["band_all"][rows]][:, P["cats"]]
    F[..., 2] = np.log(P["nsets"])[None]
    F[..., 3] = P["made"][:, np.maximum(d["colour_all"][rows], 0)].T
    if "rec_all" in d:
        F[..., 4] = T["rec"][0][d["rec_all"][rows]]; F[..., 5] = T["rec"][1][d["rec_all"][rows]][:, P["cats"]]
        F[..., 6] = T["ext"][0][d["ext_all"][rows]]; F[..., 7] = T["ext"][1][d["ext_all"][rows]][:, P["cats"]]
        E = d["box_all"][rows].astype(np.float32)
        F[..., 8] = np.where(E[:, :1] > 0, -np.abs(np.log(np.maximum(E, 1))[:, None] - np.log(P["box"])[None]).sum(-1), 0)
        if "x" in T:
            X = T["x"]
            F[..., 9] = X["band"][0][d["band_all"][rows]]; F[..., 10] = X["rec"][0][d["rec_all"][rows]]
            F[..., 11] = X["ext"][0][d["ext_all"][rows]]
    return F


def context(d: dict, P: dict, T: dict, w1: np.ndarray) -> np.ndarray:
    """(len(sel), m) the ctx feature: p1 = softmax(base_feats . w1) over every surface cell, then per sampled cell
    log of the mean over its neighbours of p1 @ A."""
    import scipy.sparse as sp
    n, m = len(d["cells"]), len(P["ld"])
    p1 = np.empty((n, m), np.float32)
    for s0 in range(0, n, 1000):
        rows = np.arange(s0, min(n, s0 + 1000))
        F = base_feats(d, P, T, rows); F -= F.mean(1, keepdims=True)
        z = F @ w1; z = np.exp(z - z.max(1, keepdims=True)); p1[rows] = z / z.sum(1, keepdims=True)
    i, j = d["nbr"]
    S = sp.csr_matrix((np.ones(len(i), np.float32), (i, j)), shape=(n, n))[d["sel"]]
    cnt = np.maximum(np.asarray(S.sum(1)).ravel(), 1)
    return np.log((S @ (p1 @ T["A"])) / cnt[:, None] + 1e-4)


def bank_of(ds: list) -> tuple:
    """The labelled seen cells of sets `ds` with their DINOv2 patches: (vectors, part, set id)."""
    V, L, S = [], [], []
    for k, d in enumerate(ds):
        if "patch_all" in d:
            q = d["pseen_all"] & (d["lab"] >= 0)
            V.append(d["patch_all"][q]); L.append(d["lab"][q]); S.append(np.full(q.sum(), id(d)))
    if not V:
        return None
    return np.concatenate(V).astype(np.float32), np.concatenate(L), np.concatenate(S)


def knn(d: dict, P: dict, bank: tuple | None) -> np.ndarray:
    """(len(sel), m) the knn feature: log share of the K_NN most alike bank cells (not of this set) holding each part."""
    out = np.zeros((len(d["sel"]), len(P["ld"])), np.float32)
    if bank is None or "patch_all" not in d:
        return out
    q = d["pseen_all"][d["sel"]]
    if not q.any():
        return out
    V, L, S = bank
    sim = d["patch_all"][d["sel"][q]].astype(np.float32) @ V.T
    sim[:, S == id(d)] = -np.inf
    top = np.argpartition(-sim, K_NN, 1)[:, :K_NN]
    votes = np.zeros((q.sum(), len(P["ld"])), np.float32)
    np.add.at(votes, (np.repeat(np.arange(q.sum()), K_NN), L[top].ravel()), 1)
    out[q] = np.log(votes / K_NN + 1e-3)
    return out


def features(d: dict, ev: dict | None, P: dict, prior_p: np.ndarray, prior_c: np.ndarray, T: dict | None = None,
             ctx: np.ndarray | None = None, kn: np.ndarray | None = None) -> np.ndarray:
    """(n_cells, n_parts, len(FEATS)) for one set; with the count tables T the band priors are T's and the geometry
    features are filled, ctx / kn their features (else 0)."""
    n, m = len(d["true"]), len(P["ld"])
    F = np.zeros((n, m, len(FEATS)), np.float32)
    sw = d["seen"][:, None].astype(np.float32)
    F[..., 0] = d["tile"] @ P["look"].T * sw
    F[..., 1] = d["tile"] @ P["desc"].T * sw
    names = unit(ev["name_emb"]) if ev is not None and len(ev["names"]) else np.zeros((0, P["ph_vec"].shape[1]))
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
    if "pairs" in d and len(d["pairs"]):                            # each part: the cell's best region
        u = d["reg"]
        rs = np.zeros((len(u), m, 4), np.float32)
        rs[..., 0] = u @ P["look"].T
        ps = u @ P["ph_vec"].T * P["idf"][None]
        for i in range(len(u)):
            np.maximum.at(rs[i, :, 1], P["ph_part"], ps[i])
        import semantic_labels as S
        S.MODEL = "google/siglip2-so400m-patch14-384"                  # the model the regions were embedded with
        cp = S.class_probs(u)                                         # (regions, classes)
        ccat = np.array([c or "" for _, _, c in S.CLASSES])
        lp = {c: np.log(cp[:, ccat == c].sum(1) + 1e-6) for c in set(ccat)}
        rs[..., 2] = np.array([lp[x] if x in lp and x else lp[""] for x in P["cat"]]).T
        th = json.loads(S.THRESHOLDS.read_text())[f"{S.MODEL}|views3|dim0.4"]["thresholds"]   # region_evidence's crops: dim 0.4
        for c, v in th.items():                                       # shortcut: categories without a fitted threshold get no keep
            if v is not None and c in lp:
                rs[:, P["cat"] == c, 3] = (np.exp(lp[c]) >= v)[:, None]
        best = np.full((n, m, 4), -np.inf, np.float32)
        np.maximum.at(best, d["pairs"][:, 0], rs[d["pairs"][:, 1]])
        q = np.isfinite(best[:, 0, 0])
        F[q, :, 8:12] = best[q]
    if T is not None:
        B = base_feats(d, P, T, d["sel"])
        F[..., 4:8] = B[..., :4]; F[..., 12:17] = B[..., 4:9]; F[..., 19:22] = B[..., 9:]
    if ctx is not None:
        F[..., 17] = ctx
    if kn is not None:
        F[..., 18] = kn
    F -= F.mean(1, keepdims=True)                                   # centred over parts: no hub part
    return F


def fit(Fs: list, ts: list, iters: int = 200, twin: np.ndarray | None = None) -> np.ndarray:
    """Weights of a softmax over parts (score = F . w) maximising the log-probability of the true part, or with
    `twin` (m, m) of any of its substitutes (the hits the benchmark counts): L-BFGS on the held-in cells."""
    from scipy.optimize import minimize
    F = np.concatenate(Fs); t = np.concatenate(ts)
    T = np.eye(F.shape[1], dtype=bool)[t] if twin is None else twin[t]          # (n, m) parts that count

    def f(w):
        s = F @ w; mx = s.max(1, keepdims=True); e = np.exp(s - mx); p = e / e.sum(1, keepdims=True)
        q = p * T; pt = q.sum(1)
        ll = np.log(pt + 1e-30).mean()
        g = (np.einsum("np,npf->nf", q / pt[:, None], F) - np.einsum("np,npf->nf", p, F)).mean(0)
        return -ll + 1e-3 * w @ w, -g + 2e-3 * w
    return minimize(f, np.zeros(F.shape[-1]), jac=True, method="L-BFGS-B", options={"maxiter": iters}).x


def cell_probs(d: dict, P: dict, T: dict, w1, use, w) -> np.ndarray:
    """(n, m) P(part) of every surface cell of d (surface_rows) as a build sees it: no picture, the features without
    it plus ctx (w1 not None / non-empty)."""
    n, D = len(d["cells"]), P["look"].shape[1]
    ctx = context(dict(d, sel=np.arange(n)), P, T, w1) if w1 is not None and len(w1) else None
    ev = dict(names=np.array([]), name_emb=np.zeros((0, D)))
    p = np.empty((n, len(P["ld"])), np.float32)
    for s0 in range(0, n, 500):
        rows = np.arange(s0, min(n, s0 + 500)); k = len(rows)
        q = dict(d, sel=rows, true=np.zeros(k, int), band=d["band_all"][rows], colour=d["colour_all"][rows],
                 tile=np.zeros((k, D), np.float32), seen=np.zeros(k, bool), pairs=np.zeros((0, 2), int))
        z = features(q, ev, P, *T["band"], T, ctx=None if ctx is None else ctx[rows])[..., use] @ w
        z = np.exp(z - z.max(1, keepdims=True)); p[rows] = z / z.sum(1, keepdims=True)
    return p


def train(P: dict, tr: list, ex: list, use: list, twin) -> tuple:
    """Count tables (training sets tr, extra sets ex as their own x_ tables), then (ctx in use) the first-pass
    weights w1, then the weights w: (T, knn bank, w1, w)."""
    base = [FEATS.index(x) for x in BASE]
    T = tables([d for _, d, _ in tr], P)
    if ex:
        T["x"] = tables([d for _, d in ex], P)
        T["A"] = tables([d for _, d, _ in tr] + [d for _, d in ex], P)["A"]     # adjacency: from every model
    bank = bank_of([d for _, d, _ in tr]) if FEATS.index("knn") in use else None
    F = [features(d, ev, P, *T["band"], T, kn=knn(d, P, bank)) for _, d, ev in tr]
    w1 = None
    if FEATS.index("ctx") in use:
        w1 = fit([x[..., base] for x in F], [d["true"] for _, d, _ in tr], twin=twin)
        for x, (_, d, _) in zip(F, tr):
            x[..., FEATS.index("ctx")] = (cx := context(d, P, T, w1)) - cx.mean(1, keepdims=True)
    return T, bank, w1, fit([x[..., use] for x in F], [d["true"] for _, d, _ in tr], twin=twin)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bench", type=Path); ap.add_argument("--cells", type=int, default=300)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--only", default="", help="comma list of features to use (default all)")
    ap.add_argument("--exact", action="store_true", help="fit to the true part only (default: any substitute the benchmark counts)")
    ap.add_argument("--random", action="store_true", help="score parts at random: the chance level of this metric")
    ap.add_argument("--save", type=Path, default=None, help="also fit on every set and write the model (priors, weights) here, for cell_labels.py")
    ap.add_argument("--extra", default="", help="glob of more cell_bench files for the count tables only (e.g. '*.ldeg.npz')")
    a = ap.parse_args()
    t0 = time.time()
    P = parts_table(); rng = np.random.default_rng(0)
    load = lambda f: dict(np.load(f, allow_pickle=True))
    sets = []
    for f in sorted(a.bench.glob("[!.]*.gen.npz")):
        ev = a.bench / f.name.replace(".gen.npz", ".evidence.npz")
        r = load(f)
        if "uv" not in r or not ev.exists():
            continue
        side = lambda x: a.bench / f.name.replace(".gen.npz", x)
        ar = load(side(".regions.npz")) if side(".regions.npz").exists() else None
        pa = load(side(".patch.npz")) if side(".patch.npz").exists() else None
        ev = load(ev); d = cell_rows(r, ev, P, a.cells, rng, ar, geometry(f, r), pa)
        if d is not None:
            sets.append((f.name.split(".")[0], d, ev))
    extra = []
    for f in sorted(a.bench.glob(a.extra)) if a.extra else []:
        r = load(f); d = cell_rows(r, None, P, 0, rng, geo=geometry(f, r))
        if d is not None:
            del d["nbr"], d["cells"]                                 # tables need only the keys and pairlab
            extra.append((f.name.split(".")[0], d))
    print(f"{len(sets)} sets, {sum(len(d['true']) for _, d, _ in sets)} cells, {len(extra)} extra sets "
          f"({sum((d['lab'] >= 0).sum() for _, d in extra)} cells), {len(P['ld'])} parts; load {time.time() - t0:.0f} s", flush=True)
    use = [FEATS.index(x) for x in a.only.split(",")] if a.only else list(range(len(FEATS)))
    family = lambda n: n.split("_")[0].split("-")[0]               # a model's cars / variants: one fold
    fam = [family(n) for n, _, _ in sets]
    ids = {f: i for i, f in enumerate(dict.fromkeys(fam))}
    fold = np.array([ids[f] % a.folds for f in fam])
    twin = None if a.exact else P["twin"]

    def score(d, ev, T, bank, w1, w):
        return features(d, ev, P, *T["band"], T, ctx=None if w1 is None else context(d, P, T, w1), kn=knn(d, P, bank))[..., use] @ w

    ranks, cats, seen_all, W = [], [], [], []
    for k in range(a.folds):
        tr = [s for s, q in zip(sets, fold) if q != k]; te = [s for s, q in zip(sets, fold) if q == k]
        tf = {family(n) for n, _, _ in te}
        T, bank, w1, w = train(P, tr, [x for x in extra if family(x[0]) not in tf], use, twin); W.append(w)
        for _, d, ev in te:
            s = score(d, ev, T, bank, w1, w)
            if a.random:
                s = rng.random(s.shape)
            order = np.argsort(-s, 1)
            hit = P["twin"][d["true"][:, None], order]               # (n, m): rank positions holding a substitute
            ranks.append(np.argmax(hit, 1)); cats.append(P["cat"][d["true"]]); seen_all.append(d["seen"])
        print(f"fold {k}: {time.time() - t0:.0f} s", flush=True)
    r, c, sn = np.concatenate(ranks), np.concatenate(cats), np.concatenate(seen_all)
    from part_match_bench import BASIC
    sp = ~np.isin(c, list(BASIC))
    print("weights (mean over folds): " + ", ".join(f"{FEATS[i]} {x:+.2f}" for i, x in zip(use, np.mean(W, 0))))
    print(f"{'cells':16s} {'n':>6s} {'top1':>6s} {'top5':>6s} {'top20':>6s} {'top50':>6s}")
    for name, sel in (("all", np.ones(len(r), bool)), ("plain", ~sp), ("special", sp), ("special seen", sp & sn),
                      ("special unseen", sp & ~sn)):
        q = r[sel]
        print(f"{name:16s} {sel.sum():6d} " + " ".join(f"{(q < k).mean():6.3f}" for k in (1, 5, 20, 50)))
    for cname in sorted(set(c[sp])):                                 # special categories with enough cells
        sel = c == cname
        if sel.sum() >= 40:
            print(f"  {cname[:24]:24s} {sel.sum():5d} " + " ".join(f"{(r[sel] < k).mean():6.3f}" for k in (1, 5, 20, 50)))
    if a.save:                                                      # the model from all sets, for building
        T, bank, w1, w = train(P, sets, extra, use, twin)
        np.savez_compressed(a.save, prior_p=T["band"][0], prior_c=T["band"][1], w=w, use=np.array(use), ld=P["ld"],
                            T=np.array(T, dtype=object), w1=w1 if w1 is not None else np.zeros(0))
        print(f"model -> {a.save}: " + ", ".join(f"{FEATS[i]} {x:+.2f}" for i, x in zip(use, w)))
    print(f"total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
