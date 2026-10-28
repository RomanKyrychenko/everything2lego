"""Mesh2Brick as a refiner (user's design, 2026-10-27): take a finished build, cut out a connected region of ~50 bricks,
regenerate it with Mesh2Brick (the whole build's occupancy as target, the kept bricks as state, the kept neighbours as the
sequence prefix the new bricks attach to), score the whole build with and without the patch (V13 / V14), keep the patch only if it raises it and adds no
connector-graph pieces nor shrinks the largest one (whole_checks).

    BRICKNET_DATA=/Volumes/PRO-G40/lego_data/bricknet .venv/bin/python tools/learn/m2b_refine.py \
        --ldr build.ldr --ckpt model.pt --out $LEGO_DATA/analysis/m2b/refine/run --iters 40

Writes <out>/refined.ldr (+ before/after renders) and log.jsonl (one line per proposal).
shortcut: regions are BFS-connected sets in the parsed connector graph; bricks without any connector edge are never picked."""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "tools" / "set_study"))
from bricknet_shape import boxes_to_grid, part_boxes  # noqa: E402
from m2b_data import _shift_mats, mesh_grid  # noqa: E402
from m2b_gen import NCONN, Catalog, VoxelScene, _rel_pose, box_of, make_edge  # noqa: E402
from m2b_model import Mesh2Brick  # noqa: E402
from m2b_train import Vocab  # noqa: E402

CLS = Path("/Volumes/PRO-G40/lego_data/analysis/vis_classifiers")


def load_build(path: Path):
    """LDR/MPD -> (bricknet Graph rooted with absolute poses, colours).  Non-catalogue / scaled parts are dropped."""
    from bricknet import load_catalog
    from bricknet.data import load_aliases
    from bricknet.graph import parse_ldr
    from brickae_data import SCALE_TOL
    from ldraw_read import flatten
    stems, aliases = set(load_catalog().id_to_stem), load_aliases(); lines = []
    for p in flatten(path):
        st = p.name.removesuffix(".dat")
        if (st not in stems and (st not in aliases or aliases[st][0] not in stems)) or np.abs(np.linalg.norm(p.M[:3, :3], axis=0) - 1).max() > SCALE_TOL: continue
        M = p.M; lines.append(f"1 {p.colour} " + " ".join(f"{v:.6g}" for v in [*M[:3, 3], *M[:3, :3].ravel()]) + f" {st}.dat")
    return parse_ldr("\n".join(lines) + "\n")


def to_ldr(pids, cols, mats, stems, shift) -> str:
    # undo part_boxes' centring shift: it can be a half stud, and the scorers are not invariant to off-grid placement
    mats = _shift_mats(np.asarray(mats), -np.asarray(shift)) if len(mats) else mats
    return "".join(f"1 {c} " + " ".join(f"{v:.6g}" for v in [*M[:3, 3], *M[:3, :3].ravel()]) + f" {stems[p]}.dat\n" for p, c, M in zip(pids, cols, mats))


class Scorer:
    """V13 on a local set of bricks: mean P(real) over k random connected subgraphs (fixed seed -> comparable)."""

    def __init__(self, name="v13", k=24):
        from realism_gnn import _part_id_map
        from v8_score import load_v8
        self.model = load_v8(CLS / f"{name}_model.pt"); self.pm = _part_id_map(); self.bbox = {}; self.k = k   # pm maps BrickNet names too

    def __call__(self, ldr_text: str, seed: int) -> float:
        from brickae_data import load_items
        from ldraw_read import part_bbox
        from v8_score import score_items
        with tempfile.NamedTemporaryFile("w", suffix=".ldr", delete=False) as f: f.write(ldr_text)
        items = load_items(Path(f.name)); Path(f.name).unlink()
        for it in items:
            if it[0] not in self.bbox:
                try: self.bbox[it[0]] = part_bbox(it[0])
                except Exception: self.bbox[it[0]] = None                              # noqa: BLE001
        if len(items) < 12: return float("nan")
        s = score_items(self.model, items, self.bbox, self.pm, k=self.k, target_n=min(50, max(10, int(0.6 * len(items)))), seed=seed)
        return float("nan") if s == 0.5 else s


@torch.no_grad()
def regenerate(model, vocab, cat, target, state, scene, prefix, hole, colours_of, grid, k_parts=8, retries=6, max_new=120, rng=None):
    """Grow new bricks attached to `prefix` (list of (pid, colour, M, box)) until `hole` is covered or nothing fits.
    Returns list of (pid, colour, M, box)."""
    H, W, D = grid; off = np.array([W / 2, 0, D / 2], np.float32); dev = torch.device("cpu"); rng = rng or np.random.default_rng(0)
    from scipy.ndimage import binary_dilation
    dil = binary_dilation(target, iterations=1)
    vol = torch.from_numpy(np.stack([target, state]).astype(np.float32))[None]; Mv = model.memory(vol)
    seq = list(prefix); new = []; built = np.zeros(grid, bool); used = set(); why = {"pose": 0, "outside": 0, "collision": 0, "nogain": 0, "nomate": 0}

    def pick(logits, mask=None, greedy=True):
        l = logits.clone()
        if mask is not None: l[~torch.as_tensor(mask)] = float("-inf")
        if not torch.isfinite(l).any(): return None
        if greedy: return int(l.argmax())
        top = torch.topk(l, min(4, int(torch.isfinite(l).sum()))); p = torch.softmax(top.values, 0).numpy()
        return int(top.indices[rng.choice(len(p), p=p / p.sum())])

    while len(new) < max_new and (hole & ~built).sum() > 0:
        T = len(seq); b = np.stack([x[3] for x in seq])
        tok, _ = model.tokens(torch.tensor([[x[0] for x in seq]]), torch.tensor([[vocab.col_idx.get(x[1], 0) for x in seq]]),
                              torch.tensor((b[:, 0] + b[:, 1]) / 2)[None], torch.tensor(b[:, 1] - b[:, 0])[None], torch.tensor(np.stack([x[2] for x in seq])[:, :3, :3], dtype=torch.float32)[None])
        c = torch.cat([torch.zeros(1, 1, 3), torch.tensor((b[:, 0] + b[:, 1]) / 2, dtype=torch.float32)[None]], 1)
        anchors = torch.stack([c[..., 1], c[..., 0], c[..., 2]], -1).div(2).floor().long()
        s = model.decode(tok, Mv, anchors, torch.cat([torch.zeros(1, 1, dtype=torch.bool), torch.ones(1, T, dtype=torch.bool)], 1), torch.zeros(1, T, dtype=torch.bool))
        h = s[0, T]; ptr = (model.ptr_q(h) @ model.ptr_k(s[0, 1:T + 1]).T) / model.d ** 0.5
        empty = hole & ~built; front = np.array([bool((boxes_to_grid(np.stack([x[3][0] - off - 1, x[3][1] - off + 1])[None], grid) & empty).any()) for x in seq])
        if not front.any(): break                                                       # nothing borders the empty region
        edil = binary_dilation(empty, iterations=1)

        def near_empty(pid, Mb):
            fr_ = cat.info(pid).frames[:NCONN, :3, 3] @ Mb[:3, :3].T + Mb[:3, 3]           # connector origins, LDU grid frame
            ix = np.floor(np.stack([fr_[:, 1] * -1 / 8, fr_[:, 0] / 20 + W / 2, fr_[:, 2] / 20 + D / 2], 1)).astype(int)
            ok = (ix >= 0).all(1) & (ix < np.array(grid)).all(1); m = np.zeros(NCONN, bool)
            m[np.nonzero(ok)[0]] = edil[ix[ok, 0], ix[ok, 1], ix[ok, 2]]
            return m
        ptr = ptr.masked_fill(~torch.as_tensor(front), float("-inf"))                 # parents must border what is still empty
        best = None
        for attempt in range(retries):
            parent = pick(ptr, greedy=attempt == 0)
            if parent is None: break
            ppid = seq[parent][0]; ctx = torch.cat([h, s[0, 1 + parent]])
            fr = (np.arange(NCONN) < min(len(cat.conns[ppid]), NCONN)) & near_empty(ppid, seq[parent][2])
            for (b_, c_) in used:
                if b_ == parent and c_ < NCONN: fr[c_] = False
            pci = pick(model.h_pconn(ctx), fr, greedy=attempt == 0)
            if pci is None: why["nomate"] += 1; continue
            pc = cat.conns[ppid][pci]; ctx1 = torch.cat([ctx, model.pconn_emb.weight[pci]])
            pl = model.h_part(ctx1); pl[~torch.as_tensor(cat.part_mask(pc))] = float("-inf")
            for part in torch.topk(pl, k_parts).indices.tolist():
                if not torch.isfinite(pl[part]): break
                ctx2 = torch.cat([ctx1, model.part_emb.weight[part]]); cci = pick(model.h_cconn(ctx2), cat.conn_mask(part, pc))
                if cci is None: continue
                cc = cat.conns[part][cci]; g = model.h_geom(torch.cat([ctx2, model.cconn_emb.weight[cci]]))
                yaw = int(g[:36].argmax()) * 10; flip = bool(g[36:38].argmax()); slide = int(g[38:79].argmax()) - 20
                rot = tuple(int(g[79 + 36 * i:79 + 36 * (i + 1)].argmax()) * 10 for i in range(3))
                try: Mq = seq[parent][2] @ _rel_pose(make_edge(0, 1, pc, cc, yaw, flip, slide, rot), ppid, part)
                except Exception: why["pose"] += 1; continue                         # noqa: BLE001
                bx = box_of(vocab.stems[part], Mq, grid); cells = boxes_to_grid(bx[None] - off, grid)
                mc, _ = mesh_grid(np.array([part]), Mq[None], vocab.stems, grid, bx[None] - off); test = mc if mc.sum() else cells
                if test.sum() == 0 or (test & ~dil).sum() / test.sum() > 0.15: why["outside"] += 1; continue
                if scene.check(part, Mq): why["collision"] += 1; continue
                gain = int((cells & hole & ~built).sum())
                if not gain: why["nogain"] += 1
                if gain and (best is None or gain > best[0]): best = (gain, part, Mq, bx, cells, parent, pci, cci)
            if best is not None: break
        if best is None: break
        _, part, Mq, bx, cells, parent, pci, cci = best
        item = (part, colours_of(cells), Mq, bx); seq.append(item); new.append(item); scene.add(part, Mq); built |= cells
        used |= {(parent, pci), (T, cci)}
    regenerate.why = why
    return new, built


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ldr", type=Path, required=True); ap.add_argument("--ckpt", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--iters", type=int, default=40); ap.add_argument("--region", type=int, default=50); ap.add_argument("--prefix", type=int, default=96)
    ap.add_argument("--min-cover", type=float, default=0.85, help="new region must cover this share of the old region's cells")
    ap.add_argument("--margin", type=float, default=0.0); ap.add_argument("--scorer", default="v13", help="scores the whole build; a patch is kept only if it raises it"); ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True); rng = random.Random(a.seed); nrng = np.random.default_rng(a.seed)
    from bricknet import load_catalog
    from bricknet.graph import _adjacency
    i2s = load_catalog().id_to_stem
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False); args = ck["args"]; grid = tuple(ck["grid"]); H, W, D = grid
    vocab = Vocab(); model = Mesh2Brick(vocab.n_part, len(vocab.colors), grid, d=args["d"], layers=args["layers"], C=args["C"]); model.load_state_dict(ck["model"]); model.eval()
    cat = Catalog(vocab.n_part); scorer = Scorer(a.scorer, k=20)
    whole = lambda p, c, m, seeds: float(np.nanmean([scorer(to_ldr(p, c, m, i2s, shift), sd) for sd in seeds]))   # noqa: E731
    g = load_build(a.ldr); boxes, shift = part_boxes(g, i2s, mats=g.transforms, return_shift=True)
    pids = list(g.part_ids); cols = list(g.colors); mats = list(_shift_mats(np.asarray(g.transforms), shift)); bxs = list(boxes + np.array([W / 2, 0, D / 2], np.float32))
    off = np.array([W / 2, 0, D / 2], np.float32)
    print(f"build: {len(pids)} parts, {len(g.edges)} connector edges, {len(g.components)} components", flush=True)
    (a.out / "before.ldr").write_text(to_ldr(pids, cols, mats, i2s, shift))
    log = open(a.out / "log.jsonl", "w"); acc = 0; t0 = time.time(); fit = range(a.seeds)
    s_old = whole(pids, cols, mats, fit); print(f"whole-build {a.scorer.upper()} at start: {s_old:.3f}", flush=True)
    from whole_checks import whole_checks_text
    chk = lambda p, c, m: whole_checks_text(to_ldr(p, c, m, i2s, shift))                     # noqa: E731  connector-graph pieces
    c_old = chk(pids, cols, mats); print(f"pieces {c_old['pieces']}  largest {c_old['largest']:.3f}", flush=True)
    for it in range(a.iters):
        from bricknet.core import Graph  # noqa: F401  (graph rebuilt from the current bricks each iteration)
        from bricknet.graph import parse_ldr
        cur = parse_ldr(to_ldr(pids, cols, mats, i2s, shift).replace(".dat\n", ".dat\n")); adj = _adjacency(cur.edges)
        seeds = [i for i in range(len(pids)) if i in adj]
        if not seeds: break
        start = rng.choice(seeds); region, queue = [start], [start]; seen = {start}
        while queue and len(region) < a.region:                                     # BFS -> connected region
            u = queue.pop(0)
            for v, _, _ in adj.get(u, []):
                if v not in seen: seen.add(v); region.append(v); queue.append(v)
        rset = set(region); keep = [i for i in range(len(pids)) if i not in rset]
        bx_r = np.stack([bxs[i] for i in region]); lo, hi = bx_r[:, 0].min(0) - 2, bx_r[:, 1].max(0) + 2
        near = [i for i in keep if (bxs[i][1] > lo).all() and (bxs[i][0] < hi).all()]
        prefix_idx = sorted(near, key=lambda i: (bxs[i][0][1], bxs[i][0][2], bxs[i][0][0]))[-a.prefix:]
        if not prefix_idx: continue
        old_cells = np.zeros(grid, bool)
        for i in region: old_cells |= boxes_to_grid(bxs[i][None] - off, grid)
        state = np.zeros(grid, bool)
        for i in keep: state |= boxes_to_grid(bxs[i][None] - off, grid)
        target = state | old_cells; hole = old_cells & ~state
        colgrid = np.full(grid, -1, np.int64)
        for i in region: colgrid[boxes_to_grid(bxs[i][None] - off, grid)] = cols[i]
        colours_of = lambda cells: int(np.bincount(colgrid[cells & (colgrid >= 0)]).argmax()) if (cells & (colgrid >= 0)).any() else 71   # noqa: E731
        scene = VoxelScene(grid, vocab.stems)
        for i in keep: scene.add(pids[i], mats[i])
        prefix = [(pids[i], cols[i], mats[i], bxs[i]) for i in prefix_idx]
        new, built = regenerate(model, vocab, cat, target, state, scene, prefix, hole, colours_of, grid, rng=nrng)
        cover = (built & hole).sum() / max(1, hole.sum())
        np_, nc, nm = [pids[i] for i in keep] + [x[0] for x in new], [cols[i] for i in keep] + [x[1] for x in new], [mats[i] for i in keep] + [x[2] for x in new]
        s_new = whole(np_, nc, nm, fit) if new and cover >= a.min_cover else float("nan")   # whole build with the patch in place
        ok = s_new == s_new and s_new > s_old + a.margin
        if ok:                                                                          # a patch may not break the model apart
            c_new = chk(np_, nc, nm); ok = c_new["pieces"] <= c_old["pieces"] and c_new["largest"] >= c_old["largest"]
            if ok: c_old = c_new
        if ok:
            acc += 1; pids, cols, mats = np_, nc, nm; bxs = [bxs[i] for i in keep] + [x[3] for x in new]
        rec = {"iter": it, "region": len(region), "new": len(new), "cover": round(float(cover), 3), "old": round(s_old, 4), "new_score": round(s_new, 4), "accepted": bool(ok), "parts": len(pids)}
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(f"[{it}] why {regenerate.why}  hole {int(hole.sum())} cells  prefix {len(prefix)}"); print(f"[{it}] region {len(region)} -> {len(new)} new  cover {cover:.2f}  {a.scorer.upper()} {s_old:.3f} -> {s_new:.3f}  {'ACCEPT' if ok else 'reject'}  ({time.time() - t0:.0f}s)", flush=True)
        if ok: s_old = s_new
    (a.out / "refined.ldr").write_text(to_ldr(pids, cols, mats, i2s, shift))
    print(f"done: {acc}/{a.iters} accepted, {len(pids)} parts", flush=True)
    for name in ("before", "refined"):                                                  # held-out seeds: acceptance can't overfit these
        txt = (a.out / f"{name}.ldr").read_text(); v = [scorer(txt, 1000 + sd) for sd in range(10)]
        print(f"held-out {a.scorer} whole build {name}: {np.nanmean(v):.3f}  {np.round(v, 3).tolist()}", flush=True)
    for name in ("before", "refined"):
        subprocess.run([sys.executable, str(ROOT / "tools" / "render_build.py"), str(a.out / f"{name}.ldr"), str(a.out / f"{name}.png"), "--views", "45,135,225,315", "--top"], check=False)


if __name__ == "__main__":
    main()
