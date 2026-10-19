"""Realism GNN: a 2-layer message-passing classifier over the placement
graph.  Node = part, edge = physical adjacency.  Replaces realism.py's
hand-crafted histograms with learned graph features so SNOT additions
and structural degradations have a chance to shift the signal.

    python tools/learn/realism_gnn.py build   # extract per-build graphs
    python tools/learn/realism_gnn.py train   # train GNN
    python tools/learn/realism_gnn.py score path/to/model.ldr

Dataset reuses realism.py's data sources (real / ours / degraded /
ours_from_omr) but stores graphs instead of 542-dim histograms."""
from __future__ import annotations

import argparse
import os
import pickle
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "tools" / "set_study"))

DATA = Path(os.environ.get("LEGO_DATA", "/Volumes/PRO-G40/lego_data"))
OUT = DATA / "analysis" / "realism_gnn"
GRAPHS_PKL = OUT / "graphs.pkl"
MODEL_PT = OUT / "model.pt"

from realism import (CATEGORIES, COLOUR_BINS, _category, _colour_bin, _size_bin,
                     _is_snot, _parts_with_grid, _neighbour_pairs,
                     _building_omr_ids, _ours_meshes, _generate_ours,
                     _generate_ours_from_omr, _degrade, OMR)

NODE_DIM = len(CATEGORIES) + 5 + (len(COLOUR_BINS) + 1) + 4         # cat + size + colour (+other) + (y, snot, w, d)
EDGE_DIM = 3 + 3                                                      # direction one-hot + normalized offset


def build_graph(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Returns (node_features N×NODE_DIM, edge_index 2×E, edge_attr E×EDGE_DIM)
    or None on parse failure.  Edges are undirected (both directions)."""
    from ldraw_read import flatten, part_bbox
    try:
        parts = flatten(path)
    except Exception:
        return None
    if not parts:
        return None
    bbox_cache = {}
    for p in parts:
        if p.name not in bbox_cache:
            try: bbox_cache[p.name] = part_bbox(p.name)
            except Exception: bbox_cache[p.name] = None
    items = _parts_with_grid(parts, bbox_cache)
    if not items:
        return None
    if len(items) > 2500:
        items = items[:2500]
    # Node features
    y_coords = [it[3][1] for it in items]
    y_min, y_max = min(y_coords), max(y_coords)
    y_rng = max(1.0, y_max - y_min)
    N = len(items)
    nodes = np.zeros((N, NODE_DIM), dtype=np.float32)
    cat_bins = len(CATEGORIES)
    col_bins = len(COLOUR_BINS) + 1
    for i, (ldraw, colour, M, glo, ghi) in enumerate(items):
        nodes[i, _category(ldraw)] = 1.0                        # category one-hot
        sz = _size_bin(ldraw, bbox_cache)
        nodes[i, cat_bins + sz] = 1.0                           # size one-hot
        nodes[i, cat_bins + 5 + _colour_bin(colour)] = 1.0      # colour one-hot
        off = cat_bins + 5 + col_bins
        nodes[i, off] = (glo[1] - y_min) / y_rng                # normalized y
        nodes[i, off + 1] = 1.0 if _is_snot(M) else 0.0         # is_snot
        nodes[i, off + 2] = (ghi[0] - glo[0]) / 10.0            # width proxy
        nodes[i, off + 3] = (ghi[2] - glo[2]) / 10.0            # depth proxy
    # Edges
    pairs = _neighbour_pairs(items)
    E = len(pairs) * 2
    edge_index = np.zeros((2, E), dtype=np.int64)
    edge_attr = np.zeros((E, EDGE_DIM), dtype=np.float32)
    DIR_IX = {"up": 0, "down": 1, "level": 2}
    for k, (i, j, d) in enumerate(pairs):
        for side, (a, b) in enumerate(((i, j), (j, i))):
            k2 = 2 * k + side
            edge_index[0, k2] = a
            edge_index[1, k2] = b
            edge_attr[k2, DIR_IX[d]] = 1.0
            # relative offset normalised by typical stud/plate units
            dx = (items[b][3][0] - items[a][3][0]) / 10.0
            dy = (items[b][3][1] - items[a][3][1]) / 10.0
            dz = (items[b][3][2] - items[a][3][2]) / 10.0
            edge_attr[k2, 3] = max(-5, min(5, dx))
            edge_attr[k2, 4] = max(-5, min(5, dy))
            edge_attr[k2, 5] = max(-5, min(5, dz))
    return nodes, edge_index, edge_attr


def cmd_build(a) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    bids = _building_omr_ids()
    omr_all = [f for f in sorted(OMR.glob("*.mpd")) if f.stem.split("_")[0] in bids]
    random.Random(a.seed).shuffle(omr_all)
    omr = omr_all[:a.n_real]
    omr_pool = [f for f in omr_all if f not in set(omr)]

    bucket: dict[str, list] = {"real": [], "ours": [], "deg": [], "ours_omr": []}

    # Real + collect parts for degradation
    sample_parts = []
    print(f"real: parsing {len(omr)} OMR building sets", flush=True)
    for i, f in enumerate(omr):
        g = build_graph(f)
        if g is not None:
            bucket["real"].append(g)
        if len(sample_parts) < a.n_degraded:
            from ldraw_read import flatten
            try:
                sample_parts.append(flatten(f))
            except Exception:
                pass
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(omr)}", flush=True)
    print(f"real: {len(bucket['real'])}", flush=True)

    # Ours
    meshes = _ours_meshes()
    scratch = OUT / "_ours_scratch"
    print(f"ours: {a.n_ours} builds", flush=True)
    for i in range(a.n_ours):
        mesh = meshes[i % len(meshes)]
        seed = a.seed * 1000 + i
        ldr = _generate_ours(mesh, seed, scratch, iters=a.iters)
        if ldr is None: continue
        g = build_graph(ldr)
        if g is not None:
            bucket["ours"].append(g)
    print(f"ours: {len(bucket['ours'])}", flush=True)

    # Degraded
    rng = random.Random(a.seed + 1)
    kinds = ["replace", "colour", "position"]
    print(f"degraded: processing {len(sample_parts)}", flush=True)
    import tempfile
    from everything2lego.io_ldraw import write_ldr                   # need Placement — we don't have it from flatten alone
    # Simpler: write degraded parts back to a temp .ldr by flatten+rewrite
    from ldraw_read import flatten
    for i, parts in enumerate(sample_parts):
        kind = kinds[i % len(kinds)]
        deg_parts = _degrade(parts, kind, rng)
        # write a minimal LDraw file: one type-1 line per part
        lines = []
        for p in deg_parts:
            M = p.M
            x, y, z = M[0, 3], M[1, 3], M[2, 3]
            a_, b_, c_ = M[0, 0], M[0, 1], M[0, 2]
            d_, e_, f_ = M[1, 0], M[1, 1], M[1, 2]
            g_, h_, i_ = M[2, 0], M[2, 1], M[2, 2]
            lines.append(f"1 {p.colour} {x} {y} {z} {a_} {b_} {c_} {d_} {e_} {f_} {g_} {h_} {i_} {p.name}")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ldr", delete=False) as fh:
            fh.write("0 degraded\n" + "\n".join(lines) + "\n")
            tmp = Path(fh.name)
        try:
            g = build_graph(tmp)
        finally:
            tmp.unlink(missing_ok=True)
        if g is not None:
            bucket["deg"].append(g)
    print(f"degraded: {len(bucket['deg'])}", flush=True)

    # Ours-from-OMR
    print(f"ours-from-omr: {a.n_ours_from_omr} builds", flush=True)
    scratch_omr = OUT / "_ours_omr_scratch"
    for i in range(a.n_ours_from_omr):
        src = omr_pool[i % len(omr_pool)] if omr_pool else None
        if src is None: break
        seed = a.seed * 1000 + 500 + i
        ldr = _generate_ours_from_omr(src, seed, scratch_omr, iters=a.iters)
        if ldr is None: continue
        g = build_graph(ldr)
        if g is not None:
            bucket["ours_omr"].append(g)
    print(f"ours-from-omr: {len(bucket['ours_omr'])}", flush=True)

    with GRAPHS_PKL.open("wb") as f:
        pickle.dump(bucket, f, protocol=pickle.HIGHEST_PROTOCOL)
    sizes = {k: len(v) for k, v in bucket.items()}
    print(f"saved -> {GRAPHS_PKL}  classes: {sizes}", flush=True)


def cmd_train(a) -> None:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    with GRAPHS_PKL.open("rb") as f:
        bucket = pickle.load(f)
    # Flatten with labels; label 1 = real, 0 = everything else; keep source tag.
    items = []
    for src in ("real", "ours", "deg", "ours_omr"):
        for g in bucket.get(src, []):
            items.append((g, 1 if src == "real" else 0, src))
    rng = np.random.default_rng(a.seed)
    idx = rng.permutation(len(items))
    val_n = int(len(items) * a.val_frac)
    val, train = [items[i] for i in idx[:val_n]], [items[i] for i in idx[val_n:]]
    print(f"train={len(train)}, val={len(val)} ({sum(1 for _,_,s in train if s=='real')} real train, "
           f"{sum(1 for _,_,s in val if s=='real')} real val)", flush=True)

    device = torch.device(a.device)

    class MP(nn.Module):
        """One message-passing layer: for each edge (i -> j), compute message
        from node_i's embedding + edge features; aggregate (mean) at j; update
        node_j via GRU-ish gated residual."""
        def __init__(self, dim: int):
            super().__init__()
            self.msg = nn.Linear(dim + EDGE_DIM, dim)
            self.upd = nn.Linear(dim * 2, dim)

        def forward(self, x, ei, ea):
            src = ei[0]; dst = ei[1]
            m = torch.relu(self.msg(torch.cat([x[src], ea], dim=-1)))
            agg = torch.zeros_like(x)
            agg.index_add_(0, dst, m)
            # mean: divide by degree
            deg = torch.zeros(x.size(0), device=x.device).scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
            deg = deg.clamp(min=1).unsqueeze(-1)
            agg = agg / deg
            return torch.relu(self.upd(torch.cat([x, agg], dim=-1)))

    class GNN(nn.Module):
        def __init__(self, hidden=32):
            super().__init__()
            self.embed = nn.Linear(NODE_DIM, hidden)
            self.mp1 = MP(hidden)
            self.mp2 = MP(hidden)
            self.head = nn.Sequential(nn.Linear(hidden, 16), nn.ReLU(), nn.Linear(16, 2))

        def forward(self, nodes, ei, ea):
            x = torch.relu(self.embed(nodes))
            x = self.mp1(x, ei, ea)
            x = self.mp2(x, ei, ea)
            graph = x.mean(dim=0)
            return self.head(graph).unsqueeze(0)                # (1, 2)

    net = GNN(hidden=a.hidden).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    best = -1.0
    for ep in range(1, a.epochs + 1):
        net.train()
        rng_t = np.random.default_rng(a.seed + ep)
        order = rng_t.permutation(len(train))
        tr_hit = tr_n = tr_loss = 0.0
        for ix in order:
            (nodes, ei, ea), y, _ = train[ix]
            nodes_t = torch.tensor(nodes, device=device)
            ei_t = torch.tensor(ei, device=device, dtype=torch.long)
            ea_t = torch.tensor(ea, device=device)
            y_t = torch.tensor([y], device=device, dtype=torch.long)
            logits = net(nodes_t, ei_t, ea_t)
            loss = F.cross_entropy(logits, y_t)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item(); tr_n += 1
            tr_hit += int(logits.argmax(-1).item() == y)
        net.eval()
        vl_hit = 0
        per: dict = {}
        with torch.no_grad():
            for (nodes, ei, ea), y, s in val:
                nodes_t = torch.tensor(nodes, device=device)
                ei_t = torch.tensor(ei, device=device, dtype=torch.long)
                ea_t = torch.tensor(ea, device=device)
                pred = int(net(nodes_t, ei_t, ea_t).argmax(-1).item())
                vl_hit += int(pred == y)
                per.setdefault(s, [0, 0])
                per[s][1] += 1
                per[s][0] += int(pred == y)
        va = vl_hit / max(1, len(val))
        tag = "  ".join(f"{k}={h/max(1,t):.3f}" for k, (h, t) in per.items())
        print(f"ep {ep:02d}  tr loss {tr_loss/max(1,tr_n):.4f} acc {tr_hit/max(1,tr_n):.3f}  val acc {va:.3f}  [{tag}]", flush=True)
        if va > best:
            best = va
            import torch as _t
            _t.save({"state_dict": net.state_dict(), "val_acc": va, "epoch": ep,
                      "hidden": a.hidden}, MODEL_PT)
    print(f"best val acc {best:.3f}  saved -> {MODEL_PT}", flush=True)


def cmd_score(a) -> None:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    g = build_graph(a.path)
    if g is None:
        raise SystemExit(f"could not build graph from {a.path}")
    nodes, ei, ea = g
    ckpt = torch.load(MODEL_PT, map_location="cpu", weights_only=True)
    hidden = ckpt["hidden"]

    class MP(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.msg = nn.Linear(dim + EDGE_DIM, dim)
            self.upd = nn.Linear(dim * 2, dim)

        def forward(self, x, ei, ea):
            src = ei[0]; dst = ei[1]
            m = torch.relu(self.msg(torch.cat([x[src], ea], dim=-1)))
            agg = torch.zeros_like(x)
            agg.index_add_(0, dst, m)
            deg = torch.zeros(x.size(0)).scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
            agg = agg / deg.clamp(min=1).unsqueeze(-1)
            return torch.relu(self.upd(torch.cat([x, agg], dim=-1)))

    class GNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Linear(NODE_DIM, hidden)
            self.mp1 = MP(hidden)
            self.mp2 = MP(hidden)
            self.head = nn.Sequential(nn.Linear(hidden, 16), nn.ReLU(), nn.Linear(16, 2))

        def forward(self, nodes, ei, ea):
            x = torch.relu(self.embed(nodes))
            x = self.mp1(x, ei, ea)
            x = self.mp2(x, ei, ea)
            return self.head(x.mean(dim=0)).unsqueeze(0)

    net = GNN(); net.load_state_dict(ckpt["state_dict"]); net.eval()
    with torch.no_grad():
        logits = net(torch.tensor(nodes), torch.tensor(ei, dtype=torch.long), torch.tensor(ea))
        p_real = float(F.softmax(logits, dim=-1)[0, 1].item())
    print(f"{a.path}: realism = {p_real:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--n-real", type=int, default=200)
    b.add_argument("--n-ours", type=int, default=30)
    b.add_argument("--n-degraded", type=int, default=100)
    b.add_argument("--n-ours-from-omr", type=int, default=30)
    b.add_argument("--iters", type=int, default=10000)
    b.add_argument("--seed", type=int, default=0)
    b.set_defaults(func=cmd_build)
    t = sub.add_parser("train")
    t.add_argument("--epochs", type=int, default=40)
    t.add_argument("--lr", type=float, default=5e-4)
    t.add_argument("--hidden", type=int, default=32)
    t.add_argument("--val-frac", type=float, default=0.2)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default="cpu")
    t.set_defaults(func=cmd_train)
    s = sub.add_parser("score")
    s.add_argument("path", type=Path)
    s.set_defaults(func=cmd_score)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
