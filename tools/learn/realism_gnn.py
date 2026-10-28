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

# Richer node features (v4): categorical ints + continuous.  Model turns
# each int into a learned embedding, concatenates, feeds to the GNN.
PART_VOCAB_SIZE = 500                                               # top 499 LDraws + "other"
NODE_CAT_DIM = 4                                                     # (part_id, category, size, colour)
NODE_CONT_DIM = 4                                                    # (y, snot, w, d)
PART_INDEX_PATH = DATA / "analysis" / "realism_gnn" / "part_index.json"


def _part_id_map() -> dict:
    """Load (or build) ldraw → part_id map.  Covers the top PART_VOCAB_SIZE-1
    most common LDraws across OMR; everything else maps to 0 ("other")."""
    import json
    if PART_INDEX_PATH.exists():
        return _with_bricknet_aliases({k: int(v) for k, v in json.loads(PART_INDEX_PATH.read_text()).items()})
    # Build by scanning OMR corpus
    from collections import Counter
    sys.path.insert(0, str(ROOT / "tools" / "set_study"))
    from ldraw_read import flatten
    omr_files = sorted(OMR.glob("*.mpd"))[:500]                      # 500 samples is enough for a stable top-N
    counts: Counter = Counter()
    for f in omr_files:
        try:
            parts = flatten(f)
            counts.update(p.name for p in parts)
        except Exception:
            continue
    top = [name for name, _ in counts.most_common(PART_VOCAB_SIZE - 1)]
    m = {"__other__": 0}
    for i, name in enumerate(top, start=1):
        m[name] = i
    PART_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    PART_INDEX_PATH.write_text(json.dumps(m))
    return _with_bricknet_aliases(m)


def _with_bricknet_aliases(m: dict) -> dict:
    """BrickNet's loader renames some LDraw parts (3023 -> 3023b); give each new name its original's id,
    else the scorers see "other" and score those builds higher than the same build in LDraw names."""
    try:
        from bricknet.data import load_aliases
        aliases = load_aliases()
    except Exception:                                                  # noqa: BLE001  (no BRICKNET_DATA: nothing to alias)
        return m
    for src, (dst, _) in aliases.items():
        if f"{src}.dat" in m and f"{dst}.dat" not in m: m[f"{dst}.dat"] = m[f"{src}.dat"]
    return m
EDGE_FAMILIES = ("unknown", "stud", "hinge", "axle", "ball", "fixed")
_FAM_IX = {f: i for i, f in enumerate(EDGE_FAMILIES)}
EDGE_DIM = len(EDGE_FAMILIES) + 3                                     # 6-dim family one-hot + 3-dim offset (STORED)
# v16 augmentation: edge colour-pair one-hot (src_col, dst_col) prepended at
# training/scoring time only.  Stored edge_attr keeps EDGE_DIM=9 shape so
# existing graphs.pkl stays valid; _augment_edges() computes the extra
# columns from node_cat on the fly.
EDGE_COLPAIR_DIM = 2 * (len(COLOUR_BINS) + 1)                         # 62 dims = 2 x 31-bin colour one-hot
EDGE_FULL_DIM = EDGE_DIM + EDGE_COLPAIR_DIM                           # what the MP layer actually sees


def _augment_edges(ea, ei, nc):
    """Return edge_attr augmented with (src_col, dst_col) one-hot — the
    direct 'colour combinations matter' signal.  Pure torch, GPU-friendly."""
    import torch
    E = ea.shape[0]
    if E == 0:
        return ea.new_zeros((0, EDGE_FULL_DIM))
    col_bins = len(COLOUR_BINS) + 1
    src_col = nc[ei[0], 3].clamp(0, col_bins - 1)
    dst_col = nc[ei[1], 3].clamp(0, col_bins - 1)
    src_onehot = torch.zeros(E, col_bins, device=ea.device, dtype=ea.dtype)
    dst_onehot = torch.zeros(E, col_bins, device=ea.device, dtype=ea.dtype)
    src_onehot.scatter_(1, src_col.unsqueeze(1), 1.0)
    dst_onehot.scatter_(1, dst_col.unsqueeze(1), 1.0)
    return torch.cat([ea, src_onehot, dst_onehot], dim=-1)


def _subgraph_bootstrap(graph, target_n: int, rng: random.Random):
    """BFS-sample up to `target_n` connected nodes from `graph`, return a new
    5-tuple with reindexed edges.  Used to equalise graph sizes at training
    time so the classifier can't exploit the real-vs-ours node-count gap.
    If the graph has fewer than target_n nodes, returns the whole graph."""
    if len(graph) == 5:
        nodes, edge_index, edge_attr, node_cat, node_cont = graph
    else:
        nodes, edge_index, edge_attr = graph
        node_cat = node_cont = None
    N = nodes.shape[0]
    if N <= target_n or N <= 10:
        return graph                                               # too small to subsample
    # Build undirected adjacency
    adj: dict = {}
    for k in range(edge_index.shape[1]):
        s, d = int(edge_index[0, k]), int(edge_index[1, k])
        adj.setdefault(s, []).append(d)
    seed = rng.randrange(N)
    kept = [seed]
    kept_set = {seed}
    frontier = [seed]
    while len(kept) < target_n and frontier:
        nxt = []
        for u in frontier:
            for v in adj.get(u, ()):
                if v not in kept_set:
                    kept_set.add(v); kept.append(v); nxt.append(v)
                    if len(kept) >= target_n:
                        break
            if len(kept) >= target_n:
                break
        frontier = nxt
    if len(kept) < 10:
        return graph
    remap = {old: new for new, old in enumerate(kept)}
    kept_arr = np.array(kept, dtype=np.int64)
    new_nodes = nodes[kept_arr]
    new_ei = []
    new_ea = []
    for k in range(edge_index.shape[1]):
        s, d = int(edge_index[0, k]), int(edge_index[1, k])
        if s in kept_set and d in kept_set:
            new_ei.append((remap[s], remap[d]))
            new_ea.append(edge_attr[k])
    if not new_ei:
        return graph                                                # kept but disconnected — fall back
    new_edge_index = np.array(new_ei, dtype=np.int64).T
    new_edge_attr = np.array(new_ea, dtype=np.float32)
    new_cat = node_cat[kept_arr] if node_cat is not None else None
    new_cont = node_cont[kept_arr] if node_cont is not None else None
    if new_cat is not None:
        return new_nodes, new_edge_index, new_edge_attr, new_cat, new_cont
    return new_nodes, new_edge_index, new_edge_attr


def _augment_graph(graph, rng: random.Random):
    """Return an augmented copy: subgraph crop + colour permutation.
    Handles 5-tuple graphs (v4: nodes, edges, edge_attr, node_cat, node_cont)
    and legacy 3-tuples."""
    if len(graph) == 5:
        nodes, edge_index, edge_attr, node_cat, node_cont = graph
    else:
        nodes, edge_index, edge_attr = graph
        node_cat = node_cont = None
    N = nodes.shape[0]
    # (a) subgraph crop
    keep_frac = rng.uniform(0.6, 1.0)
    target_N = max(10, int(N * keep_frac))
    if target_N < N:
        # BFS from random seed over the undirected adjacency
        adj = [[] for _ in range(N)]
        for k in range(edge_index.shape[1]):
            s, d = int(edge_index[0, k]), int(edge_index[1, k])
            adj[s].append(d)
        seed = rng.randrange(N)
        kept = [seed]
        kept_set = {seed}
        frontier = [seed]
        while len(kept) < target_N and frontier:
            next_frontier = []
            for u in frontier:
                for v in adj[u]:
                    if v not in kept_set:
                        kept_set.add(v)
                        kept.append(v)
                        next_frontier.append(v)
                        if len(kept) >= target_N:
                            break
                if len(kept) >= target_N:
                    break
            frontier = next_frontier
        if len(kept) < 10:
            return None
        remap = {old: new for new, old in enumerate(kept)}
        new_nodes = nodes[np.array(kept)].copy()
        new_edges = []
        new_attrs = []
        for k in range(edge_index.shape[1]):
            s, d = int(edge_index[0, k]), int(edge_index[1, k])
            if s in kept_set and d in kept_set:
                new_edges.append((remap[s], remap[d]))
                new_attrs.append(edge_attr[k])
        if not new_edges:
            return None
        new_edge_index = np.array(new_edges, dtype=np.int64).T
        new_edge_attr = np.array(new_attrs, dtype=np.float32)
        kept_arr = np.array(kept, dtype=np.int64)
        new_node_cat = node_cat[kept_arr].copy() if node_cat is not None else None
        new_node_cont = node_cont[kept_arr].copy() if node_cont is not None else None
    else:
        new_nodes = nodes.copy()
        new_edge_index = edge_index.copy()
        new_edge_attr = edge_attr.copy()
        new_node_cat = node_cat.copy() if node_cat is not None else None
        new_node_cont = node_cont.copy() if node_cont is not None else None
    # (b) colour permutation: swap two used colour columns in the node features
    cat_bins = len(CATEGORIES)
    col_off = cat_bins + 5
    col_bins = len(COLOUR_BINS) + 1
    used_colours = [i for i in range(col_bins) if new_nodes[:, col_off + i].any()]
    if len(used_colours) >= 2:
        a, b = rng.sample(used_colours, 2)
        tmp = new_nodes[:, col_off + a].copy()
        new_nodes[:, col_off + a] = new_nodes[:, col_off + b]
        new_nodes[:, col_off + b] = tmp
        if new_node_cat is not None:
            # swap in categorical colour column too
            mask_a = new_node_cat[:, 3] == a
            mask_b = new_node_cat[:, 3] == b
            new_node_cat[mask_a, 3] = b
            new_node_cat[mask_b, 3] = a
    if new_node_cat is not None:
        return new_nodes, new_edge_index, new_edge_attr, new_node_cat, new_node_cont
    return new_nodes, new_edge_index, new_edge_attr


def build_graph(path: Path, part_id_map: dict | None = None) -> tuple | None:
    """Returns (node_features N×NODE_DIM, edge_index 2×E, edge_attr E×EDGE_DIM,
    node_cat N×NODE_CAT_DIM (int64), node_cont N×NODE_CONT_DIM (float32))
    or None on parse failure.  Edges are undirected (both directions).
    First 3 items are the v3 format (back-compat); last 2 are v4 richer
    features (part_id, cat, size, colour as ints; y, snot, w, d as floats)."""
    if part_id_map is None:
        part_id_map = _part_id_map()
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
    return items_to_graph(items, bbox_cache, part_id_map)


def items_to_graph(items: list, bbox_cache: dict, part_map: dict,
                    pairs: list | None = None) -> tuple | None:
    """Post-flatten graph construction, factored out so in-memory callers
    (the refiner) can skip the flatten + _parts_with_grid round-trip and
    build the graph directly from a modified items list.

    `pairs` optional pre-computed neighbour pairs; skips _neighbour_pairs
    (the dominant cost per profile) when caller has an up-to-date cache."""
    if not items:
        return None
    y_coords = [it[3][1] for it in items]
    y_min, y_max = min(y_coords), max(y_coords)
    y_rng = max(1.0, y_max - y_min)
    N = len(items)
    nodes = np.zeros((N, NODE_DIM), dtype=np.float32)
    node_cat = np.zeros((N, NODE_CAT_DIM), dtype=np.int64)
    node_cont = np.zeros((N, NODE_CONT_DIM), dtype=np.float32)
    cat_bins = len(CATEGORIES)
    col_bins = len(COLOUR_BINS) + 1
    for i, (ldraw, colour, M, glo, ghi) in enumerate(items):
        cat_i = _category(ldraw)
        sz_i = _size_bin(ldraw, bbox_cache)
        col_i = _colour_bin(colour)
        pid_i = part_map.get(ldraw, 0)
        nodes[i, cat_i] = 1.0
        nodes[i, cat_bins + sz_i] = 1.0
        nodes[i, cat_bins + 5 + col_i] = 1.0
        off = cat_bins + 5 + col_bins
        y_norm = (glo[1] - y_min) / y_rng
        snot_f = 1.0 if _is_snot(M) else 0.0
        w_p = (ghi[0] - glo[0]) / 10.0
        d_p = (ghi[2] - glo[2]) / 10.0
        nodes[i, off] = y_norm; nodes[i, off + 1] = snot_f
        nodes[i, off + 2] = w_p; nodes[i, off + 3] = d_p
        node_cat[i, 0] = pid_i; node_cat[i, 1] = cat_i
        node_cat[i, 2] = sz_i; node_cat[i, 3] = col_i
        node_cont[i, 0] = y_norm; node_cont[i, 1] = snot_f
        node_cont[i, 2] = w_p; node_cont[i, 3] = d_p
    if pairs is None:
        pairs = _neighbour_pairs(items)
    E = len(pairs) * 2
    edge_index = np.zeros((2, E), dtype=np.int64)
    edge_attr = np.zeros((E, EDGE_DIM), dtype=np.float32)
    fam_off = _FAM_IX["unknown"]
    for k, (i, j, _dir) in enumerate(pairs):
        for side, (a, b) in enumerate(((i, j), (j, i))):
            k2 = 2 * k + side
            edge_index[0, k2] = a
            edge_index[1, k2] = b
            edge_attr[k2, fam_off] = 1.0
            dx = (items[b][3][0] - items[a][3][0]) / 10.0
            dy = (items[b][3][1] - items[a][3][1]) / 10.0
            dz = (items[b][3][2] - items[a][3][2]) / 10.0
            edge_attr[k2, len(EDGE_FAMILIES)] = max(-5, min(5, dx))
            edge_attr[k2, len(EDGE_FAMILIES) + 1] = max(-5, min(5, dy))
            edge_attr[k2, len(EDGE_FAMILIES) + 2] = max(-5, min(5, dz))
    return nodes, edge_index, edge_attr, node_cat, node_cont


def bricknet_graph_to_ours(g_bn, part_id_map: dict | None = None):
    """Hybrid converter: use bricknet.graph_to_ldr to realise positions (since
    BrickNet's Graph stores structure, not absolute transforms), then overlay
    the TYPED connector families (stud/hinge/axle/ball/fixed) from g_bn.edges
    onto our node ordering.  Keeps BrickNet's connector semantics instead of
    rediscovering bbox adjacency."""
    import bricknet
    import tempfile
    from ldraw_read import flatten, part_bbox
    try:
        text = bricknet.graph_to_ldr(g_bn)
    except Exception:
        return None
    if not text or "1 " not in text:
        return None
    with tempfile.NamedTemporaryFile(mode="w", suffix=".ldr", delete=False,
                                       dir=os.environ.get("TMPDIR")) as fh:
        fh.write(text)
        tmp = Path(fh.name)
    try:
        parts = flatten(tmp)
    except Exception:
        parts = None
    finally:
        tmp.unlink(missing_ok=True)
    if not parts:
        return None
    bbox_cache = {}
    for p in parts:
        if p.name not in bbox_cache:
            try: bbox_cache[p.name] = part_bbox(p.name)
            except Exception: bbox_cache[p.name] = None
    # Build nodes the usual way (reuses position / colour / size logic).
    items = _parts_with_grid(parts, bbox_cache)
    if not items:
        return None
    if part_id_map is None:
        part_id_map = _part_id_map()
    N = len(items)
    y_coords = [it[3][1] for it in items]
    y_min, y_max = min(y_coords), max(y_coords)
    y_rng = max(1.0, y_max - y_min)
    nodes = np.zeros((N, NODE_DIM), dtype=np.float32)
    node_cat = np.zeros((N, NODE_CAT_DIM), dtype=np.int64)
    node_cont = np.zeros((N, NODE_CONT_DIM), dtype=np.float32)
    cat_bins = len(CATEGORIES); col_bins = len(COLOUR_BINS) + 1
    for i, (ldraw, colour, M, glo, ghi) in enumerate(items):
        cat_i = _category(ldraw); sz_i = _size_bin(ldraw, bbox_cache); col_i = _colour_bin(colour)
        pid_i = part_id_map.get(ldraw, 0)
        nodes[i, cat_i] = 1.0; nodes[i, cat_bins + sz_i] = 1.0; nodes[i, cat_bins + 5 + col_i] = 1.0
        off = cat_bins + 5 + col_bins
        y_norm = (glo[1] - y_min) / y_rng
        snot_f = 1.0 if _is_snot(M) else 0.0
        w_p = (ghi[0] - glo[0]) / 10.0
        d_p = (ghi[2] - glo[2]) / 10.0
        nodes[i, off] = y_norm; nodes[i, off+1] = snot_f; nodes[i, off+2] = w_p; nodes[i, off+3] = d_p
        node_cat[i, 0] = pid_i; node_cat[i, 1] = cat_i; node_cat[i, 2] = sz_i; node_cat[i, 3] = col_i
        node_cont[i, 0] = y_norm; node_cont[i, 1] = snot_f; node_cont[i, 2] = w_p; node_cont[i, 3] = d_p
    # Edges: typed families from g_bn.edges.  Indices (a, b) in g_bn match the
    # order graph_to_ldr emits parts, which matches our flatten() output when
    # the LDraw has no sub-models (bricknet renders a flat type-1 sequence).
    bn_edges = g_bn.edges
    valid_edges = [(int(e["a"]), int(e["b"]), str(e["family"]))
                    for e in bn_edges
                    if 0 <= int(e["a"]) < N and 0 <= int(e["b"]) < N and int(e["a"]) != int(e["b"])]
    E = len(valid_edges) * 2
    edge_index = np.zeros((2, E), dtype=np.int64)
    edge_attr = np.zeros((E, EDGE_DIM), dtype=np.float32)
    for k, (a, b, fam) in enumerate(valid_edges):
        fam_i = _FAM_IX.get(fam, _FAM_IX["unknown"])
        dx = (items[b][3][0] - items[a][3][0]) / 10.0
        dy = (items[b][3][1] - items[a][3][1]) / 10.0
        dz = (items[b][3][2] - items[a][3][2]) / 10.0
        dx = max(-5, min(5, dx)); dy = max(-5, min(5, dy)); dz = max(-5, min(5, dz))
        for side, (s, d, sign) in enumerate(((a, b, 1.0), (b, a, -1.0))):
            kk = 2 * k + side
            edge_index[0, kk] = s; edge_index[1, kk] = d
            edge_attr[kk, fam_i] = 1.0
            edge_attr[kk, len(EDGE_FAMILIES)] = sign * dx
            edge_attr[kk, len(EDGE_FAMILIES) + 1] = sign * dy
            edge_attr[kk, len(EDGE_FAMILIES) + 2] = sign * dz
    return nodes, edge_index, edge_attr, node_cat, node_cont


def cmd_build(a) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # Broaden "real" beyond kind=building: OMR has 353 building files but 1820
    # total; richer "real" training data stabilises the classifier.  Exclude
    # kind=parts (literal part catalogues, not builds).
    import csv
    exclude_ids: set[str] = set()
    try:
        with (DATA / "analysis" / "set_kinds.csv").open() as f:
            for row in csv.DictReader(f):
                if row.get("kind") == "parts":
                    exclude_ids.add(row["id"])
    except Exception:
        pass
    omr_all = [f for f in sorted(OMR.glob("*.mpd"))
               if f.stem.split("_")[0] not in exclude_ids]
    random.Random(a.seed).shuffle(omr_all)
    omr = omr_all[:a.n_real]
    # ours_from_omr uses the same OMR pool — the sampler output is a NEW
    # build with ours-style brick choices, not a copy of the real set, so
    # training on both the real set and the sampler's output isn't a leak.
    omr_pool = omr_all if omr_all else [f for f in omr_all if f not in set(omr)]

    bucket: dict[str, list] = {"real": [], "ours": [], "deg": [], "ours_omr": []}

    # Real + collect parts for degradation
    sample_parts = []
    aug_rng = random.Random(a.seed + 100)
    print(f"real: parsing {len(omr)} OMR models (+ {a.n_aug} augmentations each)", flush=True)
    for i, f in enumerate(omr):
        g = build_graph(f)
        if g is not None:
            bucket["real"].append(g)
            # Augmentations: each real gets N extra graphs via subgraph crop +
            # colour perm.  Expands effective training data without new builds.
            for _ in range(a.n_aug):
                aug = _augment_graph(g, aug_rng)
                if aug is not None:
                    bucket["real"].append(aug)
        if len(sample_parts) < a.n_degraded:
            from ldraw_read import flatten
            try:
                sample_parts.append(flatten(f))
            except Exception:
                pass
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(omr)}  real so far: {len(bucket['real'])}", flush=True)
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
    # OMR upweighting: OMR graphs are the first N entries of bucket['real']
    # (cmd_build appends OMR first; bricknet_to_graphs.py extends after).
    # Oversample them N× to shift the model's 'real' prior toward OMR
    # architectural patterns and away from BrickNet MOC statistics.
    if a.oversample_omr > 1:
        omr_n = int(a.omr_count)
        omr_subset = bucket["real"][:omr_n]
        before = len(bucket["real"])
        extra = []
        aug_rng = random.Random(a.seed + 1000)
        for _ in range(a.oversample_omr - 1):
            for g in omr_subset:
                ag = _augment_graph(g, aug_rng)                           # subgraph crop + colour perm
                if ag is not None:
                    extra.append(ag)
        bucket["real"] = bucket["real"] + extra
        print(f"oversampled OMR {a.oversample_omr}x (augmented): real {before} -> {len(bucket['real'])}", flush=True)
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
            self.msg = nn.Linear(dim + EDGE_FULL_DIM, dim)
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
        """v4+batching: per-part embedding + category/size/colour + continuous.
        forward accepts a `graph_id` tensor (per-node) for batched mean-pool;
        when graph_id is None, treats input as one graph (back-compat)."""
        def __init__(self, hidden=32, n_layers=2):
            super().__init__()
            self.part_emb = nn.Embedding(PART_VOCAB_SIZE, 32)
            self.cat_emb = nn.Embedding(len(CATEGORIES), 4)
            self.size_emb = nn.Embedding(5, 4)
            self.col_emb = nn.Embedding(len(COLOUR_BINS) + 1, 8)
            embed_dim = 32 + 4 + 4 + 8 + NODE_CONT_DIM
            self.embed = nn.Linear(embed_dim, hidden)
            self.legacy_embed = nn.Linear(NODE_DIM, hidden)
            self.layers = nn.ModuleList([MP(hidden) for _ in range(n_layers)])
            self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(),
                                       nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 2))

        def forward(self, nodes, ei, ea, node_cat=None, node_cont=None,
                     graph_id=None, n_graphs=None):
            if node_cat is not None and node_cont is not None:
                part_e = self.part_emb(node_cat[:, 0])
                cat_e = self.cat_emb(node_cat[:, 1])
                size_e = self.size_emb(node_cat[:, 2])
                col_e = self.col_emb(node_cat[:, 3])
                x = torch.cat([part_e, cat_e, size_e, col_e, node_cont], dim=-1)
                x = torch.relu(self.embed(x))
                ea_full = _augment_edges(ea, ei, node_cat)
            else:
                x = torch.relu(self.legacy_embed(nodes))
                ea_full = torch.cat([ea, ea.new_zeros((ea.shape[0], EDGE_COLPAIR_DIM))], dim=-1)
            for layer in self.layers:
                x = layer(x, ei, ea_full)
            if graph_id is None:
                graph = x.mean(dim=0)
                return self.head(graph).unsqueeze(0)             # (1, 2)
            # Batched: scatter-mean per graph_id
            B = int(n_graphs) if n_graphs is not None else int(graph_id.max().item()) + 1
            sums = torch.zeros(B, x.size(-1), device=x.device)
            sums.index_add_(0, graph_id, x)
            counts = torch.zeros(B, device=x.device).scatter_add_(
                0, graph_id, torch.ones(graph_id.size(0), device=x.device))
            graph_feats = sums / counts.clamp(min=1).unsqueeze(-1)
            return self.head(graph_feats)                        # (B, 2)

    net = GNN(hidden=a.hidden, n_layers=a.layers).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    # Class-weighted cross-entropy to compensate class imbalance.  With
    # augmentation enabled, real can outnumber ours+deg+ours_omr by 10×, so an
    # unweighted classifier learns a prior toward "real" and never-say-"ours"-
    # confidently on single examples.  Weights = inverse frequency.
    n_pos = sum(1 for _, y, _ in train if y == 1)
    n_neg = len(train) - n_pos
    total = max(1, len(train))
    w_pos = total / (2 * max(1, n_pos))
    w_neg = total / (2 * max(1, n_neg))
    class_w = torch.tensor([w_neg, w_pos], device=device, dtype=torch.float32)
    print(f"class weights: not-real={w_neg:.2f}  real={w_pos:.2f}  (counts {n_neg}/{n_pos})", flush=True)

    # Strip typed-family one-hot columns from all edges to prevent a data leak:
    # BrickNet-sourced graphs carry stud/hinge/axle/ball/fixed families, our
    # OMR / ours / deg graphs have "unknown".  A classifier can trivially use
    # the family column to separate classes.  Zero them out so the model
    # must learn from node features + edge offsets only.
    if a.strip_family:
        n_fam = len(EDGE_FAMILIES)
        for src in (train, val):
            for g, _y, _s in src:
                if len(g) >= 3 and g[2] is not None and g[2].shape[0] > 0:
                    g[2][:, :n_fam] = 0.0
        print(f"stripped {n_fam} family columns from edges", flush=True)

    # Pre-convert graphs to torch tensors once.  Each step becomes a concat +
    # offset of pre-made tensors instead of numpy→torch per batch — saves the
    # per-step allocation cost (~30-50% faster).
    def _pre(items):
        out = []
        for g, y, s in items:
            if len(g) == 5:
                nodes, ei, ea, nc, ncont = g
            else:
                nodes, ei, ea = g
                nc = ncont = None
            if ei.shape[1] == 0 or nodes.shape[0] == 0:
                continue
            t = {
                "nodes": torch.tensor(nodes),
                "ei": torch.tensor(ei, dtype=torch.long),
                "ea": torch.tensor(ea),
                "nc": torch.tensor(nc, dtype=torch.long) if nc is not None else None,
                "ncont": torch.tensor(ncont) if ncont is not None else None,
                "N": nodes.shape[0],
                "y": y,
                "s": s,
            }
            out.append(t)
        return out
    if a.pre_convert:
        print(f"pre-converting {len(train) + len(val)} graphs to torch tensors...", flush=True)
        train_t = _pre(train)
        val_t = _pre(val)
        print(f"  train_t={len(train_t)}  val_t={len(val_t)}", flush=True)
    else:
        train_t = val_t = None

    best = -1.0
    for ep in range(1, a.epochs + 1):
        net.train()
        rng_t = np.random.default_rng(a.seed + ep)
        source = train_t if a.pre_convert else train
        order = rng_t.permutation(len(source))
        if a.subset > 0 and a.subset < len(order):
            order = order[:a.subset]                             # subset-per-epoch (5× speed with same convergence)
        tr_hit = tr_n = tr_loss = 0.0
        # Batched training: concat up to a.batch graphs per step with cumulative
        # node-index offsets so edges stay valid.  Massive speedup over per-
        # graph forward+backward on CPU/MPS.
        B = a.batch
        i = 0
        _hb_step = 0
        while i < len(order):
            batch_ix = order[i:i + B]
            i += B
            _hb_step += 1
            if _hb_step % 50 == 0:
                print(f"    ... step {_hb_step}  i={i}/{len(order)}", flush=True)
            all_nodes, all_ei, all_ea, all_nc, all_ncont, all_gid, all_y = [], [], [], [], [], [], []
            n_off = 0
            kept = 0
            if a.pre_convert:
                for idx in batch_ix:
                    t = train_t[idx]
                    all_nodes.append(t["nodes"])
                    all_ei.append(t["ei"] + n_off)
                    all_ea.append(t["ea"])
                    if t["nc"] is not None:
                        all_nc.append(t["nc"]); all_ncont.append(t["ncont"])
                    all_gid.append(torch.full((t["N"],), kept, dtype=torch.long))
                    all_y.append(t["y"])
                    n_off += t["N"]; kept += 1
                if not all_nodes:
                    continue
                nodes_t = torch.cat(all_nodes, dim=0).to(device)
                ei_t = torch.cat(all_ei, dim=1).to(device)
                ea_t = torch.cat(all_ea, dim=0).to(device)
                nc_t = torch.cat(all_nc, dim=0).to(device) if all_nc else None
                ncont_t = torch.cat(all_ncont, dim=0).to(device) if all_ncont else None
                gid_t = torch.cat(all_gid, dim=0).to(device)
                y_t = torch.tensor(all_y, device=device, dtype=torch.long)
            else:
                _bootstrap_rng = random.Random(ep * 1000003 + i)
                for idx in batch_ix:
                    g, y, _ = train[idx]
                    if a.subgraph_nodes > 0:
                        g = _subgraph_bootstrap(g, a.subgraph_nodes, _bootstrap_rng)
                    if len(g) == 5:
                        nodes, ei, ea, nc, ncont = g
                    else:
                        nodes, ei, ea = g; nc = ncont = None
                    if ei.shape[1] == 0 or nodes.shape[0] == 0:
                        continue
                    all_nodes.append(nodes); all_ei.append(ei + n_off); all_ea.append(ea)
                    if nc is not None:
                        all_nc.append(nc); all_ncont.append(ncont)
                    all_gid.append(np.full(nodes.shape[0], kept, dtype=np.int64))
                    all_y.append(y); n_off += nodes.shape[0]; kept += 1
                if not all_nodes:
                    continue
                nodes_t = torch.tensor(np.concatenate(all_nodes, axis=0), device=device)
                ei_t = torch.tensor(np.concatenate(all_ei, axis=1), device=device, dtype=torch.long)
                ea_t = torch.tensor(np.concatenate(all_ea, axis=0), device=device)
                nc_t = torch.tensor(np.concatenate(all_nc, axis=0), device=device, dtype=torch.long) if all_nc else None
                ncont_t = torch.tensor(np.concatenate(all_ncont, axis=0), device=device) if all_ncont else None
                gid_t = torch.tensor(np.concatenate(all_gid, axis=0), device=device, dtype=torch.long)
                y_t = torch.tensor(all_y, device=device, dtype=torch.long)
            if a.permute_palette and nc_t is not None:
                # Per-graph random permutation of the colour column (nc[:, 3]).
                # Forces the model to be invariant to absolute colour identity:
                # it can only use colour through adjacency patterns (which the
                # same permutation preserves) rather than "colour N is real".
                _cb = len(COLOUR_BINS) + 1
                _perm = torch.stack([torch.randperm(_cb, device=device) for _ in range(len(all_y))])
                nc_t = nc_t.clone()
                nc_t[:, 3] = _perm[gid_t, nc_t[:, 3]]
            logits = net(nodes_t, ei_t, ea_t, nc_t, ncont_t, graph_id=gid_t, n_graphs=len(all_y))
            loss = F.cross_entropy(logits, y_t, weight=class_w)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item() * len(all_y); tr_n += len(all_y)
            tr_hit += (logits.argmax(-1) == y_t).sum().item()
        net.eval()
        vl_hit = 0
        per: dict = {}
        with torch.no_grad():
            _val_rng = random.Random(ep * 2003)
            for g, y, s in val:
                if a.subgraph_nodes > 0:
                    g = _subgraph_bootstrap(g, a.subgraph_nodes, _val_rng)
                if len(g) == 5:
                    nodes, ei, ea, nc, ncont = g
                else:
                    nodes, ei, ea = g
                    nc = ncont = None
                if ei.shape[1] == 0 or nodes.shape[0] == 0:
                    continue
                nodes_t = torch.tensor(nodes, device=device)
                ei_t = torch.tensor(ei, device=device, dtype=torch.long)
                ea_t = torch.tensor(ea, device=device)
                nc_t = torch.tensor(nc, device=device, dtype=torch.long) if nc is not None else None
                ncont_t = torch.tensor(ncont, device=device) if ncont is not None else None
                pred = int(net(nodes_t, ei_t, ea_t, nc_t, ncont_t).argmax(-1).item())
                vl_hit += int(pred == y)
                per.setdefault(s, [0, 0])
                per[s][1] += 1
                per[s][0] += int(pred == y)
        va = vl_hit / max(1, len(val))
        tag = "  ".join(f"{k}={h/max(1,t):.3f}" for k, (h, t) in per.items())
        # Confusion for F1 + probs for ROC AUC: positive = real (class 1).
        tp = fp = fn = tn = 0
        probs_pos: list = []    # P(class=real)
        truths: list = []
        with torch.no_grad():
            _val_rng2 = random.Random(ep * 2003 + 1)
            for g, y, s in val:
                if a.subgraph_nodes > 0:
                    g = _subgraph_bootstrap(g, a.subgraph_nodes, _val_rng2)
                if len(g) == 5:
                    nodes, ei, ea, nc, ncont = g
                else:
                    nodes, ei, ea = g
                    nc = ncont = None
                if ei.shape[1] == 0 or nodes.shape[0] == 0:
                    continue
                nodes_t = torch.tensor(nodes, device=device)
                ei_t = torch.tensor(ei, device=device, dtype=torch.long)
                ea_t = torch.tensor(ea, device=device)
                nc_t = torch.tensor(nc, device=device, dtype=torch.long) if nc is not None else None
                ncont_t = torch.tensor(ncont, device=device) if ncont is not None else None
                logits = net(nodes_t, ei_t, ea_t, nc_t, ncont_t)
                p = float(F.softmax(logits, dim=-1)[0, 1].item())
                probs_pos.append(p)
                truths.append(y)
                pred = 1 if p >= 0.5 else 0
                if y == 1 and pred == 1: tp += 1
                elif y == 0 and pred == 1: fp += 1
                elif y == 1 and pred == 0: fn += 1
                else: tn += 1
        prec_pos = tp / max(1, tp + fp)
        rec_pos = tp / max(1, tp + fn)
        f1_pos = 2 * prec_pos * rec_pos / max(1e-9, prec_pos + rec_pos)
        prec_neg = tn / max(1, tn + fn)
        rec_neg = tn / max(1, tn + fp)
        f1_neg = 2 * prec_neg * rec_neg / max(1e-9, prec_neg + rec_neg)
        f1_macro = (f1_pos + f1_neg) / 2
        # ROC AUC: threshold-free discrimination.  Hand-rolled trapezoidal rule
        # over sorted-by-probability pairs — avoids a sklearn dependency.
        auc = 0.0
        if probs_pos and 0 < sum(truths) < len(truths):
            paired = sorted(zip(probs_pos, truths), reverse=True)
            n_pos = sum(truths); n_neg = len(truths) - n_pos
            tp_r = fp_r = 0
            prev_p = None
            prev_tp = prev_fp = 0
            auc_acc = 0.0
            for p, t in paired:
                if prev_p is not None and p != prev_p:
                    auc_acc += (fp_r - prev_fp) * (tp_r + prev_tp) / 2
                    prev_tp = tp_r; prev_fp = fp_r
                if t == 1: tp_r += 1
                else: fp_r += 1
                prev_p = p
            auc_acc += (fp_r - prev_fp) * (tp_r + prev_tp) / 2
            auc = auc_acc / max(1, n_pos * n_neg)
        print(f"ep {ep:02d}  tr loss {tr_loss/max(1,tr_n):.4f} acc {tr_hit/max(1,tr_n):.3f}  "
              f"val acc {va:.3f}  AUC={auc:.3f}  F1_macro={f1_macro:.3f} (real {f1_pos:.3f}, notreal {f1_neg:.3f})  [{tag}]", flush=True)
        if auc > best:
            best = auc
            import torch as _t
            _t.save({"state_dict": net.state_dict(), "val_acc": va, "auc": auc, "f1_macro": f1_macro,
                      "epoch": ep, "hidden": a.hidden, "layers": a.layers}, MODEL_PT)
    print(f"best AUC {best:.3f}  saved -> {MODEL_PT}", flush=True)


def cmd_score(a) -> None:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    g = build_graph(a.path)
    if g is None:
        raise SystemExit(f"could not build graph from {a.path}")
    if len(g) == 5:
        nodes, ei, ea, nc, ncont = g
    else:
        nodes, ei, ea = g
        nc = ncont = None
    ckpt = torch.load(MODEL_PT, map_location="cpu", weights_only=True)
    hidden = ckpt["hidden"]
    n_layers = ckpt.get("layers", 2)

    class MP(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.msg = nn.Linear(dim + EDGE_FULL_DIM, dim)
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
            self.part_emb = nn.Embedding(PART_VOCAB_SIZE, 32)
            self.cat_emb = nn.Embedding(len(CATEGORIES), 4)
            self.size_emb = nn.Embedding(5, 4)
            self.col_emb = nn.Embedding(len(COLOUR_BINS) + 1, 8)
            embed_dim = 32 + 4 + 4 + 8 + NODE_CONT_DIM
            self.embed = nn.Linear(embed_dim, hidden)
            self.legacy_embed = nn.Linear(NODE_DIM, hidden)
            self.layers = nn.ModuleList([MP(hidden) for _ in range(n_layers)])
            self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(),
                                       nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 2))

        def forward(self, nodes, ei, ea, nc=None, ncont=None):
            if nc is not None and ncont is not None:
                part_e = self.part_emb(nc[:, 0])
                cat_e = self.cat_emb(nc[:, 1])
                size_e = self.size_emb(nc[:, 2])
                col_e = self.col_emb(nc[:, 3])
                x = torch.cat([part_e, cat_e, size_e, col_e, ncont], dim=-1)
                x = torch.relu(self.embed(x))
                ea_full = _augment_edges(ea, ei, nc)
            else:
                x = torch.relu(self.legacy_embed(nodes))
                ea_full = torch.cat([ea, ea.new_zeros((ea.shape[0], EDGE_COLPAIR_DIM))], dim=-1)
            for layer in self.layers:
                x = layer(x, ei, ea_full)
            return self.head(x.mean(dim=0)).unsqueeze(0)

    net = GNN(); net.load_state_dict(ckpt["state_dict"]); net.eval()
    # Match training: strip family one-hot and average over K subgraph samples.
    scores = []
    rng = random.Random(0)
    for _ in range(a.samples):
        gs = _subgraph_bootstrap(g, a.subgraph_nodes, rng) if a.subgraph_nodes > 0 else g
        if len(gs) == 5:
            nodes2, ei2, ea2, nc2, ncont2 = gs
        else:
            nodes2, ei2, ea2 = gs
            nc2 = ncont2 = None
        if ea2.shape[0] == 0 or nodes2.shape[0] == 0:
            continue
        if a.strip_family and ea2.shape[1] >= len(EDGE_FAMILIES):
            ea2 = ea2.copy(); ea2[:, :len(EDGE_FAMILIES)] = 0.0
        with torch.no_grad():
            nt = torch.tensor(nodes2)
            ei_t = torch.tensor(ei2, dtype=torch.long)
            ea_t = torch.tensor(ea2)
            nc_t = torch.tensor(nc2, dtype=torch.long) if nc2 is not None else None
            ncont_t = torch.tensor(ncont2) if ncont2 is not None else None
            logits = net(nt, ei_t, ea_t, nc_t, ncont_t)
            scores.append(float(F.softmax(logits, dim=-1)[0, 1].item()))
    if not scores:
        raise SystemExit(f"no valid subgraph samples from {a.path}")
    p_mean = sum(scores) / len(scores)
    print(f"{a.path}: realism = {p_mean:.3f}  (K={len(scores)} subgraph samples, min={min(scores):.3f} max={max(scores):.3f})")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--n-real", type=int, default=200)
    b.add_argument("--n-ours", type=int, default=30)
    b.add_argument("--n-degraded", type=int, default=100)
    b.add_argument("--n-ours-from-omr", type=int, default=30)
    b.add_argument("--n-aug", type=int, default=3, help="N augmented copies per real OMR (subgraph crop + colour swap)")
    b.add_argument("--iters", type=int, default=10000)
    b.add_argument("--seed", type=int, default=0)
    b.set_defaults(func=cmd_build)
    t = sub.add_parser("train")
    t.add_argument("--epochs", type=int, default=40)
    t.add_argument("--lr", type=float, default=5e-4)
    t.add_argument("--hidden", type=int, default=32)
    t.add_argument("--layers", type=int, default=2, help="number of message-passing layers")
    t.add_argument("--batch", type=int, default=32, help="graphs per batched step (1 = unbatched)")
    t.add_argument("--subset", type=int, default=0, help="per-epoch random sample size (0 = full)")
    t.add_argument("--pre-convert", action=argparse.BooleanOptionalAction, default=True, help="pre-convert graphs to torch tensors once at startup (saves ~30-50%% per step)")
    t.add_argument("--strip-family", action=argparse.BooleanOptionalAction, default=True, help="zero out the 6-dim family one-hot in all edge attrs (removes BrickNet-vs-ours trivial shortcut)")
    t.add_argument("--subgraph-nodes", type=int, default=0, help="BFS-sample N connected nodes per training+val step (0 = whole graph).  Removes the node-count class leak so classifier learns from structure within equal-sized subgraphs.")
    t.add_argument("--permute-palette", action=argparse.BooleanOptionalAction, default=True, help="per-graph random palette permutation at training time: forces colour-identity invariance, so the model can only use colour through adjacency patterns.")
    t.add_argument("--oversample-omr", type=int, default=1, help="duplicate the OMR portion of bucket['real'] N times before training, so OMR architectural patterns dominate over BrickNet MOC statistics (1 = no oversampling)")
    t.add_argument("--omr-count", type=int, default=7156, help="number of OMR graphs at the start of bucket['real'] (hardcoded from the build order)")
    t.add_argument("--val-frac", type=float, default=0.2)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default="cpu")
    t.set_defaults(func=cmd_train)
    s = sub.add_parser("score")
    s.add_argument("path", type=Path)
    s.add_argument("--samples", type=int, default=5, help="subgraph samples averaged")
    s.add_argument("--subgraph-nodes", type=int, default=50, help="BFS subgraph size (0 = whole graph)")
    s.add_argument("--strip-family", action=argparse.BooleanOptionalAction, default=True, help="match training's --strip-family")
    s.set_defaults(func=cmd_score)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
