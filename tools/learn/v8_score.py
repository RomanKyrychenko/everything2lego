"""V8 subgraph-GNN scorer, library form.

Score any build (items list from _parts_with_grid) by sampling K random
50-brick subgraphs and averaging per-subgraph P(real).  ~0.12 s per
build on CPU.  Trained with hard negatives from refiner outputs; honest
signal (clean OMR ~1.0, damaged OMR ~0.17, sampler voxel-fill ~0.16).

Model path: /Volumes/PRO-G40/lego_data/analysis/vis_classifiers/v8_model.pt
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Reuse existing helpers.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "set_study"))
from realism import CATEGORIES, COLOUR_BINS, _neighbour_pairs
from realism_gnn import _part_id_map, PART_VOCAB_SIZE
from subgraph_extract import extract_subgraph, nodes_edges_to_tensors      # type: ignore

MODEL_PT = Path("/Volumes/PRO-G40/lego_data/analysis/vis_classifiers/v8_model.pt")

EDGE_DIM = 8
NODE_EMB_DIM = 32 + 4 + 4 + 8 + 6
HIDDEN = 64


class _MP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.msg = nn.Linear(dim + EDGE_DIM, dim)
        self.upd = nn.Linear(dim * 2, dim)

    def forward(self, x, ei, ea):
        src = ei[0]; dst = ei[1]
        m = torch.relu(self.msg(torch.cat([x[src], ea], dim=-1)))
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, m)
        deg = torch.zeros(x.size(0), device=x.device).scatter_add_(
            0, dst, torch.ones(ei.size(1), device=x.device, dtype=torch.float32))
        agg = agg / deg.clamp(min=1).unsqueeze(-1)
        return torch.relu(self.upd(torch.cat([x, agg], dim=-1)))


class V8GNN(nn.Module):
    def __init__(self, hidden=HIDDEN, n_layers=3):
        super().__init__()
        self.part_emb = nn.Embedding(PART_VOCAB_SIZE, 32)
        self.cat_emb = nn.Embedding(len(CATEGORIES), 4)
        self.size_emb = nn.Embedding(5, 4)
        self.col_emb = nn.Embedding(len(COLOUR_BINS) + 1, 8)
        self.embed = nn.Linear(NODE_EMB_DIM, hidden)
        self.layers = nn.ModuleList([_MP(hidden) for _ in range(n_layers)])
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(),
                                    nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 2))

    def forward(self, ei, ea, nc, ncont, graph_id, n_graphs):
        part_e = self.part_emb(nc[:, 0])
        cat_e = self.cat_emb(nc[:, 1])
        size_e = self.size_emb(nc[:, 2])
        col_e = self.col_emb(nc[:, 3])
        x = torch.cat([part_e, cat_e, size_e, col_e, ncont], dim=-1)
        x = torch.relu(self.embed(x))
        for layer in self.layers:
            x = layer(x, ei, ea)
        sums = torch.zeros(n_graphs, x.size(-1), device=x.device)
        sums.index_add_(0, graph_id, x)
        counts = torch.zeros(n_graphs, device=x.device).scatter_add_(
            0, graph_id, torch.ones(graph_id.size(0), device=x.device))
        pooled = sums / counts.clamp(min=1).unsqueeze(-1)
        return self.head(pooled)


def load_v8(path: Path | str = MODEL_PT):
    """Load a V8 / V9 / V11 subgraph GNN; auto-detects via the ckpt's
    `arch` or `colour_agnostic` flags."""
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    arch = ckpt.get("arch")
    if arch == "v14":                                                     # all-parts judge (names -> features)
        from v14_train import load_v14
        return load_v14(path)
    if arch == "v11":
        from v11_train import V11GNN
        m = V11GNN(hidden=ckpt["hidden"], n_layers=ckpt["n_layers"],
                   dropout=ckpt.get("dropout", 0.0))
    elif ckpt.get("colour_agnostic"):
        from v9_train import V9GNN
        m = V9GNN(hidden=ckpt["hidden"], n_layers=ckpt["n_layers"])
    else:
        m = V8GNN(hidden=ckpt["hidden"], n_layers=ckpt["n_layers"])
    m.load_state_dict(ckpt["state_dict"]); m.eval()
    return m


def score_items(model: V8GNN, items: list, bbox_cache: dict, part_map: dict,
                 k: int = 20, target_n: int = 50, seed: int | None = None) -> float:
    """Mean P(real) over K random 50-brick subgraphs of `items`, BATCHED
    into one GNN forward.  Returns 0.5 if the build is too small."""
    if len(items) < target_n + 2:
        return 0.5
    pairs = _neighbour_pairs(items)
    adj: dict = {}
    for a, b, _ in pairs:
        adj.setdefault(a, []).append(b); adj.setdefault(b, []).append(a)
    rng = random.Random(seed if seed is not None else random.randrange(1 << 30))

    # Collect K subgraphs, concat into one batched forward (K× speedup).
    all_cat, all_cont, all_ei, all_ea, all_gid = [], [], [], [], []
    n_off = 0; kept = 0
    for _ in range(k):
        sub = extract_subgraph(items, adj, target_n, rng)
        if sub is None: continue
        nodes, edges = sub
        if len(edges) == 0: continue
        tens = nodes_edges_to_tensors(items, nodes, edges, bbox_cache, part_map)
        n = tens["cat"].shape[0]
        if getattr(model, "table", None) is not None:                     # V14: part names -> ids + features
            c, f = model.encode([items[i][0] for i in nodes], tens["cont"]); all_cat.append(c); all_cont.append(f)
        else:
            all_cat.append(torch.tensor(tens["cat"], dtype=torch.long))
            all_cont.append(torch.tensor(tens["cont"]))
        all_ei.append(torch.tensor(tens["ei"], dtype=torch.long) + n_off)
        all_ea.append(torch.tensor(tens["ea"]))
        all_gid.append(torch.full((n,), kept, dtype=torch.long))
        n_off += n; kept += 1
    if kept == 0:
        return 0.5
    ei = torch.cat(all_ei, dim=1)
    ea = torch.cat(all_ea, dim=0)
    nc = torch.cat(all_cat, dim=0)
    ncont = torch.cat(all_cont, dim=0)
    gid = torch.cat(all_gid, dim=0)
    with torch.no_grad():
        logits = model(ei, ea, nc, ncont, gid, kept)
        probs = torch.softmax(logits, dim=-1)[:, 1]
    return float(probs.mean().item())
