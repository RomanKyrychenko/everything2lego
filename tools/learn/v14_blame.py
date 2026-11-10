"""Where V14 thinks a build looks fake: per-part blame = mean P(fake) over the 50-part pieces that contain the part.

Pieces are grown from seed parts until every part is in at least `cover` pieces, then scored in one batched forward.
  python tools/learn/v14_blame.py model.ldr [--out blamed.ldr]   (out: parts recoloured, red = most blamed)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "src"), str(HERE.parents[1] / "tools" / "set_study")]
from subgraph_extract import extract_subgraph_from, nodes_edges_to_tensors   # noqa: E402

TARGET_N = 50


def blame(model, items: list, adj: dict, bb: dict, cover: int = 2, seed: int = 0) -> np.ndarray:
    """Per item: mean P(fake) of the pieces containing it (NaN where no piece reached it)."""
    rng = np.random.default_rng(seed)
    hits = np.zeros(len(items), np.int32)
    pieces = []
    for s in rng.permutation(len(items)):
        if hits[s] >= cover or s not in adj:
            continue
        sub = extract_subgraph_from(items, adj, TARGET_N, int(s))
        if sub is None or not sub[1]:
            hits[s] = cover
            continue
        pieces.append(sub)
        hits[sub[0]] += 1
    cat, cont, ei, ea, gid, off = [], [], [], [], [], 0
    for g, (nodes, edges) in enumerate(pieces):
        t = nodes_edges_to_tensors(items, nodes, edges, bb, {})
        c, f = model.encode([items[i][0] for i in nodes], t["cont"])
        cat.append(c); cont.append(f); ea.append(torch.tensor(t["ea"]))
        ei.append(torch.tensor(t["ei"], dtype=torch.long) + off); gid.append(torch.full((len(nodes),), g)); off += len(nodes)
    out, n = np.zeros(len(items)), np.zeros(len(items))
    if not pieces:
        return np.full(len(items), np.nan)
    with torch.no_grad():
        p_fake = torch.softmax(model(torch.cat(ei, 1), torch.cat(ea), torch.cat(cat), torch.cat(cont), torch.cat(gid), len(pieces)), -1)[:, 0].numpy()
    for (nodes, _), p in zip(pieces, p_fake):
        out[nodes] += p; n[nodes] += 1
    return np.where(n > 0, out / np.maximum(n, 1), np.nan)


_MODEL, _BB = None, {}


def blame_lines(lines: list) -> np.ndarray:
    """Blame per LDraw type-1 line (NaN: a part V14 can't read, or the build is too small to judge)."""
    global _MODEL
    from ldraw_read import Part, _matrix, part_bbox
    from realism import _neighbour_pairs, _parts_with_grid
    if _MODEL is None:
        from v8_score import load_v8
        _MODEL = load_v8("/Volumes/PRO-G40/lego_data/analysis/vis_classifiers/v14_model.pt")
    parts, kept = [], []
    for k, l in enumerate(lines):
        t = l.split()
        name = t[14].lower()
        if name not in _BB:
            try: _BB[name] = part_bbox(name)
            except Exception: _BB[name] = None                                  # noqa: BLE001
        if _BB[name] is not None:
            parts.append(Part(name, int(t[1]), _matrix(t[2:14]), 0, ())); kept.append(k)
    items = _parts_with_grid(parts, _BB)                                        # keeps all: every part has a bbox here
    out = np.full(len(lines), np.nan)
    if len(items) < TARGET_N + 5:
        return out
    adj: dict = {}
    for a, b, _ in _neighbour_pairs(items):
        adj.setdefault(a, []).append(b); adj.setdefault(b, []).append(a)
    out[kept] = blame(_MODEL, items, adj, _BB)
    return out


def main() -> None:
    from v8_score import load_v8
    from v14_build import _items
    ap = argparse.ArgumentParser()
    ap.add_argument("ldr", type=Path); ap.add_argument("--out", type=Path); ap.add_argument("--cover", type=int, default=2)
    a = ap.parse_args()
    model = load_v8("/Volumes/PRO-G40/lego_data/analysis/vis_classifiers/v14_model.pt")
    t0 = time.time(); items, adj, bb = _items(a.ldr); t1 = time.time()
    b = blame(model, items, adj, bb, a.cover); t2 = time.time()
    ok = b[~np.isnan(b)]
    print(f"{len(items)} parts: load+graph {t1 - t0:.1f}s, blame {t2 - t1:.1f}s; P(fake) mean {ok.mean():.3f}, "
          f"p10 {np.percentile(ok, 10):.3f}, p90 {np.percentile(ok, 90):.3f}, unreached {np.isnan(b).sum()}")
    if a.out:                                                                  # red: top 10 % blamed, light grey: rest
        cut = np.nanpercentile(b, 90)
        def line(it, bi):
            m = it[2]
            return " ".join(["1", "4" if bi >= cut else "71", *(f"{v:g}" for v in m[:3, 3]), *(f"{v:g}" for v in m[:3, :3].ravel()), it[0]])
        a.out.write_text("\n".join(line(it, bi) for it, bi in zip(items, np.nan_to_num(b))) + "\n")


if __name__ == "__main__":
    main()
