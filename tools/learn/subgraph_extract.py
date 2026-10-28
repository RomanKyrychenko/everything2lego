"""Subgraph extraction + spoiling + feature construction for V7.

Operates on items from _parts_with_grid, which are tuples
    (ldraw, colour, M_4x4, glo_xyz, ghi_xyz)
where glo/ghi are in stud / plate units.

Fast path: all per-item lookups vectorise against the item index; no
Python-level recomputation inside the inner loop."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np

# Feature constants matching the existing GNN vocab for continuity.
from realism import CATEGORIES, COLOUR_BINS, _category, _colour_bin, _size_bin, _neighbour_pairs
from realism_gnn import PART_VOCAB_SIZE, _part_id_map

NODE_FEAT_DIM = (
    1                                                                     # part_id (int, embedded)
    + 1                                                                   # category (int, embedded)
    + 1                                                                   # size bin (int, embedded)
    + 1                                                                   # colour bin (int, embedded)
    + 6                                                                   # continuous: rel_x, rel_y, rel_z, h_plates, w_studs, d_studs
)
EDGE_FEAT_DIM = 5 + 3                                                     # 5-dim geom-relation one-hot + 3-dim offset
GEOM_RELS = ("above", "below", "side_x", "side_z", "other")
_GEOM_IX = {g: i for i, g in enumerate(GEOM_RELS)}


def _item_h_w_d(item) -> tuple[float, float, float]:
    _, _, _, glo, ghi = item
    return float(ghi[1] - glo[1]), float(ghi[0] - glo[0]), float(ghi[2] - glo[2])


def _rel_center(items: list, nodes: list[int]) -> tuple[float, float, float]:
    xs = [(items[i][3][0] + items[i][4][0]) / 2 for i in nodes]
    ys = [(items[i][3][1] + items[i][4][1]) / 2 for i in nodes]
    zs = [(items[i][3][2] + items[i][4][2]) / 2 for i in nodes]
    return sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs)


def _geom_relation(items: list, a: int, b: int) -> tuple[str, tuple[float, float, float]]:
    """Classify how item b sits relative to item a."""
    ca = ((items[a][3][0] + items[a][4][0]) / 2, (items[a][3][1] + items[a][4][1]) / 2,
          (items[a][3][2] + items[a][4][2]) / 2)
    cb = ((items[b][3][0] + items[b][4][0]) / 2, (items[b][3][1] + items[b][4][1]) / 2,
          (items[b][3][2] + items[b][4][2]) / 2)
    dx, dy, dz = cb[0] - ca[0], cb[1] - ca[1], cb[2] - ca[2]
    if abs(dy) > max(abs(dx), abs(dz)) + 0.5:
        rel = "above" if dy > 0 else "below"
    elif abs(dx) > abs(dz):
        rel = "side_x"
    elif abs(dz) > abs(dx):
        rel = "side_z"
    else:
        rel = "other"
    return rel, (dx, dy, dz)


def extract_subgraph(items: list, adj: dict, target_n: int, rng: random.Random):
    """BFS-sample a connected subgraph of `target_n` nodes.
    Returns (node_indices, [(src, dst) edges]) or None if too small."""
    n = len(items)
    if n < target_n:
        return None
    seed = rng.randrange(n)
    kept = [seed]
    kept_set = {seed}
    frontier = [seed]
    while len(kept) < target_n and frontier:
        nxt = []
        for u in frontier:
            for v in adj.get(u, ()):
                if v not in kept_set:
                    kept.append(v); kept_set.add(v); nxt.append(v)
                    if len(kept) >= target_n:
                        break
            if len(kept) >= target_n:
                break
        frontier = nxt
    if len(kept) < target_n - 2:                                          # too disconnected
        return None
    edges = []
    kept_set = set(kept)
    for u in kept:
        for v in adj.get(u, ()):
            if v in kept_set and u < v:
                edges.append((u, v))
    return kept, edges


def extract_subgraph_from(items: list, adj: dict, target_n: int, seed_node: int):
    """BFS a connected subgraph of up to `target_n` nodes STARTING FROM
    the given node.  Deterministic — same input same output.  Returns
    (nodes, edges) or None if the connected component is too small.
    Used by local-change scoring (structural refiner moves)."""
    n = len(items)
    if seed_node < 0 or seed_node >= n:
        return None
    kept = [seed_node]
    kept_set = {seed_node}
    frontier = [seed_node]
    while len(kept) < target_n and frontier:
        nxt = []
        for u in frontier:
            for v in adj.get(u, ()):
                if v not in kept_set:
                    kept.append(v); kept_set.add(v); nxt.append(v)
                    if len(kept) >= target_n:
                        break
            if len(kept) >= target_n:
                break
        frontier = nxt
    if len(kept) < min(10, target_n):                                     # too small to score honestly
        return None
    edges = []
    for u in kept:
        for v in adj.get(u, ()):
            if v in kept_set and u < v:
                edges.append((u, v))
    return kept, edges


def nodes_edges_to_tensors(items: list, nodes: list[int], edges: list, bbox_cache: dict,
                            part_map: dict) -> dict:
    """Convert a subgraph specification into fixed-shape numpy arrays."""
    n = len(nodes)
    cx, cy, cz = _rel_center(items, nodes)
    cat_ints = np.zeros((n, 4), dtype=np.int64)
    cont = np.zeros((n, 6), dtype=np.float32)
    for i, idx in enumerate(nodes):
        ld, col, M, glo, ghi = items[idx]
        cat_ints[i, 0] = part_map.get(ld, 0)
        cat_ints[i, 1] = _category(ld)
        cat_ints[i, 2] = _size_bin(ld, bbox_cache)
        cat_ints[i, 3] = _colour_bin(int(col))
        mx, my, mz = (float(glo[0]) + float(ghi[0])) / 2, (float(glo[1]) + float(ghi[1])) / 2, (float(glo[2]) + float(ghi[2])) / 2
        cont[i, 0] = (mx - cx) / 10.0                                     # normalised to ~[-1, 1] for typical subgraphs
        cont[i, 1] = (my - cy) / 10.0
        cont[i, 2] = (mz - cz) / 10.0
        h, w, d = _item_h_w_d(items[idx])
        cont[i, 3] = h / 5.0
        cont[i, 4] = w / 5.0
        cont[i, 5] = d / 5.0
    old_to_new = {old: new for new, old in enumerate(nodes)}
    E = len(edges) * 2                                                    # undirected -> both directions
    ei = np.zeros((2, E), dtype=np.int64)
    ea = np.zeros((E, EDGE_FEAT_DIM), dtype=np.float32)
    for k, (u, v) in enumerate(edges):
        rel_uv, d_uv = _geom_relation(items, u, v)
        rel_vu, d_vu = _geom_relation(items, v, u)
        for s, (a, b, rel, d) in enumerate(((u, v, rel_uv, d_uv), (v, u, rel_vu, d_vu))):
            kk = 2 * k + s
            ei[0, kk] = old_to_new[a]
            ei[1, kk] = old_to_new[b]
            ea[kk, _GEOM_IX[rel]] = 1.0
            ea[kk, 5] = max(-5, min(5, d[0])) / 5.0
            ea[kk, 6] = max(-5, min(5, d[1])) / 5.0
            ea[kk, 7] = max(-5, min(5, d[2])) / 5.0
    return {"cat": cat_ints, "cont": cont, "ei": ei, "ea": ea}


# ---------- spoilers ----------

def spoil_random_types(subg: dict, catalog_ldraws: list[str], part_map: dict, bbox_cache: dict,
                        rng: random.Random) -> dict:
    """Replace each node's ldraw with a random choice from the catalog.
    Updates part_id, category, size.  Position/colour kept."""
    out = {k: v.copy() for k, v in subg.items()}
    n = out["cat"].shape[0]
    for i in range(n):
        new_ld = rng.choice(catalog_ldraws)
        out["cat"][i, 0] = part_map.get(new_ld, 0)
        out["cat"][i, 1] = _category(new_ld)
        out["cat"][i, 2] = _size_bin(new_ld, bbox_cache)
    return out


def spoil_random_colours(subg: dict, rng: random.Random) -> dict:
    out = {k: v.copy() for k, v in subg.items()}
    n = out["cat"].shape[0]
    for i in range(n):
        out["cat"][i, 3] = rng.randrange(len(COLOUR_BINS) + 1)
    return out


def spoil_random_positions(subg: dict, rng: random.Random) -> dict:
    """Shuffle positions (preserve the set of positions, but assign them to
    random nodes).  Preserves structural scale but breaks the connectivity-
    position correspondence."""
    out = {k: v.copy() for k, v in subg.items()}
    n = out["cont"].shape[0]
    perm = list(range(n))
    rng.shuffle(perm)
    out["cont"][:, :3] = out["cont"][perm, :3]
    return out


def spoil_edges(subg: dict, rng: random.Random, drop_frac: float = 0.3,
                 add_count: int = 10) -> dict:
    """Randomly drop drop_frac of edges and add `add_count` bogus ones."""
    out = {"cat": subg["cat"].copy(), "cont": subg["cont"].copy()}
    E = subg["ei"].shape[1]
    n = out["cat"].shape[0]
    keep_mask = np.array([rng.random() > drop_frac for _ in range(E)], dtype=bool)
    ei_k = subg["ei"][:, keep_mask]
    ea_k = subg["ea"][keep_mask]
    # Add random edges with "other" relation + random offset
    extra_ei = np.zeros((2, add_count * 2), dtype=np.int64)
    extra_ea = np.zeros((add_count * 2, EDGE_FEAT_DIM), dtype=np.float32)
    for k in range(add_count):
        a, b = rng.sample(range(n), 2)
        for s, (u, v) in enumerate(((a, b), (b, a))):
            kk = 2 * k + s
            extra_ei[0, kk] = u
            extra_ei[1, kk] = v
            extra_ea[kk, _GEOM_IX["other"]] = 1.0
    out["ei"] = np.concatenate([ei_k, extra_ei], axis=1)
    out["ea"] = np.concatenate([ea_k, extra_ea], axis=0)
    return out


def spoil_full(subg: dict, catalog_ldraws: list[str], part_map: dict, bbox_cache: dict,
                rng: random.Random) -> dict:
    """All spoilers applied in sequence."""
    out = spoil_random_types(subg, catalog_ldraws, part_map, bbox_cache, rng)
    out = spoil_random_colours(out, rng)
    out = spoil_random_positions(out, rng)
    out = spoil_edges(out, rng)
    return out


SPOILERS = {
    "types": lambda s, cat, pm, bc, rng: spoil_random_types(s, cat, pm, bc, rng),
    "colours": lambda s, cat, pm, bc, rng: spoil_random_colours(s, rng),
    "positions": lambda s, cat, pm, bc, rng: spoil_random_positions(s, rng),
    "edges": lambda s, cat, pm, bc, rng: spoil_edges(s, rng),
    "full": spoil_full,
}
