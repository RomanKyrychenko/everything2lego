"""Colour smoothing of a sampler target as a Markov random field (Potts model with learned pair costs).

A mesh's texture gives neighbouring cells slightly different palette colours; the sampler then keeps many small
parts to copy each patch, and `merge` (same colour only) can't join them.  Here every target cell picks a colour c
minimising

    E = sum_cells |Lab(mesh colour) - Lab(c)|  +  beta * sum_touching pairs  cost(c_a, c_b)

cost(a, a) = 0; cost(a, b) = -log of how much more often real sets put a and b side by side than chance (PMI),
floored at `min_cost`: colour pairs real sets combine are cheap, the rest dear.  Solved by iterated conditional
modes on a checkerboard (each half of the grid updates at once), a few sweeps.

    from everything2lego.colour_mrf import smooth
    target.colour = smooth(target.colour, target.want, beta=6.0)
"""
from __future__ import annotations

import json
from functools import lru_cache
from importlib.resources import files
from pathlib import Path

import numpy as np

from .colors import _palette_lab, palette

ADJ = "colour_adjacency.json"


@lru_cache(maxsize=1)
def adjacency() -> np.ndarray:
    """(P, P) touching-pair counts between palette colours in real sets (data/colour_adjacency.json)."""
    d = json.loads(files("everything2lego.data").joinpath(ADJ).read_text())
    codes = [c.code for c in palette()]; ix = {c: i for i, c in enumerate(codes)}
    A = np.zeros((len(codes), len(codes)))
    for (a, b), n in zip(d["pairs"], d["counts"]):
        if a in ix and b in ix: A[ix[a], ix[b]] += n; A[ix[b], ix[a]] += n
    return A


def pair_cost(min_cost: float = 0.2, max_cost: float = 3.0) -> np.ndarray:
    """(P, P): 0 on the diagonal, else clip(-PMI) from real-set adjacency (unseen pairs -> max_cost)."""
    A = adjacency() + 1e-9; P = A / A.sum(); m = P.sum(1)
    pmi = np.log(P / np.outer(m, m))
    C = np.clip(-pmi, min_cost, max_cost); C[adjacency() == 0] = max_cost; np.fill_diagonal(C, 0.0)
    return C


def smooth(colour: np.ndarray, want: np.ndarray, beta: float = 6.0, sweeps: int = 8, labels: np.ndarray | None = None) -> np.ndarray:
    """colour (H, X, Z) palette indices (-1 unknown) -> smoothed copy; only `want` cells change.
    labels: palette indices allowed (default: those the target already uses)."""
    lab = _palette_lab(); C = pair_cost()
    L = np.unique(colour[want & (colour >= 0)]) if labels is None else np.asarray(labels)
    if len(L) < 2 or beta <= 0: return colour.copy()
    D = np.full(colour.shape + (len(L),), 0.0)                                   # data term per cell and label
    known = want & (colour >= 0)
    D[known] = np.linalg.norm(lab[colour[known]][:, None, :] - lab[L][None], axis=-1)
    CL = C[np.ix_(L, L)]
    cur = np.where(known, np.searchsorted(L, np.where(known, colour, L[0])), 0)  # label index per cell
    cur[~np.isin(colour, L) & known] = D[~np.isin(colour, L) & known].argmin(-1)
    par = np.indices(colour.shape).sum(0) % 2
    for _ in range(sweeps):
        changed = 0
        for p in (0, 1):
            E = D.copy()
            for ax in range(3):
                for sh in (1, -1):
                    nb = np.roll(cur, sh, axis=ax); ok = np.roll(want, sh, axis=ax)
                    edge = [slice(None)] * 3; edge[ax] = 0 if sh == 1 else -1; ok[tuple(edge)] = False   # no wrap-around
                    E += beta * np.where(ok[..., None], CL[nb], 0.0)
            new = E.argmin(-1); upd = want & (par == p)
            changed += int((new[upd] != cur[upd]).sum()); cur[upd] = new[upd]
        if changed == 0: break
    out = colour.copy(); out[want] = L[cur[want]]
    return out


def build_adjacency(out: Path, limit: int = 0) -> None:
    """Count touching-pair colours over the real OMR sets (flatten -> grid boxes -> neighbour pairs)."""
    import sys
    from collections import Counter
    root = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(root / "tools" / "learn"), str(root / "tools" / "set_study")]
    from ldraw_read import flatten, part_bbox
    from realism import _neighbour_pairs, _parts_with_grid
    files_ = sorted(Path("/Volumes/PRO-G40/lego_data/omr/files").glob("*.mpd"))[: limit or None]
    cnt: Counter = Counter(); bb: dict = {}
    for k, f in enumerate(files_):
        try: ps = flatten(f)
        except Exception: continue                                               # noqa: BLE001
        for q in ps:
            if q.name not in bb:
                try: bb[q.name] = part_bbox(q.name)
                except Exception: bb[q.name] = None                              # noqa: BLE001
        it = _parts_with_grid(ps, bb)
        for a, b, _ in _neighbour_pairs(it):
            if a < b: cnt[tuple(sorted((int(it[a][1]), int(it[b][1]))))] += 1
        if k % 200 == 0: print(f"  {k}/{len(files_)} sets, {sum(cnt.values())} pairs", flush=True)
    keys = sorted(cnt)
    out.write_text(json.dumps({"_source": "colour_mrf.build_adjacency over OMR sets (touching grid boxes)",
                               "pairs": [list(k) for k in keys], "counts": [cnt[k] for k in keys]}))
    print(f"wrote {out}: {len(keys)} colour pairs, {sum(cnt.values())} touching pairs", flush=True)


if __name__ == "__main__":
    build_adjacency(Path(__file__).resolve().parent / "data" / ADJ)
