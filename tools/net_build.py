"""Greedy builder that uses the proposal network *alone* — no
Metropolis walk, no style/support/outline scoring on top.  Produces a
build by iterating cells in fill order and placing the top-scored
option (restricted to options that fit the cell and don't collide).

The point: test whether the learned policy — the option 2 line of
"learned model produces designs by construction" — can produce a
build by itself.  If it can, the Metropolis walk is only polishing; if
it can't, the walk is carrying real design decisions and the net alone
isn't a sufficient policy.

    python tools/net_build.py mesh.glb out/ --studs 32 --kind building

Output: out/model.ldr + out/render.png.  No info.json (not a sampler
result).  Deterministic given --seed (random tie-breaking only).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from sampler_demo import target_from_mesh  # noqa: E402
from everything2lego import proposals  # noqa: E402
from everything2lego.pack import Placement  # noqa: E402
from everything2lego.io_ldraw import write_ldr  # noqa: E402
from everything2lego.sampler import select_vocabulary, SUB, popcount, _rotate  # noqa: E402
from everything2lego.style import kind_of  # noqa: E402


def _options(vocab):
    """Flat list of (vp, orient) choices and per-option (token, box_kind, shape)."""
    opts = []
    for v in vocab:
        for o in v.orients:
            opts.append((v, o))
    return opts


def _best_colour(cell_cols: np.ndarray, allowed: set | None) -> int:
    """Dominant colour among (y, x, z) cells; filtered to allowed codes."""
    from collections import Counter
    from everything2lego.colors import palette
    pal = palette()
    vals = cell_cols[cell_cols >= 0]
    if not len(vals):
        c = 0
    else:
        c = int(Counter(vals.tolist()).most_common(1)[0][0])
    if allowed is None or pal[c].code in allowed:
        return c
    # pick first allowed palette index
    for i, p in enumerate(pal):
        if p.code in allowed:
            return i
    return c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh")
    ap.add_argument("out", type=Path)
    ap.add_argument("--studs", type=int, default=32)
    ap.add_argument("--shell", type=int, default=2)
    ap.add_argument("--kind", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-special", type=int, default=160, help="vocabulary width (same as sampler_demo)")
    a = ap.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    target = target_from_mesh(a.mesh, a.studs, a.shell, a.kind)
    net = proposals.load()
    if net is None:
        raise SystemExit("no proposal_net.npz; train one first via tools/learn/proposal_net.py train")
    vocab = select_vocabulary(target.kind, a.n_special)
    opts = _options(vocab)
    # per-option token index (for net output lookup) and (opt_h, opt_W, opt_D, opt_cells, opt_bits)
    opt_token = np.array([net.token_index.get(v.token, net.other()) for v, _ in opts])
    opt_sizes = [(o.size[0], o.size[1], o.size[2], o.cells, o.bits, v.part, v.colours, o.rot)
                 for v, o in opts]

    H, X, Z = target.want.shape
    owner = np.full((H, X, Z), -1, dtype=np.int32)
    kindgrid = np.full((H, X, Z), -1, dtype=np.int64)
    static = proposals.static_context(target.fill)
    obj_idx = net.object_index.get(target.kind or "", -1)

    print(f"net.generate @ {a.studs} studs, {int(target.want.sum())} cells, {len(opts)} options", flush=True)
    t0 = time.time()
    rnd = np.random.default_rng(a.seed)
    placements: list[Placement] = []
    # fill cells bottom-up, left-to-right (same order as filler.fill)
    ys, xs, zs = np.nonzero(target.want)
    order = np.lexsort((zs, xs, ys))
    cells = list(zip(ys[order].tolist(), xs[order].tolist(), zs[order].tolist()))
    placed = 0
    for cy, cx, cz in cells:
        if owner[cy, cx, cz] >= 0:
            continue
        ctx = proposals.features(target.fill, kindgrid, static, cy, cx, cz, obj_idx,
                                  len(net.kinds), len(net.objects))
        lp = net.logprobs(ctx[None])[0]                            # (n_tokens+1,)
        # filter options whose placement fits
        best_k, best_score = -1, -np.inf
        for k, (h, W, D, cells_arr, bits_arr, part, colours, rot) in enumerate(opt_sizes):
            if cy + h > H or cx + W > X or cz + D > Z:
                continue
            abs_cells = cells_arr + np.array([cy, cx, cz])
            if (owner[abs_cells[:, 0], abs_cells[:, 1], abs_cells[:, 2]] >= 0).any():
                continue
            if not target.want[abs_cells[:, 0], abs_cells[:, 1], abs_cells[:, 2]].all():
                continue
            score = float(lp[opt_token[k]])
            if score > best_score:
                best_score, best_k = score, k
        if best_k < 0:
            continue
        h, W, D, cells_arr, bits_arr, part, colours, rot = opt_sizes[best_k]
        abs_cells = cells_arr + np.array([cy, cx, cz])
        allowed = colours
        from everything2lego.colors import palette
        pal = palette()
        allowed_codes = set(colours) if colours else None
        colour = _best_colour(target.colour[abs_cells[:, 0], abs_cells[:, 1], abs_cells[:, 2]], allowed_codes)
        # place
        owner[abs_cells[:, 0], abs_cells[:, 1], abs_cells[:, 2]] = len(placements)
        kind_i = net.kind_index.get(kind_of(opts[best_k][0].token), len(net.kinds))
        kindgrid[abs_cells[:, 0], abs_cells[:, 1], abs_cells[:, 2]] = kind_i
        placements.append(Placement(part, colour, cx, cz, cy, W, D, rot))
        placed += 1

    secs = time.time() - t0
    print(f"generated {len(placements)} placements in {secs:.1f}s "
          f"({owner.sum() >= 0 and int((owner >= 0).sum())} cells covered of {int(target.want.sum())})")

    write_ldr(placements, a.out / "model.ldr")
    subprocess.run([sys.executable, str(ROOT / "tools/render_build.py"), str(a.out / "model.ldr"),
                    str(a.out / "render.png"), "--views", "30,120,210", "--size", "500"], check=False)


if __name__ == "__main__":
    main()
