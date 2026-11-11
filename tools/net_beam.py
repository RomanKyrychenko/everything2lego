"""Beam-search generation with the trained proposal network — the
option-2 approach the greedy net_build.py pointed to: instead of
picking the locally best option at each cell, maintain K candidate
partial sequences and at each step keep the top-K cumulative log-prob
extensions.

Score = sum of log P(option | context) across the sequence.  Beams
that run out of valid options at a cell are dropped.  Final build is
the surviving beam with the lowest total NLL (same seed-plus-order,
so deterministic).

    python tools/net_beam.py mesh.glb out/ --studs 32 --kind building
        [--beams 8]  [--per-cell 6]  [--seed 0]

Memory: K beams × owner/kindgrid int arrays — a few MB at 32 studs.
Compute: K × cells × top-M scoring; net forward is cheap enough that
a 1500-cell build runs in tens of seconds.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from sampler_demo import target_from_mesh  # noqa: E402
from everything2lego import proposals  # noqa: E402
from everything2lego.pack import Placement  # noqa: E402
from everything2lego.io_ldraw import write_ldr  # noqa: E402
from everything2lego.sampler import select_vocabulary  # noqa: E402
from everything2lego.style import kind_of  # noqa: E402
from everything2lego.colors import palette  # noqa: E402


@dataclass
class Beam:
    owner: np.ndarray      # (H, X, Z) int32, owner id per cell (-1 empty)
    kindgrid: np.ndarray   # (H, X, Z) int64, kind index per cell for the net context
    placements: list       # list[Placement]
    nll: float             # cumulative -log P


def _best_colour(cell_cols: np.ndarray, allowed_codes: set | None) -> int:
    from collections import Counter
    vals = cell_cols[cell_cols >= 0]
    pal = palette()
    c = int(Counter(vals.tolist()).most_common(1)[0][0]) if len(vals) else 0
    if allowed_codes is None or pal[c].code in allowed_codes:
        return c
    for i, p in enumerate(pal):
        if p.code in allowed_codes:
            return i
    return c


def _valid_options(opts, cy, cx, cz, owner, want, H, X, Z):
    """Yield k for each option that fits at (cy, cx, cz) in `owner`."""
    for k, opt in enumerate(opts):
        h, W, D, cells_arr = opt[0], opt[1], opt[2], opt[3]
        if cy + h > H or cx + W > X or cz + D > Z:
            continue
        abs_cells = cells_arr + np.array([cy, cx, cz])
        if (owner[abs_cells[:, 0], abs_cells[:, 1], abs_cells[:, 2]] >= 0).any():
            continue
        if not want[abs_cells[:, 0], abs_cells[:, 1], abs_cells[:, 2]].all():
            continue
        yield k, abs_cells


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh")
    ap.add_argument("out", type=Path)
    ap.add_argument("--studs", type=int, default=32)
    ap.add_argument("--shell", type=int, default=2)
    ap.add_argument("--kind", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--beams", type=int, default=8)
    ap.add_argument("--per-cell", type=int, default=6, help="top-M options per beam at each cell")
    ap.add_argument("--n-special", type=int, default=160)
    a = ap.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    target = target_from_mesh(a.mesh, a.studs, a.shell, a.kind)
    net = proposals.load()
    if net is None:
        raise SystemExit("no proposal_net.npz; train one first via tools/learn/proposal_net.py train")
    vocab = select_vocabulary(target.kind, a.n_special)
    opts = [(o.size[0], o.size[1], o.size[2], o.cells, o.bits, v.part, v.colours, o.rot, v.token)
            for v in vocab for o in v.orients]
    opt_token = np.array([net.token_index.get(opt[8], net.other()) for opt in opts])
    H, X, Z = target.want.shape
    static = proposals.static_context(target.fill)
    obj_idx = net.object_index.get(target.kind or "", -1)

    ys, xs, zs = np.nonzero(target.want)
    order = np.lexsort((zs, xs, ys))
    cells = list(zip(ys[order].tolist(), xs[order].tolist(), zs[order].tolist()))
    print(f"net.beam @ {a.studs} studs, {len(cells)} cells, {len(opts)} options, "
          f"beams={a.beams}, per-cell={a.per_cell}", flush=True)

    # start with a single empty beam
    beams = [Beam(np.full((H, X, Z), -1, np.int32),
                   np.full((H, X, Z), -1, np.int64), [], 0.0)]
    t0 = time.time()
    rnd = np.random.default_rng(a.seed)

    for ci, (cy, cx, cz) in enumerate(cells):
        if ci % 500 == 0:
            print(f"  cell {ci}/{len(cells)}  beams={len(beams)}  "
                  f"best_nll={min(b.nll for b in beams):.1f}", flush=True)
        expansions = []  # (nll_after, beam_idx, k, abs_cells)
        for bi, beam in enumerate(beams):
            if beam.owner[cy, cx, cz] >= 0:
                # cell already covered in this beam (from an earlier multi-cell part);
                # beam passes through unchanged at zero cost
                expansions.append((beam.nll, bi, None, None))
                continue
            ctx = proposals.features(target.fill, beam.kindgrid, static, cy, cx, cz, obj_idx,
                                      len(net.kinds), len(net.objects))
            lp = net.logprobs(ctx[None])[0]
            # score each valid option
            opt_scores = []
            for k, abs_cells in _valid_options(opts, cy, cx, cz, beam.owner, target.want, H, X, Z):
                opt_scores.append((float(-lp[opt_token[k]]), k, abs_cells))
            # top M by lowest -log P (best log P)
            opt_scores.sort(key=lambda t: t[0])
            for nll_add, k, abs_cells in opt_scores[:a.per_cell]:
                expansions.append((beam.nll + nll_add, bi, k, abs_cells))
            if not opt_scores:
                # beam has no valid option; drops out next step unless cell is skipped
                pass
        if not expansions:
            break
        # keep top K by nll
        expansions.sort(key=lambda t: t[0])
        new_beams = []
        used_prefixes = set()                                        # dedupe identical placements path
        for nll, bi, k, abs_cells in expansions:
            key = (bi, k, cy, cx, cz) if k is not None else (bi, None)
            if key in used_prefixes:
                continue
            used_prefixes.add(key)
            src = beams[bi]
            if k is None:
                new_beams.append(src)                                # pass-through, same owner
            else:
                owner = src.owner.copy()
                kindgrid = src.kindgrid.copy()
                owner[abs_cells[:, 0], abs_cells[:, 1], abs_cells[:, 2]] = len(src.placements)
                kind_i = net.kind_index.get(kind_of(opts[k][8]), len(net.kinds))
                kindgrid[abs_cells[:, 0], abs_cells[:, 1], abs_cells[:, 2]] = kind_i
                h, W, D, cells_arr, bits_arr, part, colours, rot, tok = opts[k]
                allowed = set(colours) if colours else None
                colour = _best_colour(target.colour[abs_cells[:, 0], abs_cells[:, 1], abs_cells[:, 2]], allowed)
                plc = Placement(part, colour, cx, cz, cy, W, D, rot)
                new_beams.append(Beam(owner, kindgrid, src.placements + [plc], nll))
            if len(new_beams) >= a.beams:
                break
        beams = new_beams

    secs = time.time() - t0
    winner = min(beams, key=lambda b: b.nll)
    covered = int((winner.owner >= 0).sum())
    want_total = int(target.want.sum())
    print(f"done: {len(winner.placements)} placements, {covered}/{want_total} cells covered, "
          f"nll {winner.nll:.1f} ({secs:.1f}s, {len(beams)} beams remaining)")

    write_ldr(winner.placements, a.out / "model.ldr")
    subprocess.run([sys.executable, str(ROOT / "tools/render_build.py"), str(a.out / "model.ldr"),
                    str(a.out / "render.png"), "--views", "30,120,210", "--size", "500"], check=False)


if __name__ == "__main__":
    main()
