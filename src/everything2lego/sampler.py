"""A universal LEGO generator: sample a build of a target shape whose local
construction follows the statistics of real sets.

The target is a grid (y plates up, x and z studs) of the cells to fill,
their colours, and which of them show.  A state is a set of non-overlapping
parts.  Its energy is

    E = w_cov  * cells wrongly filled or left empty
      + w_col  * colour error of the cells that show (Lab, per cell)
      + w_loc  * sum over touching pairs of -log P(B, where | A)
      + w_glob * pairs * KL(pattern mix || official models of similar shape)
      + w_sup  * parts held by no stud above or below

— fidelity, local plausibility (each part's neighbours are what real sets
put there) and global typicality (the mix of patterns is real sets', not
the same few repeated: style.RunningKL).  Both statistics come from the
official models nearest the target in shape (tools/partsdb/features.py),
smoothed from exact tokens (kind + size) to kinds to uniform.

A move re-tiles a small window: the parts meeting it are removed and its
cells re-filled part by part, each drawn from the learned P(part | the
parts already around it).  Metropolis acceptance at a temperature that
anneals to `t_final` > 0 — a sample, not the single likeliest build, which
would be monotonous.

    from everything2lego.sampler import Target, sample
    placements = sample(Target(want, colours, visible), refs, iters=20000)
"""
from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from importlib.resources import files
from typing import Optional

import numpy as np

from .bricks import Part
from .colors import _palette_lab, nearest_where, palette
from .pack import Placement
from .style import Reference, RunningKL, kind_of


# ---------------------------------------------------------------------------
# vocabulary: box parts from the parts database, as tokens the statistics know
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _vocab() -> tuple[tuple[Part, ...], dict[str, str], dict[str, frozenset[int]]]:
    """Plain bricks, plates and tiles LEGO makes now, with their real
    geometry, statistics token and colours (data/vocabulary.json, from the
    parts database: tools/partsdb/vocabulary.py)."""
    raw = json.loads(files("everything2lego.data").joinpath("vocabulary.json").read_text())
    parts, tok, cols = [], {}, {}
    for r in raw["parts"]:
        p = Part(ldraw=r["ldraw"], bl=r["part_num"], shape=r["shape"],
                 footprint=tuple(r["footprint"]), height_units=r["height_units"], name=r["name"])
        parts.append(p)
        tok[p.ldraw] = r["token"]
        cols[p.ldraw] = frozenset(int(c) for c in r["colours"])
    return tuple(parts), tok, cols


def vocabulary() -> tuple[Part, ...]:
    return _vocab()[0]


def token_of(part: Part) -> str:
    """The features.py token of a vocabulary part."""
    return _vocab()[1][part.ldraw]


def available(part: Part, code: int) -> bool:
    """Whether LEGO currently makes `part` in LDraw colour `code`."""
    return code in _vocab()[2][part.ldraw]


# ---------------------------------------------------------------------------
# target and statistics
# ---------------------------------------------------------------------------

@dataclass
class Target:
    want: np.ndarray                  # (H, X, Z) bool: cells to fill
    colour: np.ndarray                # (H, X, Z) palette index (-1 unknown)
    visible: np.ndarray               # (H, X, Z) bool: cells that show

    def shape_descriptor(self) -> np.ndarray:
        ys, xs, zs = np.nonzero(self.want)
        w, d = sorted((float(np.ptp(xs) + 1), float(np.ptp(zs) + 1)), reverse=True)
        h = float(np.ptp(ys) + 1) / 2.5
        fill = self.want.sum() / max(1.0, w * d * h * 2.5)
        return np.array([np.log(w), np.log(d), np.log(max(h, 0.4)), min(1.0, fill)])


class Stats:
    """Local conditional P(B, where | A) and the global reference, from the
    official models' pattern counts (a list of features rows)."""

    def __init__(self, models: list[dict], lam=(0.8, 0.17, 0.03)):
        self.ref = Reference.from_models(models)
        self.lam = lam
        self.pair = self.ref.counts                       # (a, b, d) -> weight
        self.a_tot: Counter = Counter()
        self.ka_tot: Counter = Counter()
        self.kpair: Counter = Counter()
        self.spread: Counter = Counter()
        for (a, b, d), n in self.pair.items():
            self.a_tot[a] += n
            self.ka_tot[kind_of(a)] += n
            self.kpair[(kind_of(a), kind_of(b), d)] += n
        for (a, b, d) in self.pair:
            self.spread[(kind_of(a), kind_of(b), d)] += 1
        self.vocab = max(1, len(self.pair))
        self._cache: dict = {}

    def nll(self, a: str, b: str, d: str) -> float:
        """-log P(B=b, where=d | A=a), smoothed exact -> kind -> uniform."""
        key = (a, b, d)
        v = self._cache.get(key)
        if v is None:
            l1, l2, l3 = self.lam
            ex = self.pair.get(key, 0) / self.a_tot[a] if self.a_tot.get(a) else 0.0
            kk = (kind_of(a), kind_of(b), d)
            ka = self.ka_tot.get(kind_of(a), 0)
            kd = self.kpair.get(kk, 0) / ka / (self.spread.get(kk, 0) + 1) if ka else 0.0
            v = self._cache[key] = -math.log(l1 * ex + l2 * kd + l3 / self.vocab)
        return v


def nearest_models(rows: list[dict], shape, k: int = 50) -> list[dict]:
    X = np.array([r["shape"] for r in rows])
    mu, sd = X.mean(0), X.std(0) + 1e-9
    d = (((X - mu) / sd - (np.asarray(shape) - mu) / sd) ** 2).sum(1)
    return [rows[i] for i in np.argsort(d)[:k]]


# ---------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------

@dataclass
class Item:
    part: Part
    colour: int
    y: int
    x: int
    z: int
    w: int                       # along x
    d: int                       # along z

    @property
    def h(self) -> int:
        return self.part.height_units

    def cells(self):
        return (slice(self.y, self.y + self.h), slice(self.x, self.x + self.w), slice(self.z, self.z + self.d))


@dataclass
class Weights:
    cov: float = 4.0
    col: float = 0.04
    loc: float = 1.0
    glob: float = 1.0
    sup: float = 3.0


class State:
    def __init__(self, target: Target, stats: Stats, weights: Weights):
        self.t, self.s, self.w = target, stats, weights
        self.owner = np.full(target.want.shape, -1, dtype=np.int32)
        self.items: dict[int, Item] = {}
        self.next_id = 0
        self.kl = RunningKL(stats.ref)
        self.loc = 0.0
        self.col = 0.0
        self.unsup: set[int] = set()
        self.bad_cells = int(target.want.sum())            # wanted and empty, or filled and unwanted
        self.lab = _palette_lab()
        self.tlab = np.where(target.colour[..., None] >= 0, self.lab[np.maximum(target.colour, 0)], np.nan)

    # -- geometry -----------------------------------------------------------
    def neighbours(self, i: int) -> dict[int, str]:
        """{j: where j is relative to i} for parts touching i (above, below, beside)."""
        it = self.items[i]
        H, X, Z = self.owner.shape
        out: dict[int, str] = {}
        x0, x1, z0, z1 = max(0, it.x - 1), min(X, it.x + it.w + 1), max(0, it.z - 1), min(Z, it.z + it.d + 1)
        # above and below over the footprint and its ring (diagonals count,
        # as the official models' patterns were counted)
        found = set(self.owner[max(0, it.y - 1):min(H, it.y + it.h + 1), x0:x1, z0:z1].ravel().tolist())
        found.discard(-1)
        found.discard(i)
        for j in found:
            # where j is, from the two boxes — as the official models'
            # patterns were classified (features.vertical), so the label
            # is the same seen from either part
            b = self.items[j]
            out[j] = "up" if b.y >= it.y + it.h else "down" if b.y + b.h <= it.y else "level"
        return out

    def _supported(self, i: int, nb: dict[int, str]) -> bool:
        """Held by studs: a part directly over or under its footprint (or the ground)."""
        it = self.items[i]
        if it.y == 0:
            return True
        H = self.owner.shape[0]
        below = self.owner[it.y - 1, it.x:it.x + it.w, it.z:it.z + it.d]
        if (below >= 0).any():
            return True
        if it.y + it.h < H:
            return bool((self.owner[it.y + it.h, it.x:it.x + it.w, it.z:it.z + it.d] >= 0).any())
        return False

    # -- energy bookkeeping ---------------------------------------------------
    def _pairs(self, i: int, nb: dict[int, str], sign: int, done: set) -> None:
        """Add (sign=+1) or remove (-1) i's pairs with its neighbours, both directions."""
        ti = token_of(self.items[i].part)
        flip = {"up": "down", "down": "up", "level": "level"}
        for j, d in nb.items():
            if (min(i, j), max(i, j)) in done:
                continue
            done.add((min(i, j), max(i, j)))
            tj = token_of(self.items[j].part)
            for a, b, dd in ((ti, tj, d), (tj, ti, flip[d])):
                self.kl.add((a, b, dd), sign)
                self.loc += sign * self.s.nll(a, b, dd)

    def _colour_cost(self, it: Item) -> float:
        sl = it.cells()
        vis = self.t.visible[sl]
        if not vis.any():
            return 0.0
        tl = self.tlab[sl][vis]
        tl = tl[~np.isnan(tl).any(axis=1)]
        if not len(tl):
            return 0.0
        return float(np.linalg.norm(tl - self.lab[it.colour], axis=1).sum())

    def energy(self) -> float:
        w = self.w
        return (w.cov * self.bad_cells + w.col * self.col + w.loc * self.loc
                + w.glob * self.kl.n * self.kl.value() + w.sup * len(self.unsup))

    # -- edits ----------------------------------------------------------------
    def add(self, it: Item) -> int:
        i = self.next_id
        self.next_id += 1
        self.items[i] = it
        sl = it.cells()
        want = self.t.want[sl]
        self.bad_cells += int((~want).sum()) - int(want.sum())
        self.owner[sl] = i
        self.col += self._colour_cost(it)
        nb = self.neighbours(i)
        self._pairs(i, nb, +1, set())
        self._refresh_support([i] + list(nb))
        return i

    def remove(self, i: int) -> Item:
        it = self.items[i]
        nb = self.neighbours(i)
        self._pairs(i, nb, -1, set())
        sl = it.cells()
        want = self.t.want[sl]
        self.bad_cells -= int((~want).sum()) - int(want.sum())
        self.owner[sl] = -1
        self.col -= self._colour_cost(it)
        del self.items[i]
        self.unsup.discard(i)
        self._refresh_support(list(nb))
        return it

    def _refresh_support(self, ids) -> None:
        for j in ids:
            if j in self.items:
                if self._supported(j, None):
                    self.unsup.discard(j)
                else:
                    self.unsup.add(j)

    def placements(self) -> list[Placement]:
        out = []
        for it in self.items.values():
            rot = 0 if (it.w, it.d) == it.part.footprint else 90
            out.append(Placement(it.part, it.colour, it.x, it.z, it.y, it.w, it.d, rot))
        return out


# ---------------------------------------------------------------------------
# proposals
# ---------------------------------------------------------------------------

class Filler:
    """Fills free wanted cells part by part, each drawn from P(part | the
    parts around it), colour from the target."""

    def __init__(self, state: State, rnd: random.Random):
        self.st, self.rnd = state, rnd
        self.parts = vocabulary()
        self.tok = {p.ldraw: token_of(p) for p in self.parts}
        a_tot = state.s.a_tot
        z = sum(a_tot.values()) or 1.0
        self.prior = {p.ldraw: (a_tot.get(self.tok[p.ldraw], 0) + 1.0) / z for p in self.parts}

    def colour_for(self, sl, part: Part) -> int:
        tc = self.st.t.colour[sl]
        vals = tc[tc >= 0]
        c = int(np.bincount(vals).argmax()) if len(vals) else 0
        return nearest_where(c, lambda code: available(part, code))

    def candidates(self, y: int, x: int, z: int):
        st = self.st
        H, X, Z = st.owner.shape
        out = []
        for p in self.parts:
            for w, d in {(p.footprint[0], p.footprint[1]), (p.footprint[1], p.footprint[0])}:
                if y + p.height_units > H or x + w > X or z + d > Z:
                    continue
                sl = (slice(y, y + p.height_units), slice(x, x + w), slice(z, z + d))
                if (st.owner[sl] >= 0).any() or not st.t.want[sl].all():
                    continue
                out.append((p, w, d, sl))
        return out

    def score(self, p: Part, w: int, d: int, y: int, x: int, z: int) -> float:
        """log P(p | neighbours already placed), up to a constant."""
        st = self.st
        tp = self.tok[p.ldraw]
        s = math.log(self.prior[p.ldraw])
        H, X, Z = st.owner.shape
        seen = set()
        for yy, d_ in ((y - 1, "down"), (y + p.height_units, "up")):
            if 0 <= yy < H:
                for j in set(st.owner[yy, x:x + w, z:z + d].ravel().tolist()):
                    if j >= 0 and j not in seen:
                        seen.add(j)
                        s -= st.s.nll(tp, token_of(st.items[j].part), d_)
        return s

    def fill(self, cells: list[tuple[int, int, int]]) -> list[int]:
        added = []
        cells = sorted(cells)                           # bottom-up: parts land on what's there
        for (y, x, z) in cells:
            if self.st.owner[y, x, z] >= 0:
                continue
            cands = self.candidates(y, x, z)
            if not cands:
                continue
            scores = [self.score(p, w, d, y, x, z) + 0.6 * math.log(w * d * p.height_units)
                      for p, w, d, _ in cands]
            m = max(scores)
            ws = [math.exp(s - m) for s in scores]
            p, w, d, sl = self.rnd.choices(cands, weights=ws)[0]
            added.append(self.st.add(Item(p, self.colour_for(sl, p), y, x, z, w, d)))
        return added


def retile(state: State, filler: Filler, rnd: random.Random):
    """Remove the parts meeting a random window and re-fill its cells.
    Returns an undo function."""
    t = state.t
    ys, xs, zs = np.nonzero(t.want)
    k = rnd.randrange(len(ys))
    cy, cx, cz = int(ys[k]), int(xs[k]), int(zs[k])
    # a small window — one course or plate, a few studs — so a move is local
    hy, hx, hz = rnd.choice((1, 3)), rnd.randint(1, 3), rnd.randint(1, 3)
    H, X, Z = state.owner.shape
    win = state.owner[cy:min(H, cy + hy), max(0, cx - hx + 1):min(X, cx + hx), max(0, cz - hz + 1):min(Z, cz + hz)]
    ids = [j for j in set(win.ravel().tolist()) if j >= 0]
    removed = [(i, state.remove(i)) for i in ids]
    cells = []
    for _, it in removed:
        sl = it.cells()
        for y in range(sl[0].start, sl[0].stop):
            for x in range(sl[1].start, sl[1].stop):
                for z in range(sl[2].start, sl[2].stop):
                    cells.append((y, x, z))
    # wanted cells in the window left empty before also get a chance
    for y in range(cy, min(H, cy + hy)):
        for x in range(max(0, cx - hx + 1), min(X, cx + hx)):
            for z in range(max(0, cz - hz + 1), min(Z, cz + hz)):
                if t.want[y, x, z] and state.owner[y, x, z] < 0:
                    cells.append((y, x, z))
    added = filler.fill(cells)

    def undo():
        for i in added:
            state.remove(i)
        for _, it in removed:
            state.add(it)
    return undo


_BOX: dict = {}


def _box_part(shape: str, h: int, w: int, d: int) -> Optional[Part]:
    """The vocabulary part of this shape and height covering w x d."""
    if not _BOX:
        for p in vocabulary():
            a, b = p.footprint
            _BOX[(p.shape, p.height_units, a, b)] = p
            _BOX[(p.shape, p.height_units, b, a)] = p
    return _BOX.get((shape, h, w, d))


def merge(state: State, filler: Filler, rnd: random.Random):
    """Join a random part with a neighbour of the same shape, height and
    colour beside it into the one real part covering both, where there is
    one.  Returns an undo function, or None when no merge exists."""
    if not state.items:
        return None
    i = rnd.choice(list(state.items))
    a = state.items[i]
    for j, where in state.neighbours(i).items():
        if where != "level":
            continue
        b = state.items[j]
        if (b.part.shape, b.h, b.colour, b.y) != (a.part.shape, a.h, a.colour, a.y):
            continue
        x0, z0 = min(a.x, b.x), min(a.z, b.z)
        x1, z1 = max(a.x + a.w, b.x + b.w), max(a.z + a.d, b.z + b.d)
        if (x1 - x0) * (z1 - z0) != a.w * a.d + b.w * b.d:
            continue                                  # not a rectangle together
        p = _box_part(a.part.shape, a.h, x1 - x0, z1 - z0)
        if p is None or not available(p, palette()[a.colour].code):
            continue
        ra, rb = state.remove(i), state.remove(j)
        k = state.add(Item(p, a.colour, a.y, x0, z0, x1 - x0, z1 - z0))

        def undo(k=k, ra=ra, rb=rb):
            state.remove(k)
            state.add(ra)
            state.add(rb)
        return undo
    return None


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def sample(target: Target, models: list[dict], *, iters: int = 20000, seed: int = 0,
           weights: Optional[Weights] = None, t_start_share: float = 0.05, t_final_share: float = 0.002,
           k_models: int = 50, log_every: int = 2000, warmup: int = 200) -> tuple[list[Placement], dict]:
    """The temperature comes from the data: the median energy change of
    `warmup` proposals (measured, then undone) is the scale; it anneals
    from `t_start_share` to `t_final_share` of it.  Monotony is the KL
    term's to prevent, so the walk may cool close to zero."""
    rnd = random.Random(seed)
    refs = nearest_models(models, target.shape_descriptor(), k_models)
    st = State(target, Stats(refs), weights or Weights())
    filler = Filler(st, rnd)
    ys, xs, zs = np.nonzero(target.want)
    filler.fill(list(zip(ys.tolist(), xs.tolist(), zs.tolist())))
    e = st.energy()
    history = [e]
    deltas = []
    for _ in range(warmup):                         # measure the scale, change nothing
        undo = (merge if rnd.random() < 0.5 else retile)(st, filler, rnd)
        if undo is None:
            continue
        deltas.append(abs(st.energy() - e))
        undo()
    scale = float(np.median([d for d in deltas if d > 0] or [1.0]))
    t_start, t_final = scale * t_start_share, scale * t_final_share
    accepted = 0
    for it in range(iters):
        T = t_start * (t_final / t_start) ** (it / max(1, iters - 1))
        undo = (merge if rnd.random() < 0.5 else retile)(st, filler, rnd)
        if undo is None:
            continue
        e2 = st.energy()
        if e2 <= e or rnd.random() < math.exp(-(e2 - e) / T):
            e = e2
            accepted += 1
        else:
            undo()
        if log_every and (it + 1) % log_every == 0:
            history.append(e)
    info = {"energy": e, "accepted": accepted / max(1, iters), "parts": len(st.items), "t_start": t_start,
            "kl": st.kl.value(), "local_per_pair": st.loc / max(1, st.kl.n),
            "bad_cells": st.bad_cells, "unsupported": len(st.unsup), "history": history,
            "references": [r["model"] for r in refs[:10]]}
    return st.placements(), info
