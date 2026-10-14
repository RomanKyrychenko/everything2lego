"""How much a build's local construction looks like real sets of its kind:
the divergence of its neighbourhood-pattern distribution from the official
models' (tools/partsdb/features.py).

A pattern is (token A, token B, where B is: up / down / level) for two
touching parts, a token is a part's kind and size ("plate1x2h1",
"slope45-2x2h3").  The measure is

    KL(p_build || p_ref) = sum_k p_build(k) * log(p_build(k) / p_ref(k))

It is small when the build uses real sets' patterns in about real sets'
proportions; it grows both for patterns real sets don't make (p_ref small)
and for monotony (all the mass on a few patterns, which a varied reference
gives only part of theirs).  p_ref is smoothed hierarchically: the exact
pattern's frequency, backed off to the same pattern with sizes dropped
(kind only), backed off to uniform — so an unseen but sensible pattern is
not infinitely improbable.

`RunningKL` keeps the divergence under count changes in O(1) per change
(sum c log c, sum c log p_ref, N), for a sampler's local moves.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

_SIZE = re.compile(r"\d+x\d+h\d+$")


def kind_of(token: str) -> str:
    return _SIZE.sub("", token)


@dataclass
class Reference:
    """A smoothed pattern distribution from a set of models' pattern counts."""
    counts: Counter
    lam: tuple[float, float, float] = (0.85, 0.13, 0.02)   # exact, kind-level, uniform
    _total: float = 0.0
    _kind: Counter = field(default_factory=Counter)
    _kind_total: float = 0.0
    _kinds_of: dict = field(default_factory=dict)
    _vocab: int = 1

    def __post_init__(self):
        self._total = float(sum(self.counts.values())) or 1.0
        for (a, b, d), n in self.counts.items():
            self._kind[(kind_of(a), kind_of(b), d)] += n
        self._kind_total = float(sum(self._kind.values())) or 1.0
        # how many exact patterns share each kind-level pattern (to spread its mass)
        spread = Counter((kind_of(a), kind_of(b), d) for (a, b, d) in self.counts)
        self._kinds_of = spread
        self._vocab = max(1, len(self.counts) * 4)

    def p(self, key: tuple[str, str, str]) -> float:
        a, b, d = key
        exact = self.counts.get(key, 0) / self._total
        kk = (kind_of(a), kind_of(b), d)
        kind = self._kind.get(kk, 0) / self._kind_total / (self._kinds_of.get(kk, 0) + 1)
        l1, l2, l3 = self.lam
        return l1 * exact + l2 * kind + l3 / self._vocab

    @classmethod
    def from_models(cls, models: list[dict], key: str = "patterns", **kw) -> "Reference":
        """`key`: "patterns" (touching pairs) or "paths" (pairs through a
        shared neighbour: features.paths)."""
        c = Counter()
        for m in models:
            n = sum(m.get(key, {}).values()) or 1
            for k, v in m.get(key, {}).items():
                c[k] += v / n                       # each model weighs the same
        return cls(c, **kw)


def token_degree(rows: list[dict]) -> dict[str, float]:
    """Each token's mean number of neighbours over these models (features
    rows: their pair patterns and token counts)."""
    deg, n = Counter(), Counter()
    for r in rows:
        for (a, _, _), c in r["patterns"].items():
            deg[a] += c
        n.update(r.get("tokens", {}))
    return {t: deg[t] / n[t] for t in n if n[t]}


def middle_weight(degree: dict[str, float], token: str, default: float = 16.0) -> float:
    """A path's weight through a middle part of this token: 1 / (its mean
    neighbours - 1) — a two-step random walk at the part type's typical
    degree, so a baseplate under a thousand parts counts like any part, and
    the weight does not change as a build's parts come and go."""
    return 1.0 / (max(2.0, degree.get(token, default)) - 1.0)


def kl(patterns: Counter, ref: Reference) -> float:
    n = float(sum(patterns.values()))
    if n == 0:
        return 0.0
    return sum(c / n * math.log((c / n) / ref.p(k)) for k, c in patterns.items() if c > 0)


class RunningKL:
    """KL(p_build || p_ref) under incremental pattern-count changes."""
    EPS = 1e-9

    def __init__(self, ref: Reference, patterns: Counter | None = None):
        self.ref = ref
        self.c: Counter = Counter()
        self.n = 0
        self.s_clogc = 0.0       # sum c log c
        self.s_clogp = 0.0       # sum c log p_ref
        self._logp: dict = {}
        for k, v in (patterns or {}).items():
            self.add(k, v)

    def _lp(self, k) -> float:
        v = self._logp.get(k)
        if v is None:
            v = self._logp[k] = math.log(self.ref.p(k))
        return v

    def add(self, k, dc: float = 1) -> None:
        """Counts may be weights (paths); a count within EPS of 0 is 0."""
        c0 = self.c.get(k, 0)
        c1 = c0 + dc
        if c1 < self.EPS:
            c1 = 0
        if c0 > 0:
            self.s_clogc -= c0 * math.log(c0)
        if c1 > 0:
            self.s_clogc += c1 * math.log(c1)
        self.s_clogp += dc * self._lp(k)
        self.n += dc
        if c1:
            self.c[k] = c1
        else:
            self.c.pop(k, None)

    def value(self) -> float:
        if self.n <= self.EPS:
            return 0.0
        return (self.s_clogc - self.s_clogp) / self.n - math.log(self.n)
