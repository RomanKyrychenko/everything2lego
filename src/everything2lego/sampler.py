"""A universal LEGO generator: sample a build of a target shape whose local
construction follows the statistics of real sets.

The target is a grid (y plates up, x and z studs) of each cell's shape
(`bits`: 4x4x4 samples of the object in it, as one 64-bit mask; a cell a
sloped surface cuts is part full, on the side it is), its colour, which
cells show, and optionally what kind of object it is (building,
road_vehicle, creature, ...).  A state is a set of non-overlapping parts,
each a real part in one of four yaws, with the shape the parts database
measured from its geometry (data/vocabulary.json).

As in neural style transfer, the energy is a content loss plus a style
loss (plus what makes it a build at all):

    E = w_cov  * sum over cells of the share of samples wrong          } content:
      + w_sil  * outline pixels wrong in the three axis views, per     }  the form
                 cell face                                             }
      + w_col  * colour error of the cells that show (Lab, per cell)   }
      + style * N_ref * (w_loc   * mean over touching pairs of -log P(B, where | A)   } style:
                       + w_glob  * KL(pair patterns || reference models)               }  real
                       + w_glob2 * KL(path patterns || reference models)               }  sets
                       + w_dens  * (log(pairs / N_ref))^2 / (2 sd^2)
                       + w_cnt   * (log(parts / P_ref))^2 / (2 sd_p^2))                  }
      + w_sup  * parts no stud holds (a stud of the part below in a
                 hole of this one, or of this one in a hole of the part above)
      + w_side * side studs and holes facing out of the object (a set puts
                 something on every side stud; here nothing can be)

Content.  The volume term fits each cell's samples, so a slope is the part
for a sloped surface (it fills the cells such a surface cuts where the
surface does) and a round brick is not a wall a cell boundary happens to
cut.  The outline term sees the form as it is seen: a piece sticking out
of the object changes the outline in two views, though its few samples of
volume barely change the volume term.

Style.  Matched at two scales, as style transfer matches feature
correlations at more than one layer: pairs of parts that touch (token A,
token B, where B is: up / down / level; a token is a part's kind and size),
and paths of two through a shared neighbour (what sits around a part's
neighbours: a two-step random walk, each path weighted by its middle
part type's typical degree; style.middle_weight, features.paths).  Both from the official models
of the target's kind nearest it in shape, smoothed from exact tokens to
kinds to uniform.  N_ref is the number of touching pairs a real set of the
target's volume has (the references' pairs per occupied cell times the
cells the target fills) — not the build's own count, or fewer, bigger parts
would win for having fewer pairs, however untypical each; the last two
terms ask for real sets' pairs per cell and parts per cell (P_ref: the
references' parts per occupied cell times the cells the target fills —
not too few big parts, not too many small ones), log-normal as the
references spread.  `style` trades style against
content.  Which parts may be used at all, and how readily, comes from how
often sets of the target's kind use them (the vocabulary's per-kind share
and lift) — so a house is not built of wheel arches.

Learned from the official models (tools/learn): the style weights, by
moment matching (Weights.learned: builds of official models' shapes get
the official models' statistics), and the proposals — a network that says
which part comes next from the cell's context (proposals.py).

A move re-tiles a small window: the parts meeting it are removed and its
cells re-filled part by part, each drawn from P(part, yaw | the cell's
context, the fit), or joins two parts into the one real part covering
both.  Metropolis acceptance at a temperature that anneals to a small
`t_final` > 0 — a sample, not the single likeliest build, which would be
monotonous.

    from everything2lego.sampler import Target, sample
    placements, info = sample(Target(want, colours, visible, kind="building"), refs)

The walk itself runs in Rust when the library is built (native/sampler,
everything2lego.native: `python -m everything2lego.native build`), about
a hundred times faster; this module is its reference and fallback, and the
two keep the same energy (tests/test_sampler.py checks it).
"""
from __future__ import annotations

import json
import math
import random
import warnings
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from typing import Optional

import numpy as np

from .bricks import Part
from .colors import _palette_lab, nearest_where, palette
from .io_ldraw import _ROT as _LDRAW_ROT
from .pack import Placement
from . import proposals
from .style import Reference, RunningKL, kind_of, middle_weight, token_degree

# (x' from x, x' from z, z' from x, z' from z): io_ldraw's yaw matrices on the x-z plane
_ROT = {r: (m[0], m[2], m[6], m[8]) for r, m in _LDRAW_ROT.items()}


# ---------------------------------------------------------------------------
# vocabulary: real parts with their measured shapes, as tokens the statistics know
# ---------------------------------------------------------------------------

SUB = 4                                     # samples per cell edge: a cell's shape is a 64-bit mask
FULL = np.uint64(0xFFFFFFFFFFFFFFFF)
_M1, _M2, _M4 = np.uint64(0x5555555555555555), np.uint64(0x3333333333333333), np.uint64(0x0F0F0F0F0F0F0F0F)
_H01 = np.uint64(0x0101010101010101)


def popcount(x: np.ndarray) -> np.ndarray:
    """Set bits per uint64 (SWAR)."""
    x = np.asarray(x, dtype=np.uint64)
    x = x - ((x >> np.uint64(1)) & _M1)
    x = (x & _M2) + ((x >> np.uint64(2)) & _M2)
    x = (x + (x >> np.uint64(4))) & _M4
    return ((x * _H01) >> np.uint64(56)).astype(np.int64)


def pack(fine: np.ndarray, size) -> np.ndarray:
    """(h*4, w*4, d*4) samples -> (h, w, d) masks, bit sy*16 + sx*4 + sz
    (as tools/partsdb/shapes.pack)."""
    h, w, d = size
    b = fine.reshape(h, SUB, w, SUB, d, SUB).transpose(0, 2, 4, 1, 3, 5).reshape(h, w, d, SUB ** 3)
    weights = np.left_shift(np.uint64(1), np.arange(SUB ** 3, dtype=np.uint64))
    return (b.astype(np.uint64) * weights).sum(axis=-1, dtype=np.uint64)


def unpack(bits: np.ndarray) -> np.ndarray:
    h, w, d = bits.shape
    sh = np.arange(SUB ** 3, dtype=np.uint64)
    b = ((bits[..., None] >> sh) & np.uint64(1)).astype(bool)
    return b.reshape(h, w, d, SUB, SUB, SUB).transpose(0, 3, 1, 4, 2, 5).reshape(h * SUB, w * SUB, d * SUB)


def _yaw(rot: int, ux, uz):
    a, b, c, e = _ROT[rot]
    return a * ux + b * uz, c * ux + e * uz


def _rotate(fine: np.ndarray, rot: int) -> np.ndarray:
    """A part's samples in yaw `rot`, turned about its box's centre as
    io_ldraw turns the part."""
    if rot == 0:
        return fine
    H, w, d = fine.shape
    W, D = (w, d) if rot == 180 else (d, w)
    I, K = np.meshgrid(np.arange(W), np.arange(D), indexing="ij")
    vx, vz = I + 0.5 - W / 2, K + 0.5 - D / 2
    a, b, c, e = _ROT[rot]
    ux, uz = a * vx + c * vz, b * vx + e * vz           # the inverse turn
    i = np.round(ux + w / 2 - 0.5).astype(int)
    k = np.round(uz + d / 2 - 0.5).astype(int)
    return fine[:, i, k]


@dataclass(frozen=True, eq=False)
class Orient:
    """A part in one yaw, on the grid: its box (h, W, D) and its occupied
    cells as offsets from the box's minimum corner, with their shapes."""
    rot: int
    size: tuple[int, int, int]
    cells: np.ndarray            # (n, 3) int: y, x, z offsets of the cells it occupies at all
    bits: np.ndarray             # (n,) uint64: each cell's shape
    o: np.ndarray                # (n,) how full each is
    first: tuple[int, int, int]  # its first occupied cell in (y, x, z) order
    top: frozenset               # (x, z) offsets with a stud on top
    bottom: frozenset            # (x, z) offsets with a hole underneath
    sides: tuple = ()            # (hx, hy, hz, dx, dz, kind): a side connector at that point of the face
                                 # facing (dx, dz) — half studs / half plates from the box's corner
    halo: np.ndarray = None      # (m, 3): offsets of the cells next to its cells (26-neighbourhood), not its own
    pix: tuple = ()              # its outline along x, y, z: (sy, sz), (sx, sz), (sy, sx) sample pixels, box-local
    snot: Optional[tuple] = None  # sideways (snot_options): (LDraw 3x3 rotation, row-major; origin box-local LDU x, y-up, z)


def _halo(cells: np.ndarray, size) -> np.ndarray:
    """The cells within one of its cells (diagonals too), its own left out:
    two parts touch when one occupies a cell of the other's halo — the
    same either way round, whatever empty corners their boxes have."""
    h, W, D = size
    occ = np.zeros((h + 2, W + 2, D + 2), bool)
    occ[tuple((cells + 1).T)] = True
    from scipy.ndimage import binary_dilation
    ring = binary_dilation(occ, structure=np.ones((3, 3, 3), bool)) & ~occ
    return np.argwhere(ring) - 1


@dataclass(frozen=True, eq=False)
class VPart:
    part: Part
    token: str
    category: str
    colours: frozenset
    kinds: dict
    n_sets: int
    orients: tuple
    box: Optional[str]           # "brick" / "plate" / "tile" for a plain full box, else None


# side connector kinds, and which fit which (a pin in a pin hole, a wheel's
# hole on a holder's bar, ...)
CONNECTORS = {k: i for i, k in enumerate((
    "stud:M", "antistud:F", "pin:M", "pinhole:F", "axle:M", "axlehole:F", "bar:M", "barhole:F",
    "clip:F", "finger:M", "finger:F", "generic:M", "generic:F"))}
FITS = np.zeros((len(CONNECTORS), len(CONNECTORS)), bool)
for _a, _b in (("stud:M", "antistud:F"), ("pin:M", "pinhole:F"), ("axle:M", "axlehole:F"), ("axle:M", "pinhole:F"),
               ("bar:M", "barhole:F"), ("bar:M", "clip:F"), ("bar:M", "pinhole:F"), ("finger:M", "finger:F"),
               ("generic:M", "generic:F")):
    FITS[CONNECTORS[_a], CONNECTORS[_b]] = FITS[CONNECTORS[_b], CONNECTORS[_a]] = True


def side_cells(o: "Orient", oy: int, ox: int, oz: int, sd: tuple) -> list[tuple[int, int, int]]:
    """The cells a side connector faces: just outside its face, one or (on a
    cell boundary) two along it."""
    hx, hy, hz, dx, dz, _ = sd
    h, W, D = o.size
    y = oy + min(hy // 2, h - 1)
    if dx:
        x = ox + (W if dx > 0 else -1)
        zs = {hz // 2, (hz - 1) // 2} if hz % 2 == 0 else {hz // 2}
        return [(y, x, oz + z) for z in sorted(zs) if 0 <= z < D]
    z = oz + (D if dz > 0 else -1)
    xs = {hx // 2, (hx - 1) // 2} if hx % 2 == 0 else {hx // 2}
    return [(y, ox + x, z) for x in sorted(xs) if 0 <= x < W]


def sides_join(a: "Orient", ay: int, ax: int, az: int, b: "Orient", by: int, bx: int, bz: int) -> bool:
    """Whether a side connector of a meets one of b: facing each other at
    the same point (height within half a plate), and of kinds that fit."""
    for (hx, hy, hz, dx, dz, k) in a.sides:
        if k < 0:
            continue
        px, py, pz = 2 * ax + hx, 2 * ay + hy, 2 * az + hz
        for (gx, gy, gz, ex, ez, kb) in b.sides:
            if (kb >= 0 and FITS[k, kb] and ex == -dx and ez == -dz and 2 * bx + gx == px
                    and 2 * bz + gz == pz and abs(2 * by + gy - py) <= 1):
                return True
    return False


def _orients(bits: np.ndarray, top: list, bottom: list, sides: list = ()) -> tuple:
    h, w, d = bits.shape
    fine = unpack(bits)
    out, seen = [], set()
    for rot in _ROT:
        W, D = (w, d) if rot in (0, 180) else (d, w)

        def to_world(i, k):
            vx, vz = _yaw(rot, i + 0.5 - w / 2, k + 0.5 - d / 2)
            return int(round(vx + W / 2 - 0.5)), int(round(vz + D / 2 - 0.5))
        wbits = pack(_rotate(fine, rot), (h, W, D))
        wt = frozenset(to_world(i, k) for i, k in top)
        wb = frozenset(to_world(i, k) for i, k in bottom)
        face = {"+x": (1, 0), "-x": (-1, 0), "+z": (0, 1), "-z": (0, -1)}
        ws = []
        for kind, f, t, hy in sides:
            fx, fz = face[f]
            # the point on the face, half studs, turned about the box's centre
            px, pz = ((w + fx * w, t) if fx else (t, d + fz * d))
            vx, vz = _yaw(rot, px - w, pz - d)
            dx, dz = _yaw(rot, fx, fz)
            ws.append((int(round(vx + W)), int(hy), int(round(vz + D)), int(round(dx)), int(round(dz)),
                       CONNECTORS.get(kind, -1)))
        ws = tuple(ws)
        key = (wbits.tobytes(), wbits.shape, wt, wb, ws)
        if key in seen:
            continue                                    # a symmetric part: this yaw is another's
        seen.add(key)
        cells = np.argwhere(wbits != 0)
        if not len(cells):
            continue
        cb = wbits[tuple(cells.T)]
        fr = _rotate(fine, rot)
        pix = (np.argwhere(fr.any(axis=1)), np.argwhere(fr.any(axis=0)), np.argwhere(fr.any(axis=2)))
        out.append(Orient(rot, (h, W, D), cells, cb, popcount(cb) / SUB ** 3,
                          tuple(int(v) for v in cells[0]), wt, wb, ws, _halo(cells, (h, W, D)), pix))
    return tuple(out)


@lru_cache(maxsize=1)
def vocabulary() -> tuple[VPart, ...]:
    """Every grid part LEGO makes now (data/vocabulary.json, from the parts
    database: tools/partsdb/vocabulary.py)."""
    raw = json.loads(files("everything2lego.data").joinpath("vocabulary.json").read_text())
    out = []
    for r in raw["parts"]:
        w, d, h = r["size"]
        occ = np.array(r["occ"], dtype=float)           # [y from the bottom][x][z]
        bits = np.array([[[int(v) for v in row] for row in layer] for layer in r["bits"]], dtype=np.uint64)
        part = Part(ldraw=r["ldraw"], bl=r["part_num"], shape=kind_of(r["token"]),
                    footprint=(w, d), height_units=h, name=r["name"], origin=tuple(r["origin"]),
                    companion=r.get("companion"))
        full = bool((occ >= 0.999).all()) and len(r["holes_bottom"]) == w * d and not r["sides"]
        box = None
        if full and r["category"] in ("Bricks", "Plates", "Tiles"):
            box = {"Bricks": "brick", "Plates": "plate", "Tiles": "tile"}[r["category"]]
            if box != "tile" and len(r["studs_top"]) != w * d:
                box = None
        out.append(VPart(part, r["token"], r["category"], frozenset(int(c) for c in r["colours"]),
                         r["kinds"], r["n_sets"], _orients(bits, r["studs_top"], r["holes_bottom"], r["sides"]), box))
    return tuple(out)


@lru_cache(maxsize=1)
def known_kinds() -> frozenset:
    """The kinds of object the vocabulary has statistics for."""
    return frozenset(k for v in vocabulary() for k in v.kinds)


def select_vocabulary(kind: Optional[str], n: int = 160) -> tuple[VPart, ...]:
    """The parts a build of this kind of object draws from: the plain
    bricks, plates and tiles, and the `n` parts sets of that kind use most
    widely (share of its sets) — none its sets never use."""
    vocab = vocabulary()
    def reach(v: VPart) -> float:
        if kind:
            return v.kinds.get(kind, {}).get("set_share", 0.0)
        return float(v.n_sets)
    plain = [v for v in vocab if v.box]
    rest = sorted((v for v in vocab if not v.box and reach(v) > 0), key=reach, reverse=True)[:n]
    return tuple(plain + rest)


# ---------------------------------------------------------------------------
# target and statistics
# ---------------------------------------------------------------------------

@dataclass
class Target:
    want: np.ndarray                  # (H, X, Z) bool: cells to build in
    colour: np.ndarray                # (H, X, Z) palette index (-1 unknown)
    visible: np.ndarray               # (H, X, Z) bool: cells that show
    bits: Optional[np.ndarray] = None  # (H, X, Z) uint64: each cell's shape, 4x4x4 (default: want, full)
    kind: Optional[str] = None        # building, road_vehicle, creature, ... (vocabulary.json kinds)
    region_kind: Optional[np.ndarray] = None   # (H, X, Z) str, per-cell kind; overrides `kind` where set
    region_label: Optional[np.ndarray] = None  # (H, X, Z) int, per-cell architectural label (classify_regions.LABEL); used by masonry_shift and future pattern moves

    def __post_init__(self):
        if self.bits is None:
            self.bits = np.where(self.want, FULL, np.uint64(0))
        self.bits = np.where(self.want, self.bits, np.uint64(0)).astype(np.uint64)
        self.fill = popcount(self.bits) / SUB ** 3      # how full each cell should be
        known = known_kinds()
        if self.kind is not None and self.kind not in known:
            raise ValueError(f"unknown kind {self.kind!r}; one of {', '.join(sorted(known))}")
        if self.region_kind is not None:
            unknown = set(np.unique(self.region_kind)) - known - {"", None}
            if unknown:
                raise ValueError(f"unknown region kinds {unknown}; valid: {', '.join(sorted(known))}")

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
        self._refs = models
        self.ref = Reference.from_models(models)
        self.ref2 = Reference.from_models(models, key="paths")
        # Optional: ref3 is 4-walks (features.paths4).  Only present when the models
        # were precomputed with `features.py paths4`; else fall back to a tiny Reference
        # so State.kl3 stays bounded (effectively unused when the walk never adds to it).
        p4_rows = [m for m in models if "paths4" in m]
        self.ref3 = Reference.from_models(p4_rows, key="paths4") if p4_rows else None
        self.density, self.density_sd = pair_density(models)
        self.part_density, self.part_density_sd = pair_density(models, default=0.3,
                                                               per=lambda m: m.get("n_parts", 0))
        self.degree: dict[str, float] = {}                # set by sample(): over all the models
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

    def middle_w(self, token: str) -> float:
        """A path's weight through a part of this token (style.middle_weight)."""
        return middle_weight(self.degree, token)

    @staticmethod
    def middle_bucket(token_b: str, token_c: str) -> int:
        """8-way hash of the sorted pair of middle kinds; matches features.paths4_bucket.
        Keeps paths4 keys dense."""
        kb, kc = kind_of(token_b), kind_of(token_c)
        pair = (kb, kc) if kb <= kc else (kc, kb)
        return hash(pair) % 8

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


def pair_density(models: list[dict], default: float = 1.0, default_sd: float = 0.5,
                 per=lambda m: sum(m["patterns"].values())) -> tuple[float, float]:
    """Directed touching pairs (or `per`: anything counted per model) per
    occupied cell (stud x stud x plate) in these models (features rows:
    patterns, shape): the median, and the spread of its log over them."""
    d = []
    for m in models:
        w, dd, h, fill = float(np.exp(m["shape"][0])), float(np.exp(m["shape"][1])), float(np.exp(m["shape"][2])), float(m["shape"][3])
        vol = fill * w * dd * h * 2.5
        if vol > 0 and m["patterns"] and per(m):
            d.append(per(m) / vol)
    if not d:
        return default, default_sd
    sd = float(np.std(np.log(d))) if len(d) > 2 else default_sd
    return float(np.median(d)), max(sd, 0.1)


def _ref_volume(row: dict) -> float:
    """A ref model's volume in cells (plate * stud * stud), from its shape
    descriptor (log_w, log_d, log_h_bricks, fill): see pair_density."""
    s = row["shape"]
    return float(np.exp(s[0]) * np.exp(s[1]) * np.exp(s[2]) * s[3] * 2.5)


def nearest_models(rows: list[dict], shape, k: int = 50, kind: Optional[str] = None,
                    target_vol: Optional[float] = None, scale_tol: float = 3.0) -> list[dict]:
    """The `k` models nearest in shape — among those of the same kind of
    object, when there are enough of them.  `target_vol` (cells) filters
    refs whose own volume is within `scale_tol`× (default 3×) of the
    target before the shape nearest-neighbour runs, so a 3000-part target
    doesn't pick its style loss from 300-part sets whose pair density is
    naturally much higher."""
    if kind:
        same = [r for r in rows if r.get("object_kind") == kind]
        if len(same) >= max(1, k // 2):
            rows = same
    if target_vol is not None and target_vol > 0:
        lo, hi = target_vol / scale_tol, target_vol * scale_tol
        scaled = [r for r in rows if lo <= _ref_volume(r) <= hi]
        if len(scaled) >= max(1, k // 2):
            rows = scaled
    X = np.array([r["shape"] for r in rows])
    mu, sd = X.mean(0), X.std(0) + 1e-9
    d = (((X - mu) / sd - (np.asarray(shape) - mu) / sd) ** 2).sum(1)
    return [rows[i] for i in np.argsort(d)[:k]]


# ---------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------

from .snot import SnotChild


@dataclass
class Item:
    vp: VPart
    ori: Orient
    colour: int
    y: int
    x: int
    z: int

    @property
    def h(self) -> int:
        return self.ori.size[0]

    @property
    def W(self) -> int:
        return self.ori.size[1]

    @property
    def D(self) -> int:
        return self.ori.size[2]

    def cells(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        c = self.ori.cells
        return c[:, 0] + self.y, c[:, 1] + self.x, c[:, 2] + self.z


@dataclass
class Weights:
    # content: the form
    cov: float = 4.0
    sil: float = 4.0          # per cell face of outline wrong
    # style: the statistics of real sets, as a whole times `style`
    style: float = 1.0
    glob2: float = 1.0        # path patterns (the second scale)
    glob3: float = 0.0        # 4-walk patterns (the third scale) -- opt-in; needs features.pkl with paths4
    max_deg_paths4: int = 0   # degree cap for the paths4 enumeration (0 = no cap, 10-20 kills baseplate blow-up)
    paths4_sample_n: int = 1  # 1 = all walks, N > 1 = keep ~1/N walks via deterministic hash + reweight
    entropy_glob3: float = 0.0  # anti mode-collapse: energy -= entropy_glob3 * H(kl3) nats
    revkl_glob3: float = 0.0    # coverage: energy -= revkl_glob3 * E_pref3[log(ours+eps)] (anti mode-diffusion)
    col: float = 0.04
    loc: float = 1.0
    glob: float = 1.0
    sup: float = 40.0         # a part no stud holds falls off: all but forbidden
    side: float = 6.0         # a side stud or hole facing out of the object: SNOT with nothing on it
    dens: float = 1.0         # pairs per cell as real sets have them (not too few big, too many small parts)
    cnt: float = 1.0          # parts per cell as real sets have them
    # Proposal biases: added to log P(part) at proposal time, not an energy term.
    # Fitted to match real-set category fractions.  sample()'s tile_boost / snot_boost
    # kwargs still override these if non-zero (for ablations).
    tile_boost: float = 0.0   # bias toward tile parts; +2 lands near real-set tile fraction 15-23%
    snot_boost: float = 0.0   # bias toward SNOT parts (brickS, plateS, tileS); +5 lands near real-set 17-32%
    snot_bonus: float = 0.5   # energy -= snot_bonus * log(1 + models) per snot_insert child; makes the sampler prefer high-popularity SNOT templates
    size: float = 0.0         # n_parts * KL(part-size mix || real sets', sizes.py); native engine only
    sideways: float = 0.0     # n_parts * KL(sideways share || p_sideways); with sample(snot=True), native engine only
    p_sideways: float = 0.10  # real buildings: ~15 % of parts not studs-up; the sideways options cover plates / tiles
    surface: float = 0.0      # > 0: shape term weighted by distance to the target's surface (surface_fit.py, alpha); native only
    big: float = 0.0          # filler proposes bigger parts first: + big * ln(cells it fills) on each candidate; native only

    @classmethod
    def learned(cls, kind: Optional[str] = None) -> "Weights":
        """The weights fitted to the official models (data/energy_weights.json;
        tools/learn/fit_energy.py): those of the object's kind when fitted
        for it, else the pooled ones; the rest as above."""
        f = files("everything2lego.data").joinpath("energy_weights.json")
        if not f.is_file():
            return cls()
        d = json.loads(f.read_text())
        return cls(**d.get("by_kind", {}).get(kind, d["weights"]))


class State:
    def __init__(self, target: Target, stats: Stats, weights: Weights):
        self.t, self.s, self.w = target, stats, weights
        self.owner = np.full(target.want.shape, -1, dtype=np.int32)
        self.cur = np.zeros(target.want.shape, dtype=np.uint64)   # each cell's shape now
        # the part kind built in each cell (proposals.Net.kinds; -1 empty), the network's context
        self.net = proposals.load()
        self.kindgrid = np.full(target.want.shape, -1, dtype=np.int64)
        self.items: dict[int, Item] = {}
        self.snot_children: dict[int, list[SnotChild]] = {}  # parent id -> SNOT decorations (slice 1 inert)
        self.snot_bonus_total: float = 0.0  # Σ log(1+models) over snot children; -w.snot_bonus * this is subtracted from energy
        self.next_id = 0
        self.kl = RunningKL(stats.ref)
        self.loc = 0.0
        self.col = 0.0
        self.unsup: set[int] = set()
        self.bare = 0                                        # side connectors facing out
        self.fid = float(target.fill.sum())                  # wrong samples / 64, everything empty
        self.n_ref = stats.density * float(target.fill.sum())  # pairs a real set this size has
        self.p_ref = stats.part_density * float(target.fill.sum())  # and parts
        self.lab = _palette_lab()
        self.tlab = np.where(target.colour[..., None] >= 0, self.lab[np.maximum(target.colour, 0)], np.nan)
        # the outline: per view (along x, y, z) the target's and how many parts cover each pixel
        fine = unpack(target.bits)
        self.tsil = (fine.any(axis=1), fine.any(axis=0), fine.any(axis=2))
        self.csil = tuple(np.zeros(t.shape, np.int32) for t in self.tsil)
        self.sil = int(sum(t.sum() for t in self.tsil))      # outline pixels wrong, everything empty
        # paths of two through a shared neighbour
        self.adj: dict[int, dict[int, str]] = {}
        self.kl2 = RunningKL(stats.ref2)
        # 4-walks a-b-c-d, keyed by (token_a, token_d, direction:bucket).  Only
        # active when stats.ref3 is present (i.e. features.pkl has `paths4`).
        self.kl3 = RunningKL(stats.ref3) if stats.ref3 is not None else None

    # -- geometry -----------------------------------------------------------
    def neighbours(self, i: int) -> dict[int, str]:
        """{j: where j is relative to i} for parts touching i (a cell of one
        next to a cell of the other, diagonals too, as the official models'
        patterns were counted); where from the two boxes."""
        it = self.items[i]
        H, X, Z = self.owner.shape
        c = it.ori.halo + np.array([it.y, it.x, it.z])
        c = c[(c >= 0).all(axis=1) & (c[:, 0] < H) & (c[:, 1] < X) & (c[:, 2] < Z)]
        found = set(self.owner[c[:, 0], c[:, 1], c[:, 2]].tolist())
        found.discard(-1)
        found.discard(i)
        out: dict[int, str] = {}
        for j in found:
            # from the two boxes, as features.vertical classifies — the same seen from either part
            b = self.items[j]
            out[j] = "up" if b.y >= it.y + it.h else "down" if b.y + b.h <= it.y else "level"
        return out

    def _supported(self, i: int) -> bool:
        """Held by a stud: one of the part below in a hole of this one, or
        one of this one in a hole of the part above (or it is on the ground)."""
        it = self.items[i]
        if it.y == 0:
            return True
        H = self.owner.shape[0]
        for (dx, dz) in it.ori.bottom:
            j = int(self.owner[it.y - 1, it.x + dx, it.z + dz])
            if j >= 0:
                b = self.items[j]
                if b.y + b.h == it.y and (it.x + dx - b.x, it.z + dz - b.z) in b.ori.top:
                    return True
        if it.y + it.h < H:
            for (dx, dz) in it.ori.top:
                j = int(self.owner[it.y + it.h, it.x + dx, it.z + dz])
                if j >= 0:
                    b = self.items[j]
                    if b.y == it.y + it.h and (it.x + dx - b.x, it.z + dz - b.z) in b.ori.bottom:
                        return True
        return self._side_held(it.ori, it.y, it.x, it.z, i)

    def _side_held(self, o: Orient, oy: int, ox: int, oz: int, me: int = -1) -> bool:
        """Held at a side: a pin, stud, bar or axle of it meets one of a part beside it."""
        H, X, Z = self.owner.shape
        for sd in o.sides:
            for (y, x, z) in side_cells(o, oy, ox, oz, sd):
                if 0 <= y < H and 0 <= x < X and 0 <= z < Z:
                    j = int(self.owner[y, x, z])
                    if j >= 0 and j != me:
                        b = self.items[j]
                        if sides_join(o, oy, ox, oz, b.ori, b.y, b.x, b.z):
                            return True
        return False

    # -- energy bookkeeping ---------------------------------------------------
    def _pairs(self, i: int, nb: dict[int, str], sign: int) -> None:
        """Add (sign=+1) or remove (-1) i's pairs with its neighbours, both directions."""
        ti = self.items[i].vp.token
        flip = {"up": "down", "down": "up", "level": "level"}
        for j, d in nb.items():
            tj = self.items[j].vp.token
            for a, b, dd in ((ti, tj, d), (tj, ti, flip[d])):
                self.kl.add((a, b, dd), sign)
                self.loc += sign * self.s.nll(a, b, dd)

    def _pix(self, o: Orient, oy: int, ox: int, oz: int):
        """Its outline pixels in the three views, as indices into them."""
        sy, sx, sz = SUB * oy, SUB * ox, SUB * oz
        px, py, pz = o.pix
        return ((px[:, 0] + sy, px[:, 1] + sz), (py[:, 0] + sx, py[:, 1] + sz), (pz[:, 0] + sy, pz[:, 1] + sx))

    def sil_delta(self, o: Orient, oy: int, ox: int, oz: int) -> int:
        """Change in outline pixels wrong from adding it."""
        d = 0
        for c, t, ix in zip(self.csil, self.tsil, self._pix(o, oy, ox, oz)):
            new = c[ix] == 0
            d += int((~t[ix][new]).sum()) - int(t[ix][new].sum())
        return d

    def _sil(self, o: Orient, oy: int, ox: int, oz: int, sign: int) -> None:
        for c, t, ix in zip(self.csil, self.tsil, self._pix(o, oy, ox, oz)):
            if sign > 0:
                new = c[ix] == 0
                self.sil += int((~t[ix][new]).sum()) - int(t[ix][new].sum())
                c[ix] += 1
            else:
                c[ix] -= 1
                gone = c[ix] == 0
                self.sil += int(t[ix][gone].sum()) - int((~t[ix][gone]).sum())

    @staticmethod
    def where(a: Item, b: Item) -> str:
        """Where b is from a, from the two boxes (features.vertical)."""
        return "up" if b.y >= a.y + a.h else "down" if b.y + b.h <= a.y else "level"

    def _paths(self, i: int, nb: dict[int, str], sign: int) -> None:
        """Add or remove the paths a - b - c that i is in: as the middle
        (every two of its neighbours), and as an end (through each
        neighbour b, to b's other neighbours).  Each weighs b's
        middle_weight; adjacency without i on the far side."""
        it = self.items[i]
        mw = self.s.middle_w
        ids = list(nb)
        w = sign * mw(it.vp.token)
        for a in ids:
            A = self.items[a]
            for c in ids:
                if a != c:
                    C = self.items[c]
                    self.kl2.add((A.vp.token, C.vp.token, self.where(A, C)), w)
        for b in ids:
            wb = sign * mw(self.items[b].vp.token)
            for c in self.adj[b]:
                if c != i:
                    C = self.items[c]
                    self.kl2.add((it.vp.token, C.vp.token, self.where(it, C)), wb)
                    self.kl2.add((C.vp.token, it.vp.token, self.where(C, it)), wb)

    def _paths4(self, i: int, nb: dict[int, str], sign: int) -> None:
        """Add/remove the 4-walks a-b-c-d that i is in.  OMR counts walks per
        middle edge (b, c) with a in nb_b \\ {c}, d in nb_c \\ {b}.  Key is
        (token_a, token_d, "where_d_from_a:mid_bucket"), weight = mw(b)*mw(c).
        Enumerate per position (a=i, b=i, c=i, d=i) and dedupe closed walks."""
        if self.kl3 is None:
            return
        it = self.items[i]
        mw = self.s.middle_w
        mb = self.s.middle_bucket
        ti = it.vp.token
        wi = mw(ti)
        kl3_add = self.kl3.add
        # position 2: i = b (middle-left).  middle edge (i, c), c in nb_i.
        for c in nb:
            C = self.items[c]
            tc = C.vp.token
            wc = mw(tc)
            w = sign * wi * wc
            bucket = mb(ti, tc)
            nb_c = self.adj[c]
            for a in nb:
                if a == c:
                    continue
                A = self.items[a]
                for d in nb_c:
                    if d == i:
                        continue
                    D = self.items[d]
                    key = (A.vp.token, D.vp.token, f"{self.where(A, D)}:{bucket}")
                    kl3_add(key, w)
        # position 3: i = c (middle-right).  middle edge (b, i), b in nb_i.
        for b in nb:
            B = self.items[b]
            tb = B.vp.token
            wb = mw(tb)
            w = sign * wb * wi
            bucket = mb(tb, ti)
            nb_b = self.adj[b]
            for a in nb_b:
                if a == i:
                    continue
                A = self.items[a]
                for d in nb:
                    if d == b:
                        continue
                    D = self.items[d]
                    key = (A.vp.token, D.vp.token, f"{self.where(A, D)}:{bucket}")
                    kl3_add(key, w)
        # position 1: i = a (walk starts at i).  Allow d = i (closed walks counted once here).
        for b in nb:
            B = self.items[b]
            tb = B.vp.token
            wb = mw(tb)
            nb_b = self.adj[b]
            for c in nb_b:
                if c == i:
                    continue                                           # walk's position 2 can't be i (that's position-2 enum)
                C = self.items[c]
                tc = C.vp.token
                w = sign * wb * mw(tc)
                bucket = mb(tb, tc)
                nb_c = self.adj[c]
                for d in nb_c:
                    if d == b:
                        continue
                    D = self.items[d]
                    key = (ti, D.vp.token, f"{self.where(it, D)}:{bucket}")
                    kl3_add(key, w)
        # position 4: i = d (walk ends at i).  Exclude a = i (closed walks already in position 1).
        for c in nb:
            C = self.items[c]
            tc = C.vp.token
            wc = mw(tc)
            nb_c = self.adj[c]
            for b in nb_c:
                if b == i:
                    continue
                B = self.items[b]
                tb = B.vp.token
                w = sign * mw(tb) * wc
                bucket = mb(tb, tc)
                nb_b = self.adj[b]
                for a in nb_b:
                    if a == c or a == i:
                        continue
                    A = self.items[a]
                    key = (A.vp.token, ti, f"{self.where(A, it)}:{bucket}")
                    kl3_add(key, w)

    def colour_cost(self, ys, xs, zs, o, colour: int) -> float:
        vis = self.t.visible[ys, xs, zs]
        if not vis.any():
            return 0.0
        tl = self.tlab[ys[vis], xs[vis], zs[vis]]
        ok = ~np.isnan(tl).any(axis=1)
        if not ok.any():
            return 0.0
        return float((np.linalg.norm(tl[ok] - self.lab[colour], axis=1) * o[vis][ok]).sum())

    def fid_delta(self, ys, xs, zs, bits) -> float:
        """Change in wrong samples (per cell, /64) from filling these empty cells so."""
        t = self.t.bits[ys, xs, zs]
        return float((popcount(bits ^ t) - popcount(t)).sum()) / SUB ** 3

    def bare_sides(self, o: Orient, oy: int, ox: int, oz: int) -> int:
        """Its side studs and holes facing a cell outside the object."""
        f = self.t.fill
        H, X, Z = f.shape
        n = 0
        for sd in o.sides:
            cells = side_cells(o, oy, ox, oz, sd)
            if not any(0 <= y < H and 0 <= x < X and 0 <= z < Z and f[y, x, z] >= 0.5 for y, x, z in cells):
                n += 1
        return n

    def energy(self) -> float:
        w = self.w
        n = self.kl.n
        kl3_v = self.kl3.value() if self.kl3 is not None else 0.0
        ent3 = -w.entropy_glob3 * self.kl3_entropy() if self.kl3 is not None and w.entropy_glob3 != 0.0 else 0.0
        style = (w.loc * self.loc / n + w.glob * self.kl.value() + w.glob2 * self.kl2.value()
                 + w.glob3 * kl3_v + ent3
                 + w.dens * math.log(n / self.n_ref) ** 2 / (2 * self.s.density_sd ** 2)
                 + w.cnt * _dens(len(self.items), self.p_ref, self.s.part_density_sd)) if n > 0 else 0.0
        return (w.cov * self.fid + w.sil * self.sil / SUB ** 2 + w.col * self.col
                + w.style * self.n_ref * style + w.sup * len(self.unsup) + w.side * self.bare
                - w.snot_bonus * self.snot_bonus_total)

    def kl3_entropy(self) -> float:
        """Shannon entropy (nats) of kl3's current count distribution."""
        kl3 = self.kl3
        if kl3 is None or kl3.n <= 1e-9:
            return 0.0
        return math.log(kl3.n) - kl3.s_clogc / kl3.n

    def content_loss(self) -> float:
        w = self.w
        return w.cov * self.fid + w.sil * self.sil / SUB ** 2 + w.col * self.col

    def style_loss(self) -> float:
        return self.energy() - self.content_loss() - self.w.sup * len(self.unsup) - self.w.side * self.bare

    # -- edits ----------------------------------------------------------------
    def add(self, it: Item) -> int:
        i = self.next_id
        self.next_id += 1
        self.items[i] = it
        ys, xs, zs = it.cells()
        self.fid += self.fid_delta(ys, xs, zs, it.ori.bits)
        self.owner[ys, xs, zs] = i
        self.cur[ys, xs, zs] = it.ori.bits
        if self.net is not None:
            self.kindgrid[ys, xs, zs] = self.net.kind_index.get(kind_of(it.vp.token), len(self.net.kinds))
        self.bare += self.bare_sides(it.ori, it.y, it.x, it.z)
        self.col += self.colour_cost(ys, xs, zs, it.ori.o, it.colour)
        self._sil(it.ori, it.y, it.x, it.z, +1)
        nb = self.neighbours(i)
        self._pairs(i, nb, +1)
        self._paths(i, nb, +1)
        flip = {"up": "down", "down": "up", "level": "level"}
        self.adj[i] = dict(nb)
        for j, d in nb.items():
            self.adj[j][i] = flip[d]
        self._paths4(i, nb, +1)                                        # adj must include i first (closed walks i-b-c-i need adj[c] to list i)
        self._refresh_support([i] + list(nb))
        return i

    def remove(self, i: int) -> Item:
        it = self.items[i]
        nb_live = dict(self.adj[i])                                    # i's neighbours while adj still intact
        self._paths4(i, nb_live, -1)                                   # must undo while adj still lists i, mirrors add() order
        nb = self.adj.pop(i)
        for j in nb:
            del self.adj[j][i]
        self._paths(i, nb, -1)
        self._pairs(i, nb, -1)
        self._sil(it.ori, it.y, it.x, it.z, -1)
        ys, xs, zs = it.cells()
        self.owner[ys, xs, zs] = -1
        self.cur[ys, xs, zs] = 0
        self.kindgrid[ys, xs, zs] = -1
        self.bare -= self.bare_sides(it.ori, it.y, it.x, it.z)
        self.fid -= self.fid_delta(ys, xs, zs, it.ori.bits)
        self.col -= self.colour_cost(ys, xs, zs, it.ori.o, it.colour)
        del self.items[i]
        for ch in self.snot_children.pop(i, ()):
            self.snot_bonus_total -= math.log(1 + ch.models)
        self.unsup.discard(i)
        self._refresh_support(list(nb))
        return it

    def _refresh_support(self, ids) -> None:
        for j in ids:
            if j in self.items:
                if self._supported(j):
                    self.unsup.discard(j)
                else:
                    self.unsup.add(j)

    def bad_cells(self) -> int:
        return int((popcount(self.cur ^ self.t.bits) >= SUB ** 3 // 2).sum())

    def placements(self) -> list[Placement]:
        return [Placement(it.vp.part, it.colour, it.x, it.z, it.y, it.W, it.D, it.ori.rot, snot=it.ori.snot)
                for it in self.items.values()]

    def placements_and_decorations(self) -> tuple[list[Placement], dict[Placement, list[SnotChild]]]:
        """Placements plus per-placement SNOT children, keyed by the
        exact Placement object (`id`-based lookup is also safe since
        callers use the dict as `decorations.get(p, ())` with the
        same object references)."""
        ps: list[Placement] = []
        deco: dict[Placement, list[SnotChild]] = {}
        for i, it in self.items.items():
            p = Placement(it.vp.part, it.colour, it.x, it.z, it.y, it.W, it.D, it.ori.rot, snot=it.ori.snot)
            ps.append(p)
            kids = self.snot_children.get(i)
            if kids:
                deco[p] = list(kids)
        return ps, deco


# ---------------------------------------------------------------------------
# proposals
# ---------------------------------------------------------------------------

def option_logprior(vocab: tuple[VPart, ...], stats: Stats, kind: Optional[str],
                     tile_boost: float = 0.0, snot_boost: float = 0.0) -> np.ndarray:
    """Per (part, yaw), in vocabulary order: log P(its token) in the
    references, plus log its lift for the kind of object, plus a pull to
    larger parts (fewer, as sets are).  `tile_boost` / `snot_boost` add
    a log-bonus to every tile / SNOT (brickS, plateS, tileS) option —
    direct levers on tile and SNOT fraction without touching the energy
    or needing a refit."""
    a_tot = stats.a_tot
    z = sum(a_tot.values()) or 1.0
    snot_shapes = {"brickS", "plateS", "tileS"}
    return np.array([
        math.log((a_tot.get(v.token, 0) + 1.0) / z)
        + (math.log(max(1e-3, v.kinds.get(kind, {}).get("lift", 0.0) or 1e-3)) if kind else 0.0)
        + 0.6 * math.log(float(o.o.sum()))
        + (tile_boost if _box_kind(v) == "tile" else 0.0)
        + (snot_boost if v.part.shape in snot_shapes else 0.0)
        + _sideways_prior(v, o)
        for v in vocab for o in v.orients])


def _sideways_prior(v: VPart, o: Orient) -> float:
    """log of how often real sets mount this part sideways (snot_options) on a sideways option, of how often
    not on its upright ones; 0 for parts without sideways options."""
    if not any(x.snot is not None for x in v.orients):
        return 0.0
    from .snot_options import sideways_share
    s = min(0.9, max(0.01, sideways_share().get(v.part.ldraw, 0.05)))
    n_side = sum(x.snot is not None for x in v.orients)
    return math.log(s / n_side) if o.snot is not None else math.log((1 - s) / (len(v.orients) - n_side))


def net_tables(net: "proposals.Net", vocab: tuple[VPart, ...], kind: Optional[str],
               tile_boost: float = 0.0, snot_boost: float = 0.0):
    """Per (part, yaw), in vocabulary order: the network's output for its
    token, and log P(part | token) — how widely sets of this kind (or all
    sets) use it among the parts of its token; and the object's index.
    `tile_boost` / `snot_boost` add a log-bonus to log P(part | token) of
    every tile / SNOT option, biasing both engines' network-driven
    proposals toward them."""
    out, use = [], []
    for v in vocab:
        for _ in v.orients:
            out.append(net.token_index.get(v.token, net.other()))
            use.append((v.kinds.get(kind, {}).get("sets", 0) if kind else v.n_sets) + 1.0)
    out, use = np.array(out), np.array(use)
    tot = np.bincount(out, weights=use, minlength=net.other() + 1)
    lpart = np.log(use / tot[out])
    if tile_boost:
        is_tile = np.array([_box_kind(v) == "tile" for v in vocab for _ in v.orients])
        lpart = lpart + tile_boost * is_tile
    if snot_boost:
        snot_shapes = {"brickS", "plateS", "tileS"}
        is_snot = np.array([v.part.shape in snot_shapes for v in vocab for _ in v.orients])
        lpart = lpart + snot_boost * is_snot
    return out, lpart, (net.object_index.get(kind, -1) if kind else -1)


class Filler:
    """Fills empty cells part by part, each part and yaw drawn from
    P(part | the parts around it) times its fit to the target, colour
    from the target's.  All the (part, yaw) fits at a cell are tested at
    once against a window of the grid."""

    def __init__(self, state: State, rnd: random.Random, vocab: tuple[VPart, ...], kind: Optional[str],
                 top_k: int = 12, net: bool = True, tile_boost: float = 0.0, snot_boost: float = 0.0):
        self.st, self.rnd, self.vocab, self.top_k = state, rnd, vocab, top_k
        self.opts: list[tuple[VPart, Orient]] = [(v, o) for v in vocab for o in v.orients]
        self.logprior = option_logprior(vocab, state.s, kind, tile_boost=tile_boost, snot_boost=snot_boost)
        from . import hinges as _hinges
        vocab_opts = {(v.part.ldraw, o.rot) for v in vocab for o in v.orients}
        # (ldraw_a, yaw_a) -> list of pairs; filtered to pairs whose B is placeable too
        self.pair_by_a = {}
        for key, ps in _hinges.by_a_yaw().items():
            if key not in vocab_opts:
                continue
            kept = [p for p in ps if (p["b_ldraw"], p["yaw_s"]) in vocab_opts]
            if kept:
                self.pair_by_a[key] = kept
        # SNOT templates: {a_ldraw -> list of templates}.  Filtered to anchors
        # whose part is in the current vocab; B doesn't need to be in vocab (it
        # renders as a decoration from its ldraw name).  Empty dict when
        # snot_adj.json is absent or missing the enriched `rot` field.
        from . import snot as _snot
        vocab_ldraws = {v.part.ldraw for v in vocab}
        self.snot_pair_by_a: dict[str, list[dict]] = {}
        for a_ld, tmpls in _snot.by_a_ldraw().items():
            if a_ld not in vocab_ldraws:
                continue
            self.snot_pair_by_a[a_ld] = tmpls
        self.net = state.net if net else None
        if self.net is not None:
            # P(part, yaw | cell) = P(token | context) * P(part | token): the network
            # proposes the token, the part within it goes by how widely sets of this
            # kind (or all sets) use it
            self.opt_out, self.opt_lpart, self.obj = net_tables(self.net, vocab, kind,
                                                                   tile_boost=tile_boost, snot_boost=snot_boost)
            self.static = proposals.static_context(state.t.fill)
        self.prior = self.logprior
        # the window every option is tested in: origin (y, x - FX, z - FZ) for a cell (y, x, z)
        self.FX = max(o.first[1] for _, o in self.opts)
        self.FZ = max(o.first[2] for _, o in self.opts)
        self.win = (max(o.size[0] for _, o in self.opts),
                    max(o.size[1] for _, o in self.opts) + self.FX,
                    max(o.size[2] for _, o in self.opts) + self.FZ)
        WH, WX, WZ = self.win
        n = max(len(o.cells) for _, o in self.opts)
        sentinel = WH * WX * WZ
        self.idx = np.full((len(self.opts), n), sentinel, dtype=np.int64)
        self.bits = np.zeros((len(self.opts), n), dtype=np.uint64)
        solid = np.zeros((len(self.opts), n), bool)
        for k, (_, o) in enumerate(self.opts):
            c = o.cells - np.array(o.first) + np.array([0, self.FX, self.FZ])
            self.idx[k, :len(c)] = np.ravel_multi_index(c.T, self.win)
            self.bits[k, :len(c)] = o.bits
            solid[k, :len(c)] = o.o >= 0.5
        self.solid = solid

    def _window(self, y: int, x: int, z: int):
        st = self.st
        H, X, Z = st.owner.shape
        WH, WX, WZ = self.win
        y0, x0, z0 = y, x - self.FX, z - self.FZ
        free = np.zeros(self.win, bool)
        tb = np.zeros(self.win, dtype=np.uint64)
        sy = slice(y0, min(H, y0 + WH))
        sx = slice(max(0, x0), min(X, x0 + WX))
        sz = slice(max(0, z0), min(Z, z0 + WZ))
        dst = (slice(0, sy.stop - sy.start), slice(sx.start - x0, sx.stop - x0), slice(sz.start - z0, sz.stop - z0))
        free[dst] = st.owner[sy, sx, sz] < 0
        tb[dst] = st.t.bits[sy, sx, sz]
        return np.append(free.ravel(), True), np.append(tb.ravel(), np.uint64(0)), (y0, x0, z0)

    def candidates(self, y: int, x: int, z: int):
        """(option index, box origin, residual) for every part and yaw whose
        first cell can go at (y, x, z): its cells free and none solid where
        the target is empty.  The residual is the fit error it leaves on the
        cells it claims, its wrong samples (/64 per cell) — a part's claim is
        final (a slope's half-empty cells take nothing else), so parts are
        compared by what they leave wrong, not by how much they cover."""
        free, tb, (y0, x0, z0) = self._window(y, x, z)
        T = tb[self.idx]
        ok = free[self.idx].all(axis=1) & ~(self.solid & (T == 0)).any(axis=1)
        resid = popcount(self.bits ^ T).sum(axis=1) / SUB ** 3
        # the detailed score (neighbours, colour, support) only for the
        # likeliest few by prior and fit
        # style and content weighed as in the energy, so a typical part that
        # fits a little worse is proposed when style counts for more
        pre = np.where(ok, self.st.w.style * self.prior - self.st.w.cov * resid, -np.inf)
        if ok.sum() > self.top_k:
            ok &= pre >= np.partition(pre, -self.top_k)[-self.top_k]
        out = []
        for k in np.flatnonzero(ok):
            o = self.opts[k][1]
            out.append((int(k), (y - o.first[0], x - o.first[1], z - o.first[2]), float(resid[k])))
        return out

    def colour_for(self, ys, xs, zs, vp: VPart) -> int:
        tc = self.st.t.colour[ys, xs, zs]
        vals = tc[tc >= 0]
        c = int(np.bincount(vals).argmax()) if len(vals) else 0
        return nearest_where(c, lambda code: code in vp.colours)

    def cell_prior(self, y: int, x: int, z: int) -> np.ndarray:
        """Per option, log P(it | the cell's context) from the network."""
        net = self.net
        obj = self.obj
        rk = self.st.t.region_kind
        if rk is not None and rk[y, x, z]:
            obj = net.object_index.get(str(rk[y, x, z]), obj)
        ctx = proposals.features(self.st.t.fill, self.st.kindgrid, self.static, y, x, z, obj,
                                 len(net.kinds), len(net.objects))
        return net.logprobs(ctx[None])[0][self.opt_out] + self.opt_lpart

    def stat_score(self, k: int, oy: int, ox: int, oz: int) -> float:
        """log P(part | the parts already above and below its box) — or, with
        the network, log P(part | the cell's context)."""
        if self.net is not None:
            return float(self.prior[k])
        st = self.st
        vp, o = self.opts[k]
        h, W, D = o.size
        H = st.owner.shape[0]
        s = float(self.logprior[k])
        seen = set()
        for yy, d_ in ((oy - 1, "down"), (oy + h, "up")):
            if 0 <= yy < H:
                for j in set(st.owner[yy, ox:ox + W, oz:oz + D].ravel().tolist()):
                    if j >= 0 and j not in seen:
                        seen.add(j)
                        s -= st.s.nll(vp.token, st.items[j].vp.token, d_)
        return s

    def held(self, o: Orient, oy: int, ox: int, oz: int) -> bool:
        """Whether a stud of a part already below would sit in one of its holes."""
        if oy == 0:
            return True
        st = self.st
        for (dx, dz) in o.bottom:
            j = int(st.owner[oy - 1, ox + dx, oz + dz])
            if j >= 0:
                b = st.items[j]
                if b.y + b.h == oy and (ox + dx - b.x, oz + dz - b.z) in b.ori.top:
                    return True
        return st._side_held(o, oy, ox, oz)

    def capped(self, o: Orient, oy: int, ox: int, oz: int) -> float:
        """The share of its top with more to build above but no stud there
        (a tile inside a wall): what goes on it would hang on nothing."""
        t = self.st.t
        h, W, D = o.size
        if oy + h >= t.fill.shape[0]:
            return 0.0
        tf = t.fill[oy + h]
        H_, D_ = tf.shape
        above = np.zeros((W, D), bool)
        x0, x1 = max(0, ox), min(H_, ox + W)             # clip to the target grid
        z0, z1 = max(0, oz), min(D_, oz + D)
        above[x0 - ox:x1 - ox, z0 - oz:z1 - oz] = tf[x0:x1, z0:z1] >= 0.5
        if not above.any():
            return 0.0
        studs = np.zeros((W, D), bool)
        for dx, dz in o.top:
            if 0 <= dx < W and 0 <= dz < D:
                studs[dx, dz] = True
        return float((above & ~studs).sum()) / (W * D)

    def fill(self, cells: list[tuple[int, int, int]]) -> list[int]:
        st, w = self.st, self.st.w
        added = []
        for (y, x, z) in sorted(cells):                   # bottom-up: parts land on what's there
            if st.owner[y, x, z] >= 0:
                continue
            if self.net is not None:
                self.prior = self.cell_prior(y, x, z)
            cands = self.candidates(y, x, z)
            if not cands:
                continue
            scored = []
            for k, (oy, ox, oz), resid in cands:
                vp, o = self.opts[k]
                ys, xs, zs = o.cells[:, 0] + oy, o.cells[:, 1] + ox, o.cells[:, 2] + oz
                colour = self.colour_for(ys, xs, zs, vp)
                stat = w.style * self.stat_score(k, oy, ox, oz)
                fit = (-w.cov * resid - w.sil * st.sil_delta(o, oy, ox, oz) / SUB ** 2
                       - w.col * st.colour_cost(ys, xs, zs, o.o, colour))
                if not self.held(o, oy, ox, oz):
                    fit -= w.sup
                fit -= w.sup * self.capped(o, oy, ox, oz) + w.side * st.bare_sides(o, oy, ox, oz)
                scored.append((stat, fit, k, (oy, ox, oz), colour))
            best_stat = max(s[0] for s in scored)
            scores = [s + f for s, f, *_ in scored]
            if st.t.fill[y, x, z] < 0.5:                   # a barely-wanted cell may stay empty
                scores.append(best_stat - w.cov * float(st.t.fill[y, x, z]))
                scored.append(None)
            m = max(scores)
            pick = self.rnd.choices(scored, weights=[math.exp(s - m) for s in scores])[0]
            if pick is None:
                continue
            _, _, k, (oy, ox, oz), colour = pick
            vp, o = self.opts[k]
            added.append(st.add(Item(vp, o, colour, oy, ox, oz)))
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
    sy, sx, sz = slice(cy, min(H, cy + hy)), slice(max(0, cx - hx + 1), min(X, cx + hx)), slice(max(0, cz - hz + 1), min(Z, cz + hz))
    ids = [j for j in set(state.owner[sy, sx, sz].ravel().tolist()) if j >= 0]
    removed = [state.remove(i) for i in ids]
    cells = set()
    for it in removed:
        cyy, cxx, czz = it.cells()
        cells.update(zip(cyy.tolist(), cxx.tolist(), czz.tolist()))
    wy, wx, wz = np.nonzero(t.want[sy, sx, sz] & (state.owner[sy, sx, sz] < 0))
    cells.update(zip((wy + sy.start).tolist(), (wx + sx.start).tolist(), (wz + sz.start).tolist()))
    added = filler.fill(list(cells))

    def undo():
        for i in added:
            state.remove(i)
        for it in removed:
            state.add(it)
    return undo


@lru_cache(maxsize=8)
def _boxes(vocab: tuple[VPart, ...]) -> dict:
    """(kind, h, W, D) -> (part, yaw) for the plain box parts."""
    out = {}
    for v in vocab:
        if v.box:
            for o in v.orients:
                out.setdefault((v.box,) + o.size, (v, o))
    return out


def merge(state: State, filler: Filler, rnd: random.Random):
    """Join a random plain part with a neighbour of the same kind, height
    and colour beside it into the one real part covering both, where there
    is one.  Returns an undo function, or None when no merge exists."""
    if not state.items:
        return None
    i = rnd.choice(list(state.items))
    a = state.items[i]
    if not a.vp.box:
        return None
    boxes = _boxes(filler.vocab)
    # among the neighbours a can merge with, pick the largest (bigger merge =
    # one more part gone, which the density term asks for)
    best = None
    for j, where in list(state.adj[i].items()):
        if where != "level":
            continue
        b = state.items[j]
        if (b.vp.box, b.h, b.colour, b.y) != (a.vp.box, a.h, a.colour, a.y):
            continue
        x0, z0 = min(a.x, b.x), min(a.z, b.z)
        x1, z1 = max(a.x + a.W, b.x + b.W), max(a.z + a.D, b.z + b.D)
        if (x1 - x0) * (z1 - z0) != a.W * a.D + b.W * b.D:
            continue                                  # not a rectangle together
        got = boxes.get((a.vp.box, a.h, x1 - x0, z1 - z0))
        if got is None or palette()[a.colour].code not in got[0].colours:
            continue
        area = (x1 - x0) * (z1 - z0)
        if best is None or area > best[0]:
            best = (area, j, x0, z0, x1, z1, got)
    if best is None:
        return None
    _, j, x0, z0, x1, z1, got = best
    ra, rb = state.remove(i), state.remove(j)
    k = state.add(Item(got[0], got[1], a.colour, a.y, x0, z0))

    def undo(k=k, ra=ra, rb=rb):
        state.remove(k)
        state.add(ra)
        state.add(rb)
    return undo


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def _box_kind(vp: "VPart") -> Optional[str]:
    """Which plain shape family a part belongs to (box, side-stud, round all
    count as 'brick'/'plate'/'tile'), for swap()'s candidate lookup."""
    k = {"brick": "brick", "brickS": "brick", "brickR": "brick",
         "plate": "plate", "plateS": "plate", "plateR": "plate",
         "tile": "tile", "tileS": "tile", "tileR": "tile"}.get(vp.part.shape)
    return None if vp.part.companion else k


@lru_cache(maxsize=8)
def _variants(vocab: tuple["VPart", ...]) -> dict:
    """(box_kind, size, cells positions) -> [(VPart, Orient)] of same shape
    and same cells (halo matches), with and without side connectors.  swap()
    picks one of these at random.  Parts whose cells differ even at the same
    box size (a corner plate vs a full plate) are in different groups: a
    swap between them would change the halo and leak energy."""
    out: dict = {}
    for vp in vocab:
        k = _box_kind(vp)
        if k is None:
            continue
        for o in vp.orients:
            cells_sig = tuple(sorted(map(tuple, o.cells.tolist())))
            out.setdefault((k, o.size, cells_sig), []).append((vp, o))
    return out


def swap(state: "State", filler: "Filler", rnd: random.Random):
    """Replace a plain part at (y, x, z) with another part of the same box
    kind and size (a side-stud variant, a round brick, a jumper plate).
    The Metropolis step accepts when it helps (a new side connector meets a
    neighbour, bare_sides goes down); it drops when it doesn't."""
    if not state.items:
        return None
    i = rnd.choice(list(state.items))
    a = state.items[i]
    k = _box_kind(a.vp)
    if k is None:
        return None
    variants = _variants(filler.vocab).get((k, a.ori.size), [])
    if len(variants) < 2:
        return None
    for _ in range(4):                                   # try a few times to find a different one
        v, o = rnd.choice(variants)
        if v is not a.vp:
            break
    else:
        return None
    if palette()[a.colour].code not in v.colours:
        return None
    ra = state.remove(i)
    j = state.add(Item(v, o, a.colour, a.y, a.x, a.z))

    def undo(j=j, ra=ra):
        state.remove(j)
        state.add(ra)
    return undo


def _dens(pairs: float, n_ref: float, sd: float) -> float:
    """The density term's statistic: log(pairs / N_ref)^2 / (2 sd^2)."""
    return math.log(pairs / n_ref) ** 2 / (2 * sd ** 2) if pairs > 0 else 0.0


def _place_init(st: "State", vocab: tuple["VPart", ...], placements: list[Placement]) -> int:
    """Pre-populate `st` with placements before sampling (multi-scale
    warm-start).  Returns the count actually placed; skips a placement
    whose part isn't in `vocab`, whose cells go out of bounds, whose
    cells aren't in `target.want`, or whose cells are already owned."""
    by_ldraw: dict[str, VPart] = {}
    for v in vocab:
        by_ldraw.setdefault(v.part.ldraw, v)
    H, X, Z = st.owner.shape
    placed = 0
    for p in placements:
        v = by_ldraw.get(p.part.ldraw)
        if v is None:
            continue
        ori = next((o for o in v.orients if o.rot == p.rotation_deg), None)
        if ori is None:
            continue
        cy = ori.cells[:, 0] + p.y_plate
        cx = ori.cells[:, 1] + p.x_stud
        cz = ori.cells[:, 2] + p.z_stud
        if (cy < 0).any() or (cy >= H).any() or (cx < 0).any() or (cx >= X).any() \
           or (cz < 0).any() or (cz >= Z).any():
            continue
        if (st.owner[cy, cx, cz] >= 0).any() or not st.t.want[cy, cx, cz].all():
            continue
        st.add(Item(v, ori, p.color_idx, p.y_plate, p.x_stud, p.z_stud))
        placed += 1
    return placed


def masonry_shift(state: "State", filler: "Filler", rnd: random.Random):
    """When a long rectangular wall-region part has its horizontal seams
    aligned with the part directly above (same (x_stud, z_stud, w, d)),
    shift this part by 1 stud along its long axis to stagger the seam
    (brick-course masonry pattern).  Requires target.region_label to be
    set (classify_regions.LABEL): only fires when the item's cells are
    dominantly wall (label 1).  Metropolis accepts when the shift
    doesn't hurt energy."""
    region = state.t.region_label
    if region is None or not state.items:
        return None
    H, X, Z = state.owner.shape
    i = rnd.choice(list(state.items))
    a = state.items[i]
    # must be a long rectangular box part
    if a.W + a.D < 3 or a.W * a.D > 8:
        return None
    cy, cx, cz = a.cells()
    sub = region[cy, cx, cz]
    vals = sub[sub >= 0]
    if not len(vals) or int(np.bincount(vals).argmax()) != 1:   # 1 = wall
        return None
    # find a part B above whose cells overlap A's top face at the same (x, z)
    y_top = a.y + a.h
    if y_top >= H:
        return None
    above = {int(state.owner[y_top, x, z]) for x in range(a.x, a.x + a.W) for z in range(a.z, a.z + a.D)}
    above.discard(-1)
    if not above:
        return None
    j = next(iter(above))
    b = state.items[j]
    if (b.y != y_top or b.x != a.x or b.z != a.z or b.W != a.W or b.D != a.D):
        return None                                    # seam not aligned
    # pick the long axis and a direction to shift
    along_x = a.W >= a.D
    for d in (1, -1):
        nx = a.x + (d if along_x else 0)
        nz = a.z + (0 if along_x else d)
        # new cells
        ncy = a.ori.cells[:, 0] + a.y
        ncx = a.ori.cells[:, 1] + nx
        ncz = a.ori.cells[:, 2] + nz
        if (ncx < 0).any() or (ncx >= X).any() or (ncz < 0).any() or (ncz >= Z).any():
            continue
        # cells must be free (ignoring a's own) and in want
        owner_cells = state.owner[ncy, ncx, ncz]
        if not ((owner_cells == -1) | (owner_cells == i)).all():
            continue
        if not state.t.want[ncy, ncx, ncz].all():
            continue
        ra = state.remove(i)
        k = state.add(Item(a.vp, a.ori, a.colour, a.y, nx, nz))

        def undo(k=k, ra=ra):
            state.remove(k)
            state.add(ra)
        return undo
    return None


def hinge_swap(state: "State", filler: "Filler", rnd: random.Random):
    """Attach B at a catalogued pose to an existing A in the state.
    Pair sources: hinges.load_pairs (step 4A/B of the hinge feature) and
    catalogue.load_pairs (top mined pair patterns from OMR).  Fails
    silently when no (ldraw, yaw) match, target B isn't in vocabulary,
    or any target cell is already filled by a different part.  Metropolis
    accepts when the pair improves style/support/outline over the
    previous occupant (if any) at B's cells."""
    pairs_by_a = filler.pair_by_a
    if not pairs_by_a or not state.items:
        return None
    # pick an existing item whose (ldraw, yaw) matches a pair's A side
    candidates = [i for i, it in state.items.items()
                   if (it.vp.part.ldraw, it.ori.rot) in pairs_by_a]
    if not candidates:
        return None
    i = rnd.choice(candidates)
    a = state.items[i]
    pair = rnd.choice(pairs_by_a[(a.vp.part.ldraw, a.ori.rot)])
    # find a VPart for B's ldraw name
    b_vp = next((v for v in filler.vocab if v.part.ldraw == pair["b_ldraw"]), None)
    if b_vp is None:
        return None
    yaw = pair["yaw_s"]
    b_ori = next((o for o in b_vp.orients if o.rot == yaw), None)
    if b_ori is None:
        return None
    # compute B's sampler position
    if pair["stacked"]:
        by, bx, bz = a.y + a.h, a.x, a.z
    else:
        by, bx, bz = a.y + pair["dy_s"], a.x + pair["dx_s"], a.z + pair["dz_s"]
    H, X, Z = state.owner.shape
    cy = b_ori.cells[:, 0] + by
    cx = b_ori.cells[:, 1] + bx
    cz = b_ori.cells[:, 2] + bz
    if (cy < 0).any() or (cy >= H).any() or (cx < 0).any() or (cx >= X).any() \
       or (cz < 0).any() or (cz >= Z).any():
        return None
    # cells must be empty or all owned by one evictable item (not A itself)
    occ_ids = {int(state.owner[y, x, z]) for y, x, z in zip(cy, cx, cz)}
    occ_ids.discard(-1)
    occ_ids.discard(i)                                                # don't evict A
    if len(occ_ids) > 1:
        return None                                                   # mixed; too complex
    evicted = None
    if occ_ids:
        j = next(iter(occ_ids))
        # only evict if ALL its cells are inside B's footprint (don't orphan)
        oy, ox, oz = state.items[j].cells()
        if not np.all(state.owner[oy, ox, oz] == j):
            return None
        if not set(zip(oy.tolist(), ox.tolist(), oz.tolist())).issubset(
                set(zip(cy.tolist(), cx.tolist(), cz.tolist()))):
            return None
        evicted = state.remove(j)
    # pick a colour that matches A's for a tight hinge pair
    colour = a.colour if (b_vp.colours is None or palette()[a.colour].code in b_vp.colours) else next(
        iter(b_vp.colours), None)
    if colour is None or palette()[colour if isinstance(colour, int) else 0].code not in (b_vp.colours or set()):
        colour = a.colour
    # if the chosen colour isn't actually in B's allowed set, bail
    if b_vp.colours and palette()[colour].code not in b_vp.colours:
        if evicted is not None:
            state.add(evicted)
        return None
    k = state.add(Item(b_vp, b_ori, colour, by, bx, bz))

    def undo(k=k, evicted=evicted):
        state.remove(k)
        if evicted is not None:
            state.add(evicted)
    return undo


def snot_insert(state: "State", filler: "Filler", rnd: random.Random):
    """Attach a SNOT decoration (B child) to an existing anchor (A) using a
    real-set template from snot_adj.json via snot.by_a_ldraw.  Decoration
    only in slice 2: the child doesn't claim cells and energy is
    unchanged, so Metropolis always accepts a successful placement.  At
    most one child per anchor to keep decoration density bounded.
    Template drawn in proportion to how many OMR models carry it."""
    pairs_by_a = filler.snot_pair_by_a
    if not pairs_by_a or not state.items:
        return None
    candidates = [i for i, it in state.items.items()
                  if it.vp.part.ldraw in pairs_by_a and i not in state.snot_children]
    if not candidates:
        return None
    i = rnd.choice(candidates)
    a = state.items[i]
    tmpls = pairs_by_a[a.vp.part.ldraw]
    # draw a template weighted by its model count
    total = 0
    for t in tmpls:
        total += int(t.get("models", 1))
    if total <= 0:
        return None
    r = rnd.random() * total
    acc = 0
    tmpl = tmpls[-1]
    for t in tmpls:
        acc += int(t.get("models", 1))
        if r <= acc:
            tmpl = t
            break
    b_vp = next((v for v in filler.vocab if v.part.ldraw == tmpl["b_ldraw"]), None)
    colour = a.colour
    if b_vp is not None and b_vp.colours:
        if palette()[a.colour].code not in b_vp.colours:
            colour = next((c for c in range(len(palette()))
                           if palette()[c].code in b_vp.colours), a.colour)
    from .bricks import LDU_PER_PLATE as _LDUP, LDU_PER_STUD as _LDUS
    child = SnotChild(
        b_ldraw=tmpl["b_ldraw"],
        b_colour=palette()[colour].code,
        rot=tmpl["rot"],
        dy_LDU=int(tmpl["dy_s"]) * _LDUP,
        dx_LDU=int(tmpl["dx_s"]) * _LDUS,
        dz_LDU=int(tmpl["dz_s"]) * _LDUS,
        models=int(tmpl.get("models", 0)),
    )
    state.snot_children.setdefault(i, []).append(child)
    state.snot_bonus_total += math.log(1 + child.models)

    def undo(i=i, child=child):
        if i in state.snot_children:
            try:
                state.snot_children[i].remove(child)
            except ValueError:
                return
            state.snot_bonus_total -= math.log(1 + child.models)
            if not state.snot_children[i]:
                del state.snot_children[i]
    return undo


def sample(target: Target, models: list[dict] | None = None, *, iters: int = 20000, seed: int = 0,
           weights: Optional[Weights] = None, t_start_share: float = 0.05, t_final_share: float = 0.002,
           k_models: int = 50, n_special: int = 160, log_every: int = 2000,
           warmup: int = 200, top_k: int = 12, engine: str = "auto",
           net: bool = True, stats: Optional["Stats"] = None,
           tile_boost: float = 0.0, snot_boost: float = 0.0,
           init_placements: Optional[list[Placement]] = None,
           opt_bias: Optional[np.ndarray] = None, snot: bool = False) -> tuple[list[Placement], dict]:
    """The temperature comes from the data: the median energy change of
    `warmup` proposals (measured, then undone) is the scale; it anneals
    from `t_start_share` to `t_final_share` of it.  Monotony is the KL
    term's to prevent, so the walk may cool close to zero.

    `net`: propose parts with the learned network (proposals.py) when there
    is one, else by the pair statistics.

    `engine`: "native" runs the walk in Rust (native/sampler, everything2lego.native),
    "python" here; "auto" the native one when it is built."""
    from . import native
    if stats is None:
        refs = nearest_models(models, target.shape_descriptor(), k_models, target.kind,
                              target_vol=float(target.want.sum()))
        stats = Stats(refs)
        stats.degree = token_degree(models)
    else:
        refs = stats._refs                               # for info["references"]
    weights = weights or Weights.learned(target.kind)
    # Default proposal biases from fitted Weights when caller didn't pass an override.
    if tile_boost == 0.0:
        tile_boost = getattr(weights, "tile_boost", 0.0)
    if snot_boost == 0.0:
        snot_boost = getattr(weights, "snot_boost", 0.0)
    vocab = select_vocabulary(target.kind, n_special)
    if snot:                                             # small plates / tiles also on their side, held by side studs
        from .snot_options import with_sideways
        vocab = with_sideways(vocab)
    if engine == "auto" and not native.available():
        warnings.warn(f"native sampler: {native.why_not()} (python -m everything2lego.native build); "
                      "running the Python walk, about a hundred times slower", stacklevel=2)
    if engine == "native" or (engine == "auto" and native.available()):
        got, info, history = native.run(
            target, stats, vocab, target.kind, weights, iters=iters, warmup=warmup, seed=seed,
            t_start_share=t_start_share, t_final_share=t_final_share, top_k=top_k, log_every=log_every,
            opts_logprior=option_logprior(vocab, stats, target.kind, tile_boost=tile_boost,
                                           snot_boost=snot_boost)
                           + (opt_bias if opt_bias is not None else 0.0),
            net=net, tile_boost=tile_boost, snot_boost=snot_boost,
            init_placements=init_placements)
        placements = [Placement(v.part, c, x, z, y, o.size[1], o.size[2], o.rot, snot=o.snot) for v, o, c, y, x, z in got]
        used = Counter(v.category for v, *_ in got)
        shapes = Counter(v.part.shape for v, *_ in got)
        n_ref = stats.density * float(target.fill.sum())
        p_ref = stats.part_density * float(target.fill.sum())
        info.update({"history": history, "vocabulary": len(vocab), "categories": dict(used),
                     "shapes": dict(shapes), "n_ref": n_ref,
                     "dens": _dens(info["pairs"], n_ref, stats.density_sd),
                     "cnt": _dens(info["parts"], p_ref, stats.part_density_sd),
                     "references": [r["model"] for r in refs[:10]], "engine": "native"})
        return placements, info
    rnd = random.Random(seed)
    st = State(target, stats, weights)
    filler = Filler(st, rnd, vocab, target.kind, top_k, net, tile_boost=tile_boost, snot_boost=snot_boost)
    # pre-populate state with warm-start placements (multi-scale: scaled-up coarse)
    if init_placements:
        _place_init(st, vocab, init_placements)
    ys, xs, zs = np.nonzero(target.want)
    cells = [(y, x, z) for y, x, z in zip(ys.tolist(), xs.tolist(), zs.tolist())
             if st.owner[y, x, z] < 0]
    filler.fill(cells)
    e = st.energy()
    history = [e]
    deltas = []
    for _ in range(warmup):                         # measure the scale, change nothing
        r = rnd.random()
        move = (masonry_shift if r < 0.02 else hinge_swap if r < 0.05
                 else snot_insert if r < 0.08
                 else swap if r < 0.15 else merge if r < 0.60 else retile)
        undo = move(st, filler, rnd)
        if undo is None:
            continue
        deltas.append(abs(st.energy() - e))
        undo()
    scale = float(np.median([d for d in deltas if d > 0] or [1.0]))
    t_start, t_final = scale * t_start_share, scale * t_final_share
    accepted = 0
    for it in range(iters):
        T = t_start * (t_final / t_start) ** (it / max(1, iters - 1))
        r = rnd.random()
        move = (masonry_shift if r < 0.02 else hinge_swap if r < 0.05
                 else snot_insert if r < 0.08
                 else swap if r < 0.15 else merge if r < 0.60 else retile)
        undo = move(st, filler, rnd)
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
    used = Counter(v.category for v in (i.vp for i in st.items.values()))
    shapes = Counter(v.part.shape for v in (i.vp for i in st.items.values()))
    snot_count = sum(len(v) for v in st.snot_children.values())
    info = {"energy": e, "accepted": accepted / max(1, iters), "parts": len(st.items), "t_start": t_start,
            "shapes": dict(shapes),
            "kl": st.kl.value(), "kl_paths": st.kl2.value(),
            "kl_paths4": (st.kl3.value() if st.kl3 is not None else 0.0),
            "entropy_paths4": st.kl3_entropy(),
            "local_per_pair": st.loc / max(1, st.kl.n),
            "fidelity": st.fid, "outline": st.sil / SUB ** 2,
            "content_loss": st.content_loss(), "style_loss": st.style_loss(),
            "bad_cells": st.bad_cells(), "unsupported": len(st.unsup), "bare_side_connectors": st.bare, "history": history,
            "vocabulary": len(vocab), "categories": dict(used),
            "snot_decorations": snot_count,
            "references": [r["model"] for r in refs[:10]], "engine": "python", "n_ref": st.n_ref,
            "pairs": st.kl.n, "dens": _dens(st.kl.n, st.n_ref, stats.density_sd),
            "cnt": _dens(len(st.items), st.p_ref, stats.part_density_sd)}
    ps, deco = st.placements_and_decorations()
    if deco:
        info["_snot_decorations"] = deco                       # kept for the LDR writer; drop before JSON dump
    return ps, info
