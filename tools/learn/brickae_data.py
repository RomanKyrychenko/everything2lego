"""Brick autoencoder data: 50-brick windows <-> per-voxel label grids (Stage 0c).

A window is a box of (H plates, W studs, D studs) cells.  Every cell of a brick
carries five labels, from which the brick is recovered without any network
involvement:

    occ   : 1 if a brick covers the cell
    off   : (3) cell - anchor, anchor = the brick's min-corner cell, each in 0..OFF_MAX
    rot   : one of the 48 axis-aligned rotations, 24 proper + 24 mirrored (SNOT lives here)
    cls   : part vocabulary id (exact LDraw part for the frequent ones, else a
            (w,h,d,category) class)
    col   : LDraw colour bin

Cells come from the parts database's per-cell part masks (`part_shape.occ`, the
visual hull from three axis views: arches and frames are hollow) and fall back
to the body box.  Decoding groups cells by (anchor, cls, rot) and validates the
group against the class's cell count (PointGroup-style).  Connections are
checked later by geometry.

Item hygiene (OMR files are messy): BrickNet's alias table renames old parts
(3023 -> 3023b, with a wrap matrix), mildly scaled sub-model transforms are
orthonormalised, hoses / primitives / 3x-scaled parts are dropped, and round
parts turned about their own axis are snapped to the grid.

    python tools/learn/brickae_data.py vocab                       # -> analysis/brickae/vocab.json
    python tools/learn/brickae_data.py stats --n-models 200        # extents, off-axis, occlusion
    python tools/learn/brickae_data.py roundtrip --n-models 100    # gate 1: exact bricks back
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1] / "tools" / "set_study"))

DATA = Path(os.environ.get("LEGO_DATA", "/Volumes/PRO-G40/lego_data"))
OMR = DATA / "omr" / "files"
PARTSDB = Path(os.environ.get("PARTSDB", DATA / "parts.sqlite"))
OUT = DATA / "analysis" / "brickae"
WIN = (48, 32, 32)                                                         # (H plates, W studs, D studs); fits ~92 % of BFS-50 windows
OFF_MAX = 15                                                               # bricks longer than 16 cells on an axis are dropped
TARGET_N = 50
STUD, PLATE = 20.0, 8.0
ROT_TOL = 0.05                                                             # max |R - snapped| still "on axis"
SCALE_TOL = 0.05                                                           # column-norm deviation we orthonormalise away

# ---- 48 axis-aligned rotations: 24 proper (det +1) first, then 24 mirrored ---------------------------
ROT48: list[np.ndarray] = []
for perm in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
    for sx in (1, -1):
        for sy in (1, -1):
            for sz in (1, -1):
                S = np.zeros((3, 3), int)
                for j, (i, s) in enumerate(zip(perm, (sx, sy, sz))):
                    S[i, j] = s
                ROT48.append(S)
ROT48.sort(key=lambda S: -round(np.linalg.det(S)))
assert len(ROT48) == 48 and round(np.linalg.det(ROT48[23])) == 1 and round(np.linalg.det(ROT48[24])) == -1
N_ROT = 48
_ROT_IX = {tuple(S.flatten()): k for k, S in enumerate(ROT48)}


def rot_class(R: np.ndarray) -> tuple[int, float]:
    """Nearest signed permutation of a 3x3 matrix (columns normalised first).
    Returns (class 0..47 or -1 if degenerate, max abs deviation from axis alignment)."""
    R = R / np.maximum(np.linalg.norm(R, axis=0), 1e-9)
    S = np.zeros((3, 3), int)
    for j in range(3):
        i = int(np.argmax(np.abs(R[:, j]))); S[i, j] = 1 if R[i, j] >= 0 else -1
    return _ROT_IX.get(tuple(S.flatten()), -1), float(np.abs(R - S).max())


# ---- parts -----------------------------------------------------------------------------------------
_bbox_cache: dict = {}
_stems: set | None = None
_aliases: dict | None = None
_shapes: dict | None = None


def _catalog():
    global _stems, _aliases
    if _stems is None:
        from bricknet import load_catalog
        from bricknet.data import load_aliases
        _stems = set(load_catalog().stem_to_id); _aliases = load_aliases()
    return _stems, _aliases


def own_box(ldraw: str):
    """Own-frame body box in LDU with the top studs clipped (as realism._parts_with_grid)."""
    if ldraw not in _bbox_cache:
        from ldraw_read import part_bbox
        try:
            bb = part_bbox(ldraw)
        except Exception:
            bb = None
        if bb is not None:
            lo, hi = bb[0].copy(), bb[1].copy()
            if hi[1] > 0:
                lo[1] = max(lo[1], 0.0)
            bb = (lo, hi)
        _bbox_cache[ldraw] = bb
    return _bbox_cache[ldraw]


def own_dims(ldraw: str) -> tuple[int, int, int] | None:
    bb = own_box(ldraw)
    if bb is None:
        return None
    lo, hi = bb
    return (max(1, round((hi[0] - lo[0]) / STUD)), max(1, round((hi[1] - lo[1]) / PLATE)),
            max(1, round((hi[2] - lo[2]) / STUD)))                         # (w studs, h plates, d studs)


def part_shape(ldraw: str):
    """Per-cell mask of a grid part from parts.sqlite: (mask[y, x, z] bool, (x0, y0, z0) LDU box min)
    or None.  Keyed by LDraw file name through ldraw_name -> part_num."""
    global _shapes
    if _shapes is None:
        _shapes = {}
        try:
            con = sqlite3.connect(f"file:{PARTSDB}?mode=ro", uri=True)
            num_of = dict(con.execute("select ldraw, part_num from ldraw_name"))
            rows = {r[0]: r[1:] for r in con.execute("select part_num, w, d, h, x0, z0, y0, occ from part_shape")}
            con.close()
            for ld, num in num_of.items():
                if num in rows:
                    w, d, h, x0, z0, y0, occ = rows[num]
                    m = np.array(json.loads(occ), float)                   # [y][x][z], y from the bottom
                    if m.shape == (h, w, d):
                        _shapes[ld.lower()] = (m >= 0.5, (float(x0), float(y0), float(z0)))
        except Exception as e:                                             # no DB: boxes only
            print(f"[brickae] part_shape unavailable: {e}", file=sys.stderr)
    return _shapes.get(ldraw.lower())


def is_round(ldraw: str) -> bool:
    """Parts symmetric about their own axis (pins, axles, round plates/tiles, bushes): a turn about that
    axis is not a pose, so snapping it to the grid loses nothing."""
    if ldraw in _round:
        return _round[ldraw]
    from realism import _category, CATEGORIES
    return CATEGORIES[_category(ldraw)] in ("round", "tech") or ldraw.startswith(("4274", "2780", "3673", "6558", "32062", "3705", "3706", "4519", "3713", "4073", "6141", "98138", "3062", "4740", "4032", "85861", "18674"))


# ---- portable part cache (training off this Mac: no LDraw library, no parts.sqlite) -----------------
_round: dict = {}


def build_partcache(names, path: Path) -> None:
    """bbox / shape mask / roundness for these LDraw names -> json, so brickae_{data,train,eval} run with
    BRICKAE_PARTCACHE=path on a machine that has neither the LDraw library nor parts.sqlite."""
    import base64
    from ldraw_read import part_bbox
    c = {}
    for ld in sorted(set(names)):
        try: bb = part_bbox(ld)
        except Exception: bb = None
        sh = part_shape(ld)
        c[ld] = {"bbox": [bb[0].tolist(), bb[1].tolist()] if bb is not None else None, "round": bool(is_round(ld)),
                 "shape": {"dims": list(sh[0].shape), "xyz0": list(sh[1]), "bits": base64.b64encode(np.packbits(sh[0].ravel())).decode()} if sh else None}
    path.write_text(json.dumps(c)); print(f"part cache: {len(c)} parts, {sum(1 for v in c.values() if v['bbox'])} with bbox, {sum(1 for v in c.values() if v['shape'])} with shape -> {path}")


def _install_partcache(path: str) -> None:
    global _shapes
    import base64
    import ldraw_read
    c = json.loads(Path(path).read_text())
    bbox = {k: (np.array(v["bbox"][0]), np.array(v["bbox"][1])) if v["bbox"] else None for k, v in c.items()}
    ldraw_read.part_bbox = lambda name: bbox.get(name)
    _shapes = {}
    for k, v in c.items():
        _round[k] = v["round"]
        if v["shape"]:
            h, w, d = v["shape"]["dims"]
            bits = np.unpackbits(np.frombuffer(base64.b64decode(v["shape"]["bits"]), np.uint8))[: h * w * d]
            _shapes[k.lower()] = (bits.reshape(h, w, d).astype(bool), tuple(v["shape"]["xyz0"]))


if os.environ.get("BRICKAE_PARTCACHE"):
    _install_partcache(os.environ["BRICKAE_PARTCACHE"])


def load_items(path: Path) -> list:
    """(ldraw, colour, M, glo, ghi) per catalogue part, grid units, via the realism pipeline.
    Applies BrickNet aliases, orthonormalises mildly scaled transforms, drops the rest."""
    from ldraw_read import flatten, part_bbox
    from realism import _parts_with_grid
    stems, aliases = _catalog()
    parts = []
    for p in flatten(path):
        st = p.name.removesuffix(".dat"); M = p.M
        if st in aliases:
            st, wrap = aliases[st]; M = M @ wrap
        if st not in stems:
            continue
        n = np.linalg.norm(M[:3, :3], axis=0)
        if np.abs(n - 1.0).max() > SCALE_TOL:
            continue
        if np.abs(n - 1.0).max() > 1e-3:                                  # sloppy sub-model scale: re-orthonormalise
            U, _, Vt = np.linalg.svd(M[:3, :3]); M = M.copy(); M[:3, :3] = U @ Vt
        p.name = st + ".dat"; p.M = M
        parts.append(p)
    raw = {}
    for p in parts:
        if p.name not in raw:
            try: raw[p.name] = part_bbox(p.name)
            except Exception: raw[p.name] = None
    return _parts_with_grid(parts, raw)


def cells_of(glo, ghi):
    """Integer cell range [a, b) per axis for a world grid box: box edges rounded to the nearest
    cell boundary, at least one cell (same rule as bricknet_shape).  LDraw puts even-sized parts
    with edges on cell centres (a 2x4 plate at the origin spans studs -2..2), so a "cell centre
    inside the box" rule mis-sizes them; this rule is exact for whole-stud lattices in either
    phase and only approximates sub-stud SNOT offsets, which no plate x stud grid can express."""
    a = np.floor(glo + 0.5).astype(int)
    b = np.maximum(np.ceil(ghi - 0.5).astype(int), a + 1)
    return a, b


def upright(r: int) -> bool:
    """Own y axis maps to world +-y: studs up or down.  Only then do a part's plate rows land on the
    plate grid; a sideways brick (3 plates = 1.2 studs) cannot, and gets its box."""
    return abs(int(ROT48[r][1, 1])) == 1


def part_cells(ldraw: str, M: np.ndarray, r: int, glo, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(n, 3) world cells (x, y, z) a placed part covers: the part_shape mask for upright placements
    (cells mapped exactly, relative to the body box), else the whole box [a, b)."""
    box = np.mgrid[a[0]:b[0], a[1]:b[1], a[2]:b[2]].reshape(3, -1).T
    sh = part_shape(ldraw)
    if sh is None or not upright(r):
        return box
    mask, (x0, y0, z0) = sh
    iy, ix, iz = np.nonzero(mask)
    h = mask.shape[0]                                                      # rows count from the bottom; LDU +y is down
    pts = np.stack([x0 + (ix + 0.5) * STUD, y0 + (h - iy - 0.5) * PLATE, z0 + (iz + 0.5) * STUD, np.ones(len(ix))], 1)
    w = (M @ pts.T).T[:, :3]
    rel = np.stack([w[:, 0] / STUD - glo[0], -w[:, 1] / PLATE - glo[1], w[:, 2] / STUD - glo[2]], 1)  # box-relative, cell units
    idx = np.clip(np.floor(rel).astype(int), 0, (b - a) - 1) + a
    cells = np.unique(idx, axis=0)
    return cells if len(cells) else box


# ---- vocabulary ------------------------------------------------------------------------------------
class Vocab:
    """cls id <- exact ldraw (frequent parts) or ('class', w, h, d, cat) fallback; 0 = unknown."""

    def __init__(self, exact: list[str], classes: list[list[int]], colours: list[int]):
        self.exact = exact; self.classes = [tuple(c) for c in classes]; self.colours = colours
        self.ix = {ld: i + 1 for i, ld in enumerate(exact)}
        self.cix = {c: len(exact) + 1 + i for i, c in enumerate(self.classes)}
        self.col_ix = {c: i + 1 for i, c in enumerate(colours)}
        self.n = 1 + len(exact) + len(classes); self.n_col = 1 + len(colours)
        self._dims = {}; self._ncells = {}

    @staticmethod
    def load(path: Path = OUT / "vocab.json") -> "Vocab":
        d = json.loads(path.read_text()); return Vocab(d["exact"], d["classes"], d["colours"])

    def part_key(self, ldraw: str):
        from realism import _category
        dims = own_dims(ldraw)
        return None if dims is None else (dims[0], dims[1], dims[2], _category(ldraw))

    def cls(self, ldraw: str) -> int:
        if ldraw in self.ix:
            return self.ix[ldraw]
        return self.cix.get(self.part_key(ldraw), 0)

    def dims(self, c: int) -> tuple[int, int, int] | None:
        """Own-frame (w, h, d) of a class id."""
        if c not in self._dims:
            if 1 <= c <= len(self.exact):
                self._dims[c] = own_dims(self.exact[c - 1])
            elif c > len(self.exact):
                k = self.classes[c - len(self.exact) - 1]; self._dims[c] = (k[0], k[1], k[2])
            else:
                self._dims[c] = None
        return self._dims[c]

    def n_cells(self, c: int) -> int:
        """Cells a brick of this class covers (mask count for grid parts, else the box)."""
        if c not in self._ncells:
            d = self.dims(c); ld = self.ldraw(c); sh = part_shape(ld) if ld else None
            self._ncells[c] = int(sh[0].sum()) if sh is not None else (int(np.prod(d)) if d else 0)
        return self._ncells[c]

    def ldraw(self, c: int) -> str | None:
        return self.exact[c - 1] if 1 <= c <= len(self.exact) else None

    def col(self, code: int) -> int:
        return self.col_ix.get(int(code), 0)


def _count_file(path_str: str):
    try:
        items = load_items(Path(path_str))
    except Exception:
        return Counter(), Counter(), Counter()
    ld = Counter(it[0] for it in items); col = Counter(int(it[1]) for it in items)
    from realism import _category
    keys = Counter()
    for it in items:
        d = own_dims(it[0])
        if d is not None:
            keys[(d[0], d[1], d[2], _category(it[0]))] += 1
    return ld, col, keys


def omr_files() -> list[Path]:
    return sorted(OMR.glob("*.mpd")) + sorted(OMR.glob("*.ldr"))


def build_vocab(n_exact: int, n_colours: int, workers: int) -> None:
    files = omr_files()
    ld, col, keys = Counter(), Counter(), Counter()
    with ProcessPoolExecutor(workers) as ex:
        for a, b, c in ex.map(_count_file, [str(f) for f in files], chunksize=8):
            ld += a; col += b; keys += c
    tot = sum(ld.values()); run = 0
    print(f"{len(files)} models, {tot} placements, {len(ld)} distinct parts, {len(keys)} (w,h,d,cat) classes")
    for k, (_, n) in enumerate(ld.most_common(4096), 1):
        run += n
        if k in (128, 256, 512, 1024, 2048, 4096):
            print(f"  top {k:5d} exact parts cover {run / tot:.1%} of placements")
    exact = [l for l, _ in ld.most_common(n_exact)]
    classes = [list(k) for k, n in keys.most_common() if n >= 20]
    colours = [c for c, _ in col.most_common(n_colours)]
    ctot = sum(col.values()); print(f"  top {n_colours} colours cover {sum(col[c] for c in colours) / ctot:.1%}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "vocab.json").write_text(json.dumps({"exact": exact, "classes": classes, "colours": colours,
                                               "n_models": len(files), "placements": tot}))
    print(f"vocab: {len(exact)} exact + {len(classes)} classes + {len(colours)} colours -> {OUT / 'vocab.json'}")


# ---- windows ---------------------------------------------------------------------------------------
def windows_of(items: list, n_windows: int, rng: random.Random) -> list[list[int]]:
    from realism import _neighbour_pairs
    from subgraph_extract import extract_subgraph
    if len(items) < TARGET_N + 5:
        return []
    adj: dict = {}
    for a, b, _ in _neighbour_pairs(items):
        adj.setdefault(a, []).append(b); adj.setdefault(b, []).append(a)
    out = []
    for _ in range(n_windows):
        sub = extract_subgraph(items, adj, TARGET_N, rng)
        if sub is not None and sub[1]:
            out.append(sub[0])
    return out


def rasterize(items: list, nodes: list[int], vocab: Vocab, win=WIN, origin=None) -> dict:
    """Label grids for one window.  Bricks are drawn largest first so smaller (more specific) parts
    win overlapping cells.  `origin` (x, y, z cell of the window corner) fixes the frame so a
    corrupted copy aligns with its clean original; default centres the window on the bricks.
    Returns grids + the brick list + diagnostics."""
    H, W, D = win
    lo = np.min([cells_of(items[i][3], items[i][4])[0] for i in nodes], 0)
    hi = np.max([cells_of(items[i][3], items[i][4])[1] for i in nodes], 0)
    ext = hi - lo                                                          # (x, y, z) cells
    org = np.array(origin, int) if origin is not None else lo - (np.array([W, H, D]) - ext) // 2
    occ = np.zeros(win, bool); off = np.zeros((3,) + win, np.int16); rot = np.zeros(win, np.int16)
    cls = np.zeros(win, np.int32); col = np.zeros(win, np.int16); bid = np.full(win, -1, np.int32)
    bricks, diag = [], Counter()
    order = sorted(nodes, key=lambda i: -np.prod(np.subtract(*cells_of(items[i][3], items[i][4])[::-1])))
    for i in order:
        ld, colour, M, glo, ghi = items[i]
        a, b = cells_of(glo, ghi); a, b = a - org, b - org                 # window-relative (x, y, z)
        r, err = rot_class(M[:3, :3])
        c = vocab.cls(ld)
        posed = err > ROT_TOL and not is_round(ld)                         # a real off-grid pose
        brick = {"node": i, "ldraw": ld, "colour": int(colour), "cls": c, "rot": r, "anchor": a.tolist(),
                 "size": (b - a).tolist(), "offaxis": posed, "kept": 0}
        if posed: diag["posed"] += 1; brick["drop"] = "posed"
        if r < 0: diag["degenerate"] += 1; brick["drop"] = "degenerate"
        if c == 0: diag["unknown_part"] += 1
        if (b - a).max() > OFF_MAX + 1: diag["too_long"] += 1; brick["drop"] = "too_long"
        if (a < 0).any() or (b > np.array([W, H, D])).any(): diag["outside"] += 1; brick["drop"] = "outside"
        bricks.append(brick)
        if "drop" in brick:
            continue
        if err > 1e-3:                                                     # snapped round part: use the snapped frame
            Ms = M.copy(); Ms[:3, :3] = ROT48[r]; M = Ms
        cells = part_cells(ld, M, r, glo, a, b)
        xx, yy, zz = cells[:, 0], cells[:, 1], cells[:, 2]
        # a part sitting inside another's solid cells (glass in a frame, tyre on a wheel, plate in a
        # headlight brick) is a companion: the host keeps the cells, the companion is implied by it
        host = bid[yy, xx, zz]; host = host[host >= 0]
        if len(host) > 0.5 * len(cells):
            h = int(Counter(host.tolist()).most_common(1)[0][0])
            brick["drop"] = "companion"; brick["host"] = h; bricks[h].setdefault("companions", []).append(len(bricks) - 1)
            diag["companion"] += 1
            continue
        occ[yy, xx, zz] = True; rot[yy, xx, zz] = r; cls[yy, xx, zz] = c; col[yy, xx, zz] = vocab.col(colour)
        off[0, yy, xx, zz] = xx - a[0]; off[1, yy, xx, zz] = yy - a[1]; off[2, yy, xx, zz] = zz - a[2]
        bid[yy, xx, zz] = len(bricks) - 1
        brick["n_cells"] = len(cells)
    for k, br in enumerate(bricks):
        br["kept"] = int((bid == k).sum())
        if "drop" not in br and br["kept"] < 0.5 * br.get("n_cells", 1):
            diag["occluded"] += 1
    return {"occ": occ, "off": off, "rot": rot, "cls": cls, "col": col, "bid": bid, "bricks": bricks,
            "origin": org.tolist(), "extent": ext.tolist(), "diag": diag}


def degroup(occ, off, rot, cls, col, vocab: Vocab, min_frac: float = 0.5) -> list[dict]:
    """Label grids -> bricks: group cells by (anchor, cls, rot), validate cell count, colour by vote."""
    ys, xs, zs = np.nonzero(occ)
    groups: dict = {}
    for y, x, z in zip(ys, xs, zs):
        key = (int(x - off[0, y, x, z]), int(y - off[1, y, x, z]), int(z - off[2, y, x, z]),
               int(cls[y, x, z]), int(rot[y, x, z]))
        groups.setdefault(key, []).append(int(col[y, x, z]))
    out = []
    for (ax, ay, az, c, r), cols in groups.items():
        d = vocab.dims(c)
        if d is None or not (0 <= r < N_ROT):
            continue
        S = np.abs(ROT48[r]); wd = S @ np.array([d[0], d[1], d[2]])         # world extents (x, y, z) in cells
        expect = vocab.n_cells(c) if upright(r) else int(np.prod(wd))
        if len(cols) < min_frac * expect:
            continue
        out.append({"anchor": [ax, ay, az], "cls": c, "rot": r, "col": Counter(cols).most_common(1)[0][0],
                    "size": wd.tolist(), "n_cells": len(cols)})
    return out


def brick_to_ldr(br: dict, vocab: Vocab, origin, ldraw: str | None = None, colour: int | None = None) -> str | None:
    """One LDraw type-1 line for a decoded brick: orientation from rot, translation from the anchor cell."""
    ld = ldraw or vocab.ldraw(br["cls"])
    if ld is None:
        return None
    bb = own_box(ld)
    if bb is None:
        return None
    S = ROT48[br["rot"]].astype(float)
    a = np.array(br["anchor"]) + np.array(origin); b = a + np.array(br["size"])
    wlo = np.array([STUD * a[0], -PLATE * b[1], STUD * a[2]])             # LDU world box min (y is -up); cell k = [20k, 20k+20)
    corners = np.array([[x, y, z] for x in (bb[0][0], bb[1][0]) for y in (bb[0][1], bb[1][1]) for z in (bb[0][2], bb[1][2])])
    t = wlo - (S @ corners.T).T.min(0)
    c = br["col"] if colour is None else colour
    return "1 %d %.3f %.3f %.3f %s %s" % (c, t[0], t[1], t[2], " ".join("%g" % v for v in S.flatten()), ld)


# ---- dataset ---------------------------------------------------------------------------------------
def _build_model(args):
    """Clean windows of one model as re-rasterisable records: the window's items + origin."""
    path_str, per_model, seed, max_miss = args
    vocab = Vocab.load(); rng = random.Random(seed)
    try:
        items = load_items(Path(path_str))
    except Exception:
        return []
    out = []
    for nodes in windows_of(items, per_model, rng):
        g = rasterize(items, nodes, vocab); bs = g["bricks"]
        if any(b.get("drop") in ("posed", "degenerate", "outside", "too_long") for b in bs):
            continue
        dec = degroup(g["occ"], g["off"], g["rot"], g["cls"], g["col"], vocab)
        truth = {(tuple(b["anchor"]), b["cls"], b["rot"], vocab.col(b["colour"])) for b in bs if "drop" not in b}
        got = {(tuple(b["anchor"]), b["cls"], b["rot"], b["col"]) for b in dec}
        if len(truth) - len(got & truth) > max_miss or len(truth) < 10:
            continue
        keep = [b["node"] for b in bs if "drop" not in b]                  # companions are implied, not labels
        out.append({"file": Path(path_str).name, "origin": g["origin"], "n": len(keep),
                    "ldraw": [items[i][0] for i in keep], "colour": [int(items[i][1]) for i in keep],
                    "M": np.stack([items[i][2] for i in keep]).astype(np.float32)})
    return out


def build_dataset(per_model: int, workers: int, max_miss: int, seed: int, kinds: str | None) -> None:
    """analysis/brickae/windows.npz: all clean OMR windows (items + origin; rasterise at train time)."""
    files = omr_files()
    if kinds:
        import csv
        kind = {r["id"]: r["kind"] for r in csv.DictReader(open(DATA / "analysis" / "set_kinds.csv"))}
        want = set(kinds.split(",")); files = [f for f in files if kind.get(f.name.split(".")[0].split("_")[0]) in want]
    recs = []
    with ProcessPoolExecutor(workers) as ex:
        for i, r in enumerate(ex.map(_build_model, [(str(f), per_model, seed + i, max_miss) for i, f in enumerate(files)], chunksize=4)):
            recs += r
            if i % 200 == 199:
                print(f"  {i + 1}/{len(files)} models, {len(recs)} windows", flush=True)
    n = np.array([r["n"] for r in recs]); off = np.concatenate([[0], np.cumsum(n)])
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "windows.npz", src=np.array([r["file"] for r in recs]), origin=np.array([r["origin"] for r in recs], np.int32),
                        n=n.astype(np.int32), offset=off.astype(np.int64), ldraw=np.array(sum((r["ldraw"] for r in recs), [])),
                        colour=np.array(sum((r["colour"] for r in recs), []), np.int32), M=np.concatenate([r["M"] for r in recs]))
    print(f"{len(recs)} windows from {len(set(r['file'] for r in recs))} models, {n.sum()} bricks (mean {n.mean():.1f}) -> {OUT / 'windows.npz'}")


def load_windows(path: Path = OUT / "windows.npz"):
    """Yields (items, origin, file) per stored window; items as in load_items (glo/ghi recomputed)."""
    z = np.load(path, allow_pickle=False)
    # read each array once: indexing z[...] inside the loop re-decompresses the whole array
    n, offset, ldraw, colour, M, origin, src = (z["n"], z["offset"], z["ldraw"].tolist(), z["colour"].tolist(),
                                                z["M"].astype(np.float64), z["origin"], z["src"].tolist())
    from realism import _parts_with_grid
    from ldraw_read import part_bbox, Part
    bb = {}
    for w in range(len(n)):
        s, e = int(offset[w]), int(offset[w + 1])
        parts = [Part(ldraw[i], int(colour[i]), M[i], 0, ()) for i in range(s, e)]
        for p in parts:
            if p.name not in bb:
                try: bb[p.name] = part_bbox(p.name)
                except Exception: bb[p.name] = None
        yield _parts_with_grid(parts, bb), origin[w], src[w]


# ---- CLI -------------------------------------------------------------------------------------------
def _model_windows(args):
    path_str, per_model, seed, roundtrip = args
    vocab = Vocab.load(); rng = random.Random(seed)
    try:
        items = load_items(Path(path_str))
    except Exception:
        return []
    res = []
    for nodes in windows_of(items, per_model, rng):
        g = rasterize(items, nodes, vocab)
        row = {"extent": g["extent"], "diag": dict(g["diag"]), "n": len(nodes),
               "n_posed": sum(1 for b in g["bricks"] if b["offaxis"])}
        if roundtrip:
            dec = degroup(g["occ"], g["off"], g["rot"], g["cls"], g["col"], vocab)
            truth = {(tuple(b["anchor"]), b["cls"], b["rot"], vocab.col(b["colour"])): b for b in g["bricks"] if "drop" not in b}
            got = {(tuple(b["anchor"]), b["cls"], b["rot"], b["col"]) for b in dec}
            hit = len(got & set(truth))
            exact_ld = sum(1 for k in got & set(truth) if vocab.ldraw(truth[k]["cls"]) == truth[k]["ldraw"])
            row.update(n_truth=len(truth), n_dec=len(dec), hit=hit, exact_ld=exact_ld)
        res.append(row)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["vocab", "stats", "roundtrip", "build"])
    ap.add_argument("--max-miss", type=int, default=1, help="build: max bricks a window may fail to round-trip")
    ap.add_argument("--n-models", type=int, default=200)
    ap.add_argument("--per-model", type=int, default=10)
    ap.add_argument("--n-exact", type=int, default=2048)
    ap.add_argument("--n-colours", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kinds", default=None, help="comma list of set kinds (analysis/set_kinds.csv), e.g. building,scene")
    a = ap.parse_args()
    if a.cmd == "vocab":
        build_vocab(a.n_exact, a.n_colours, a.workers); return
    if a.cmd == "build":
        build_dataset(a.per_model, a.workers, a.max_miss, a.seed, a.kinds); return
    files = omr_files()
    if a.kinds:
        import csv
        kind = {r["id"]: r["kind"] for r in csv.DictReader(open(DATA / "analysis" / "set_kinds.csv"))}
        want = set(a.kinds.split(","))
        files = [f for f in files if kind.get(f.name.split(".")[0].split("_")[0]) in want]
    rng = random.Random(a.seed); files = rng.sample(files, min(a.n_models, len(files)))
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(_model_windows, [(str(f), a.per_model, a.seed + i, a.cmd == "roundtrip") for i, f in enumerate(files)]):
            rows += r
    n = len(rows); print(f"{n} windows from {len(files)} models")
    ext = np.array([r["extent"] for r in rows])                            # (x, y, z)
    for name, p in (("x studs", 0), ("y plates", 1), ("z studs", 2)):
        print(f"  extent {name:8s}: median {np.median(ext[:, p]):.0f}  p90 {np.percentile(ext[:, p], 90):.0f}  max {ext[:, p].max()}")
    for win in ((32, 24, 24), (48, 32, 32)):
        fit = np.mean((ext[:, 1] <= win[0]) & (ext[:, 0] <= win[1]) & (ext[:, 2] <= win[2]))
        print(f"  fits {win}: {fit:.1%}")
    tot_br = sum(r["n"] for r in rows); diag = Counter()
    for r in rows: diag.update(r["diag"])
    print("  per-brick flags: " + ", ".join(f"{k} {v / tot_br:.2%}" for k, v in diag.most_common()))
    posed = np.array([r["n_posed"] for r in rows])
    print(f"  windows with 0 posed bricks: {np.mean(posed == 0):.1%}   <=1: {np.mean(posed <= 1):.1%}   <=5: {np.mean(posed <= 5):.1%}")
    if a.cmd == "roundtrip":
        T = sum(r["n_truth"] for r in rows); Hh = sum(r["hit"] for r in rows); Dd = sum(r["n_dec"] for r in rows)
        E = sum(r["exact_ld"] for r in rows)
        print(f"\nROUND TRIP: eligible bricks {T}/{tot_br} ({T / tot_br:.1%} of all; rest: posed / companion / too long / outside)")
        print(f"  decoded {Dd}, matched (anchor, cls, rot, colour) {Hh}/{T} = {Hh / T:.1%}   spurious {Dd - Hh}")
        print(f"  of matched, exact LDraw part recoverable: {E}/{Hh} = {E / max(Hh, 1):.1%}")
        clean = [r for r in rows if r["n_posed"] == 0]
        if clean:
            Tc = sum(r["n_truth"] for r in clean); Hc = sum(r["hit"] for r in clean)
            print(f"  windows without posed bricks ({len(clean)}): matched {Hc}/{Tc} = {Hc / Tc:.1%}, fully exact {np.mean([r['hit'] == r['n_truth'] for r in clean]):.1%}")


if __name__ == "__main__":
    main()
