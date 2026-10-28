"""The statistics sampler's inner loop in Rust (native/sampler), called
through its C interface.

sampler.py stays the reference: this module only lays its data out as
arrays — the target's cells, every (part, yaw) option with its cell masks
and connections, the statistics as dense token tables — and hands them to
`e2l_sample`, which runs the same fill, warm-up and annealing walk.

Build the library once, for the architecture of the Python that loads it
(an x86_64 Python under Rosetta cannot load an arm64 library):

    python -m everything2lego.native build

It is looked for at $E2L_NATIVE_LIB, then under native/sampler/target.
"""
from __future__ import annotations

import ctypes as C
import math
import os
import platform
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

ABI = 16
INFO_LEN = 20
PATHS4_BUCKETS = 8

_P = C.POINTER


class Problem(C.Structure):
    _fields_ = [
        ("h", C.c_int32), ("x", C.c_int32), ("z", C.c_int32),
        ("tbits", _P(C.c_uint64)), ("want", _P(C.c_uint8)), ("visible", _P(C.c_uint8)),
        ("tcolour", _P(C.c_int32)), ("tlab", _P(C.c_float)),
        ("n_col", C.c_int32), ("pal_lab", _P(C.c_float)),
        ("n_tok", C.c_int32), ("nll", _P(C.c_double)), ("logp", _P(C.c_double)),
        ("n_part", C.c_int32), ("snap", _P(C.c_int32)),
        ("n_opt", C.c_int32),
        ("opt_part", _P(C.c_int32)), ("opt_token", _P(C.c_int32)),
        ("opt_size", _P(C.c_int32)), ("opt_first", _P(C.c_int32)),
        ("opt_logprior", _P(C.c_double)), ("opt_box", _P(C.c_int32)),
        ("cell_start", _P(C.c_int32)), ("cell_yxz", _P(C.c_int32)),
        ("cell_bits", _P(C.c_uint64)), ("cell_o", _P(C.c_double)),
        ("top_start", _P(C.c_int32)), ("top_xz", _P(C.c_int32)),
        ("bot_start", _P(C.c_int32)), ("bot_xz", _P(C.c_int32)),
        ("side_start", _P(C.c_int32)), ("side_v", _P(C.c_int32)),
        ("halo_start", _P(C.c_int32)), ("halo_yxz", _P(C.c_int32)),
        ("n_conn", C.c_int32), ("fits", _P(C.c_uint8)),
        ("w_cov", C.c_double), ("w_col", C.c_double), ("w_loc", C.c_double),
        ("w_glob", C.c_double), ("w_sup", C.c_double), ("w_side", C.c_double),
        ("n_ref", C.c_double), ("w_dens", C.c_double), ("dens_sd", C.c_double),
        ("p_ref", C.c_double), ("w_cnt", C.c_double), ("cnt_sd", C.c_double),
        ("w_sil", C.c_double), ("style", C.c_double), ("w_glob2", C.c_double), ("logp2", _P(C.c_double)),
        ("middle_w", _P(C.c_double)),
        ("w_glob3", C.c_double), ("logp3", _P(C.c_double)), ("mid_bucket", _P(C.c_int32)),
        ("max_deg_paths4", C.c_int32), ("paths4_sample_n", C.c_int32),
        ("w_entropy_glob3", C.c_double),
        ("w_revkl_glob3", C.c_double), ("pref3", _P(C.c_double)),
        ("pixx_start", _P(C.c_int32)), ("pixx", _P(C.c_int32)),
        ("pixy_start", _P(C.c_int32)), ("pixy", _P(C.c_int32)),
        ("pixz_start", _P(C.c_int32)), ("pixz", _P(C.c_int32)),
        ("tsx", _P(C.c_uint8)), ("tsy", _P(C.c_uint8)), ("tsz", _P(C.c_uint8)),
        # the proposal network (proposals.py); use_net 0: the pair statistics propose
        ("use_net", C.c_int32), ("net_in", C.c_int32), ("net_h", C.c_int32), ("net_c", C.c_int32),
        ("net_kinds", C.c_int32), ("net_objects", C.c_int32), ("net_object", C.c_int32),
        ("w1", _P(C.c_float)), ("b1", _P(C.c_float)), ("w2", _P(C.c_float)), ("b2", _P(C.c_float)),
        ("mu", _P(C.c_float)), ("sd", _P(C.c_float)),
        ("opt_out", _P(C.c_int32)), ("opt_lpart", _P(C.c_double)), ("opt_kind", _P(C.c_int32)),
        ("region_object", _P(C.c_int32)),  # H*X*Z: per-cell object index, -1 = use net_object
        ("opt_variants", _P(C.c_int32)),  # n_opt: swap-group id (-1 none)
        ("net_static", C.c_int32), ("static", _P(C.c_float)),
        ("n_hinge", C.c_int32), ("hinge_data", _P(C.c_int32)),  # n_hinge*5: (opt_a, opt_b, dy, dx, dz)
        ("n_init", C.c_int32),
        ("init_opt", _P(C.c_int32)), ("init_colour", _P(C.c_int32)),
        ("init_y", _P(C.c_int32)), ("init_x", _P(C.c_int32)), ("init_z", _P(C.c_int32)),
        ("region_label", _P(C.c_int32)),
        ("t_start_share", C.c_double), ("t_final_share", C.c_double),
        ("iters", C.c_int64), ("warmup", C.c_int64), ("seed", C.c_uint64),
        ("top_k", C.c_int32), ("log_every", C.c_int64),
        ("out_cap", C.c_int32), ("out_n", _P(C.c_int32)),
        ("out_opt", _P(C.c_int32)), ("out_colour", _P(C.c_int32)),
        ("out_y", _P(C.c_int32)), ("out_x", _P(C.c_int32)), ("out_z", _P(C.c_int32)),
        ("info", _P(C.c_double)),
        ("hist_cap", C.c_int32), ("hist_n", _P(C.c_int32)), ("hist", _P(C.c_double)),
        ("w_size", C.c_double), ("n_size_class", C.c_int32),      # part-size statistics (null = off)
        ("opt_class", _P(C.c_int32)), ("size_pref", _P(C.c_double)),
        ("n_snot_rows", C.c_int32), ("snot_rows", _P(C.c_int32)),      # sideways move (snot_options.pair_rows)
        ("w_sideways", C.c_double), ("p_sideways", C.c_double), ("opt_sideways", _P(C.c_uint8)),   # sideways-share KL term
    ]


CRATE = Path(__file__).resolve().parents[2] / "native" / "sampler"


def _triple() -> Optional[str]:
    """The Rust target of this Python's architecture (None: cargo's default)."""
    m = platform.machine().lower()
    if sys.platform == "darwin":
        return {"x86_64": "x86_64-apple-darwin", "arm64": "aarch64-apple-darwin"}.get(m)
    return None


def _lib_name() -> str:
    return {"darwin": "libe2l_sampler.dylib", "win32": "e2l_sampler.dll"}.get(sys.platform, "libe2l_sampler.so")


def _lib_path() -> Optional[Path]:
    env = os.environ.get("E2L_NATIVE_LIB")
    if env:
        return Path(env)
    t = _triple()
    for d in ([CRATE / "target" / t / "release"] if t else []) + [CRATE / "target" / "release"]:
        if (d / _lib_name()).exists():
            return d / _lib_name()
    return None


def build() -> Path:
    """cargo build the library for this Python's architecture, with the
    first cargo that can: rustup's (it holds other architectures' standard
    libraries once `rustup target add`-ed), then the one on PATH (a system
    cargo builds for its own machine)."""
    # try every cargo on the machine: each rustup install has its own targets, and
    # we need one that holds the arm64 or x86_64 std library for this Python
    found = [Path.home() / ".cargo" / "bin" / "cargo",
             Path("/opt/homebrew/Cellar/rustup/1.29.1/libexec/bin/cargo"),
             Path(shutil.which("cargo") or "/dev/null")]
    cargos = [str(c) for c in found if c.exists()]
    t = _triple()
    for cargo in dict.fromkeys(cargos):
        cmd = [cargo, "build", "--release", "--manifest-path", str(CRATE / "Cargo.toml")]
        if subprocess.run(cmd + (["--target", t] if t else [])).returncode == 0:
            break
    else:
        raise RuntimeError(f"no cargo could build for {t or 'this machine'}: tried {', '.join(cargos) or 'none'}")
    library.cache_clear()
    p = _lib_path()
    if p is None or library() is None:
        raise RuntimeError(f"built, but {p} can't be used: {why_not()}")
    return p


def why_not() -> Optional[str]:
    """Why the native library can't be used, or None when it can."""
    p = _lib_path()
    if p is None or not p.exists():
        return "not built"
    src = CRATE / "src" / "lib.rs"
    if not os.environ.get("E2L_NATIVE_LIB") and src.exists() and src.stat().st_mtime > p.stat().st_mtime:
        return f"{p.name} is older than {src.name}: rebuild it"   # it would run the old walk
    try:
        lib = C.CDLL(str(p))
    except OSError:
        return f"{p.name} does not load into this Python ({platform.machine()})"
    lib.e2l_abi.restype = C.c_int32
    if lib.e2l_abi() != ABI:
        return f"{p.name} has interface {lib.e2l_abi()}, this module {ABI}: rebuild it"
    return None


@lru_cache(maxsize=1)
def library():
    """The native library, or None when it can't be used (why_not says why)."""
    if why_not() is not None:
        return None
    lib = C.CDLL(str(_lib_path()))
    lib.e2l_abi.restype = C.c_int32
    lib.e2l_sample.argtypes = [_P(Problem)]
    lib.e2l_sample.restype = C.c_int32
    return lib


def available() -> bool:
    return library() is not None


def _ptr(a: np.ndarray, ctype):
    return a.ctypes.data_as(_P(ctype))


def snap_table(vocab, n_col: int) -> np.ndarray:
    """[part, palette colour] -> the colour it becomes on that part: itself
    when the part is made in it, else the perceptually nearest it is made
    in (colors.nearest_where)."""
    from .colors import _palette_lab, palette
    lab = _palette_lab()
    codes = np.array([c.code for c in palette()])
    dist = ((lab[:, None, :] - lab[None, :, :]) ** 2).sum(-1)
    out = np.tile(np.arange(n_col, dtype=np.int32), (len(vocab), 1))
    for p, v in enumerate(vocab):
        ok = np.isin(codes, list(v.colours))
        if not ok.any():
            continue
        d = np.where(ok[None, :], dist, np.inf)
        near = d.argmin(axis=1)
        out[p] = np.where(ok, np.arange(n_col), near).astype(np.int32)
    return out


def run(target, stats, vocab, kind, weights, *, iters: int, warmup: int, seed: int,
        t_start_share: float, t_final_share: float, top_k: int, log_every: int, opts_logprior,
        net: bool = True, tile_boost: float = 0.0, snot_boost: float = 0.0,
        init_placements=None):
    """Run the native sampler.  Returns ([(vpart, orient, colour, y, x, z)], info, history)."""
    lib = library()
    if lib is None:
        raise RuntimeError("native sampler not built: python -m everything2lego.native build")
    from .colors import _palette_lab
    from .sampler import CONNECTORS, FITS
    H, X, Z = target.want.shape
    keep = []                                            # arrays the struct points into

    def arr(a, dtype):
        a = np.ascontiguousarray(a, dtype=dtype)
        keep.append(a)
        return a

    lab = _palette_lab().astype(np.float32)
    n_col = len(lab)
    tcol = np.where(target.colour >= 0, target.colour, -1).astype(np.int32)
    tlab = lab[np.maximum(tcol, 0)]
    # statistics as dense tables over the vocabulary's tokens
    toks = sorted({v.token for v in vocab})
    tid = {t: i for i, t in enumerate(toks)}
    n_tok = len(toks)
    dirs = ("up", "down", "level")
    nll = np.empty((n_tok, n_tok, 3))
    logp = np.empty((n_tok, n_tok, 3))
    logp2 = np.empty((n_tok, n_tok, 3))
    for a, ta in enumerate(toks):
        for b, tb in enumerate(toks):
            for k, d in enumerate(dirs):
                nll[a, b, k] = stats.nll(ta, tb, d)
                logp[a, b, k] = math.log(stats.ref.p((ta, tb, d)))
                logp2[a, b, k] = math.log(stats.ref2.p((ta, tb, d)))
    # paths4 term: only wired when stats.ref3 is present AND weights.glob3 > 0.
    # Vectorized logp3 build: scatter counts into dense (a, b, dir_bucket) tensors,
    # then apply the smoothing formula with numpy.  Avoids 6M Python Counter lookups.
    use_paths4 = getattr(stats, "ref3", None) is not None and getattr(weights, "glob3", 0.0) > 0.0
    if use_paths4:
        ref3 = stats.ref3
        _total = max(1.0, ref3._total); _kind_total = max(1.0, ref3._kind_total); _vocab = ref3._vocab
        l1, l2, l3 = ref3.lam
        from .style import kind_of
        tok_kind_idx: dict[str, int] = {}                              # unique kinds -> small index
        kind_arr = np.empty(n_tok, np.int32)
        for i, t in enumerate(toks):
            k = kind_of(t)
            if k not in tok_kind_idx:
                tok_kind_idx[k] = len(tok_kind_idx)
            kind_arr[i] = tok_kind_idx[k]
        n_kind = len(tok_kind_idx)
        n_dkey = 3 * PATHS4_BUCKETS
        exact = np.zeros((n_tok, n_tok, n_dkey))
        kind_count = np.zeros((n_kind, n_kind, n_dkey))
        kind_spread_set: dict[tuple, set] = {}
        dir_idx = {d: i for i, d in enumerate(dirs)}
        for (a, b, dbk), n in ref3.counts.items():
            ai = tid.get(a); bi = tid.get(b)
            if ai is None or bi is None:
                continue
            if ":" not in dbk:
                continue
            d_str, bucket_str = dbk.rsplit(":", 1)
            di = dir_idx.get(d_str)
            if di is None:
                continue
            try:
                bucket = int(bucket_str)
            except ValueError:
                continue
            if not (0 <= bucket < PATHS4_BUCKETS):
                continue
            dkey = di * PATHS4_BUCKETS + bucket
            exact[ai, bi, dkey] += n
            # kind-level aggregation (shared across all exact keys with same kinds)
            kai, kbi = kind_arr[ai], kind_arr[bi]
            kind_count[kai, kbi, dkey] += n
            kind_spread_set.setdefault((kai, kbi, dkey), set()).add((ai, bi))
        kind_spread = np.ones((n_kind, n_kind, n_dkey), np.float64)      # +1 vs _kinds_of
        for (ka, kb, dk), s in kind_spread_set.items():
            kind_spread[ka, kb, dk] += len(s)
        # Build logp3: l1 * exact/total + l2 * kind/kind_total/spread + l3/vocab
        exact_p = exact / _total
        # broadcast kind back to (n_tok, n_tok, n_dkey) via indexing
        kind_p_tok = kind_count[kind_arr[:, None], kind_arr[None, :]] / _kind_total / kind_spread[kind_arr[:, None], kind_arr[None, :]]
        p3_raw = l1 * exact_p + l2 * kind_p_tok + l3 / _vocab           # p_ref(key) for each slot
        logp3 = np.log(p3_raw).reshape(n_tok, n_tok, 3, PATHS4_BUCKETS)
        pref3 = p3_raw.reshape(n_tok, n_tok, 3, PATHS4_BUCKETS)         # parallel array; coverage term uses it
        # mid_bucket[a, b] = Stats.middle_bucket(tok_a, tok_b)
        bucket_of = stats.middle_bucket
        mid_bucket = np.array([[bucket_of(ta, tb) for tb in toks] for ta in toks], np.int32)
    else:
        logp3 = None
        mid_bucket = None
        pref3 = None
    from .sampler import unpack
    fine = unpack(target.bits)
    tsil = (fine.any(axis=1), fine.any(axis=0), fine.any(axis=2))
    # options
    pid = {id(v): i for i, v in enumerate(vocab)}
    opts = [(v, o) for v in vocab for o in v.orients]
    size_fields = {}                                                             # part-size mix term (sizes.py); off when weight 0
    if getattr(weights, "size", 0.0) > 0:
        from .sizes import N_CLASS, real_mix, size_class
        oc = arr([size_class(o.size[0], o.size[1], o.size[2]) for _, o in opts], np.int32); sp = arr(real_mix(kind), np.float64)
        size_fields = dict(w_size=weights.size, n_size_class=N_CLASS, opt_class=_ptr(oc, C.c_int32), size_pref=_ptr(sp, C.c_double))
    if any(o.snot is not None for _, o in opts):                                 # sideways options present: the combined move
        from .snot_options import pair_rows
        sr = arr(pair_rows(opts), np.int32)
        size_fields.update(n_snot_rows=len(sr), snot_rows=_ptr(sr, C.c_int32))
        if getattr(weights, "sideways", 0.0) > 0:
            sf = arr([o.snot is not None for _, o in opts], np.uint8)
            size_fields.update(w_sideways=weights.sideways, p_sideways=getattr(weights, "p_sideways", 0.10), opt_sideways=_ptr(sf, C.c_uint8))
    boxk = {"brick": 0, "plate": 1, "tile": 2}
    cs, ts, bs, ss, hs = [0], [0], [0], [0], [0]
    halo = []
    pstart = ([0], [0], [0])
    pvals = ([], [], [])
    cyxz, cbits, co, txz, bxz, sv = [], [], [], [], [], []
    for v, o in opts:
        cyxz.append(o.cells)
        cbits.append(o.bits)
        co.append(o.o)
        cs.append(cs[-1] + len(o.cells))
        txz.extend(sorted(o.top))
        ts.append(len(txz))
        bxz.extend(sorted(o.bottom))
        bs.append(len(bxz))
        sv.extend(o.sides)
        ss.append(len(sv))
        halo.append(o.halo)
        hs.append(hs[-1] + len(o.halo))
        for a in range(3):
            pvals[a].append(o.pix[a])
            pstart[a].append(pstart[a][-1] + len(o.pix[a]))
    snap = snap_table(vocab, n_col)
    cap = int(target.want.sum()) + 16
    out = {k: np.zeros(cap, np.int32) for k in ("opt", "colour", "y", "x", "z")}
    out_n = np.zeros(1, np.int32)
    info = np.zeros(INFO_LEN)
    hist_cap = max(2, iters // max(1, log_every) + 2)
    hist = np.zeros(hist_cap)
    hist_n = np.zeros(1, np.int32)
    keep += list(out.values()) + [out_n, info, hist, hist_n]
    p = Problem(
        h=H, x=X, z=Z,
        tbits=_ptr(arr(target.bits, np.uint64), C.c_uint64),
        want=_ptr(arr(target.want, np.uint8), C.c_uint8),
        visible=_ptr(arr(target.visible, np.uint8), C.c_uint8),
        tcolour=_ptr(arr(tcol, np.int32), C.c_int32),
        tlab=_ptr(arr(tlab, np.float32), C.c_float),
        n_col=n_col, pal_lab=_ptr(arr(lab, np.float32), C.c_float),
        n_tok=n_tok, nll=_ptr(arr(nll, np.float64), C.c_double), logp=_ptr(arr(logp, np.float64), C.c_double),
        n_part=len(vocab), snap=_ptr(arr(snap, np.int32), C.c_int32),
        n_opt=len(opts),
        opt_part=_ptr(arr([pid[id(v)] for v, _ in opts], np.int32), C.c_int32),
        opt_token=_ptr(arr([tid[v.token] for v, _ in opts], np.int32), C.c_int32),
        opt_size=_ptr(arr([o.size for _, o in opts], np.int32), C.c_int32),
        opt_first=_ptr(arr([o.first for _, o in opts], np.int32), C.c_int32),
        opt_logprior=_ptr(arr(opts_logprior, np.float64), C.c_double),
        opt_box=_ptr(arr([-1 if o.snot is not None else boxk.get(v.box, -1) for v, o in opts], np.int32), C.c_int32),   # sideways: no merge / retile
        cell_start=_ptr(arr(cs, np.int32), C.c_int32),
        cell_yxz=_ptr(arr(np.concatenate(cyxz), np.int32), C.c_int32),
        cell_bits=_ptr(arr(np.concatenate(cbits), np.uint64), C.c_uint64),
        cell_o=_ptr(arr(np.concatenate(co), np.float64), C.c_double),
        top_start=_ptr(arr(ts, np.int32), C.c_int32),
        top_xz=_ptr(arr(np.array(txz, np.int32).reshape(-1, 2), np.int32), C.c_int32),
        bot_start=_ptr(arr(bs, np.int32), C.c_int32),
        bot_xz=_ptr(arr(np.array(bxz, np.int32).reshape(-1, 2), np.int32), C.c_int32),
        side_start=_ptr(arr(ss, np.int32), C.c_int32),
        side_v=_ptr(arr(np.array(sv, np.int32).reshape(-1, 6), np.int32), C.c_int32),
        n_conn=len(CONNECTORS), fits=_ptr(arr(FITS, np.uint8), C.c_uint8),
        halo_start=_ptr(arr(hs, np.int32), C.c_int32),
        halo_yxz=_ptr(arr(np.concatenate(halo), np.int32), C.c_int32),
        w_cov=weights.cov, w_col=weights.col, w_loc=weights.loc, w_glob=weights.glob,
        w_sup=weights.sup, w_side=weights.side,
        n_ref=stats.density * float(target.fill.sum()), w_dens=weights.dens, dens_sd=stats.density_sd,
        p_ref=stats.part_density * float(target.fill.sum()), w_cnt=weights.cnt, cnt_sd=stats.part_density_sd,
        w_sil=weights.sil, style=weights.style, w_glob2=weights.glob2,
        logp2=_ptr(arr(logp2, np.float64), C.c_double),
        middle_w=_ptr(arr([stats.middle_w(t) for t in toks], np.float64), C.c_double),
        w_glob3=(weights.glob3 if use_paths4 else 0.0),
        logp3=(_ptr(arr(logp3, np.float64), C.c_double) if use_paths4 else _P(C.c_double)()),
        mid_bucket=(_ptr(arr(mid_bucket, np.int32), C.c_int32) if use_paths4 else _P(C.c_int32)()),
        max_deg_paths4=int(getattr(weights, "max_deg_paths4", 0)),
        paths4_sample_n=int(getattr(weights, "paths4_sample_n", 1)),
        w_entropy_glob3=float(getattr(weights, "entropy_glob3", 0.0) if use_paths4 else 0.0),
        w_revkl_glob3=float(getattr(weights, "revkl_glob3", 0.0) if use_paths4 else 0.0),
        pref3=(_ptr(arr(pref3, np.float64), C.c_double) if use_paths4 else _P(C.c_double)()),
        pixx_start=_ptr(arr(pstart[0], np.int32), C.c_int32), pixx=_ptr(arr(np.concatenate(pvals[0]), np.int32), C.c_int32),
        pixy_start=_ptr(arr(pstart[1], np.int32), C.c_int32), pixy=_ptr(arr(np.concatenate(pvals[1]), np.int32), C.c_int32),
        pixz_start=_ptr(arr(pstart[2], np.int32), C.c_int32), pixz=_ptr(arr(np.concatenate(pvals[2]), np.int32), C.c_int32),
        tsx=_ptr(arr(tsil[0], np.uint8), C.c_uint8), tsy=_ptr(arr(tsil[1], np.uint8), C.c_uint8),
        tsz=_ptr(arr(tsil[2], np.uint8), C.c_uint8),
        t_start_share=t_start_share, t_final_share=t_final_share,
        iters=iters, warmup=warmup, seed=seed, top_k=top_k, log_every=log_every,
        out_cap=cap, out_n=_ptr(out_n, C.c_int32),
        out_opt=_ptr(out["opt"], C.c_int32), out_colour=_ptr(out["colour"], C.c_int32),
        out_y=_ptr(out["y"], C.c_int32), out_x=_ptr(out["x"], C.c_int32), out_z=_ptr(out["z"], C.c_int32),
        info=_ptr(info, C.c_double),
        hist_cap=hist_cap, hist_n=_ptr(hist_n, C.c_int32), hist=_ptr(hist, C.c_double),
        **size_fields,
    )
    from . import proposals
    from .sampler import net_tables
    from .style import kind_of
    nn = proposals.load() if net else None
    if nn is not None:
        out_t, lpart, obj = net_tables(nn, vocab, kind, tile_boost=tile_boost, snot_boost=snot_boost)
        okind = [nn.kind_index.get(kind_of(v.token), len(nn.kinds)) for v, _ in opts]
        p.use_net, p.net_in, p.net_h, p.net_c = 1, nn.W1.shape[0], nn.W1.shape[1], nn.W2.shape[1]
        p.net_kinds, p.net_objects, p.net_object = len(nn.kinds), len(nn.objects), obj
        p.net_static = proposals.NSTATIC
        for name, a, ct, dt in (("w1", nn.W1, C.c_float, np.float32), ("b1", nn.b1, C.c_float, np.float32),
                                ("w2", nn.W2, C.c_float, np.float32), ("b2", nn.b2, C.c_float, np.float32),
                                ("mu", nn.mu, C.c_float, np.float32), ("sd", nn.sd, C.c_float, np.float32),
                                ("opt_out", out_t, C.c_int32, np.int32), ("opt_lpart", lpart, C.c_double, np.float64),
                                ("opt_kind", okind, C.c_int32, np.int32),
                                ("static", proposals.static_context(target.fill), C.c_float, np.float32)):
            setattr(p, name, _ptr(arr(a, dt), ct))
        # per-cell region kinds, if any
        if target.region_kind is not None:
            region = np.array([[[nn.object_index.get(str(rk) if rk else "", -1) for rk in row]
                                for row in layer] for layer in target.region_kind], dtype=np.int32)
        else:
            region = np.full(target.want.shape, -1, dtype=np.int32)
        p.region_object = _ptr(arr(region, np.int32), C.c_int32)
    if target.region_label is not None:
        p.region_label = _ptr(arr(target.region_label.astype(np.int32), np.int32), C.c_int32)
    # swap variants: same box kind, size, AND cells positions (so halo matches)
    from .sampler import _box_kind
    group_id = {}
    opt_g = []
    for v, o in opts:
        if _box_kind(v) is None:
            opt_g.append(-1); continue
        cells_sig = tuple(sorted(map(tuple, o.cells.tolist())))
        key = (_box_kind(v), o.size, cells_sig)
        opt_g.append(group_id.setdefault(key, len(group_id)))
    p.opt_variants = _ptr(arr(opt_g, np.int32), C.c_int32)
    # hinge pairs: (opt_a, opt_b, dy, dx, dz) resolved from hinges.py's catalogue
    from . import hinges as _hinges
    opt_by_ldraw_yaw: dict[tuple[str, int], int] = {}
    for k, (v_, o_) in enumerate(opts):
        opt_by_ldraw_yaw.setdefault((v_.part.ldraw, o_.rot), k)
    hinge_rows: list[int] = []
    for pair in _hinges.load_all_pairs():
        a_opt = opt_by_ldraw_yaw.get((pair["a_ldraw"], pair["a_yaw"]))
        if a_opt is None:
            continue
        b_opt = opt_by_ldraw_yaw.get((pair["b_ldraw"], pair["yaw_s"]))
        if b_opt is None:
            continue
        a_h = opts[a_opt][1].size[0]
        if pair["stacked"]:
            hinge_rows.extend([a_opt, b_opt, a_h, 0, 0])
        else:
            hinge_rows.extend([a_opt, b_opt, pair["dy_s"], pair["dx_s"], pair["dz_s"]])
    if hinge_rows:
        p.n_hinge = len(hinge_rows) // 5
        p.hinge_data = _ptr(arr(np.array(hinge_rows, np.int32), np.int32), C.c_int32)
    else:
        p.n_hinge = 0
    # multi-scale warm-start: resolve each Placement to an opt index
    if init_placements:
        opt_by_ldraw_yaw: dict[tuple[str, int], int] = {}
        for k_, (v_, o_) in enumerate(opts):
            opt_by_ldraw_yaw.setdefault((v_.part.ldraw, o_.rot), k_)
        init_opt_arr, init_c_arr, init_y_arr, init_x_arr, init_z_arr = [], [], [], [], []
        for pl in init_placements:
            k_ = opt_by_ldraw_yaw.get((pl.part.ldraw, pl.rotation_deg))
            if k_ is None:
                continue
            init_opt_arr.append(k_); init_c_arr.append(pl.color_idx)
            init_y_arr.append(pl.y_plate); init_x_arr.append(pl.x_stud); init_z_arr.append(pl.z_stud)
        if init_opt_arr:
            p.n_init = len(init_opt_arr)
            p.init_opt = _ptr(arr(np.array(init_opt_arr, np.int32), np.int32), C.c_int32)
            p.init_colour = _ptr(arr(np.array(init_c_arr, np.int32), np.int32), C.c_int32)
            p.init_y = _ptr(arr(np.array(init_y_arr, np.int32), np.int32), C.c_int32)
            p.init_x = _ptr(arr(np.array(init_x_arr, np.int32), np.int32), C.c_int32)
            p.init_z = _ptr(arr(np.array(init_z_arr, np.int32), np.int32), C.c_int32)
        else:
            p.n_init = 0
    else:
        p.n_init = 0
    rc = lib.e2l_sample(C.byref(p))
    if rc == 1:
        raise RuntimeError("native sampler panicked")
    if rc == 2:
        raise RuntimeError("native sampler: more parts than cells (out_cap)")
    n = int(out_n[0])
    items = [(opts[int(out["opt"][i])][0], opts[int(out["opt"][i])][1], int(out["colour"][i]),
              int(out["y"][i]), int(out["x"][i]), int(out["z"][i])) for i in range(n)]
    keys = ("energy", "accepted", "t_start", "kl", "local_per_pair", "fidelity", "bad_cells",
            "unsupported", "bare_side_connectors", "parts", "pairs", "energy_recomputed",
            "kl_paths", "outline", "content_loss", "style_loss",
            "kl_paths4", "entropy_paths4", "revkl_paths4", "size_kl")
    inf = {k: float(v) for k, v in zip(keys, info)}
    for k in ("bad_cells", "unsupported", "bare_side_connectors", "parts"):
        inf[k] = int(inf[k])
    return items, inf, hist[:int(hist_n[0])].tolist()


if __name__ == "__main__":
    if sys.argv[1:] == ["build"]:
        print(build())
    else:
        print(__doc__)
