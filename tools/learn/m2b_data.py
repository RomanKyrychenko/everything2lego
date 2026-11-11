"""Mesh2Brick M0 data (docs/mesh2brick_architecture.md §2, §6): BrickNet graphs or OMR models ->
canonical ground-up build order, BrickNet connector-relative actions, per-brick poses and the target
occupancy grid (bbox fill + solid watertight-mesh fill), sharded npz on LEGO_DATA/analysis/m2b/<split>/.

    python tools/learn/m2b_data.py --split val                        # 512 models, prints the M0 stats
    python tools/learn/m2b_data.py --split sft --workers 4
    python tools/learn/m2b_data.py --split omr --grid 96,64,64 --workers 4

Every connected component with >= MIN_COMP bricks becomes its own model (rootless BrickNet graphs
place every component at the origin, and the action grammar has no way to position a second one);
the dropped fraction is reported.  Per step (one per placed brick, CSR by `ptr`): part int32 (BrickNet vocab id),
color int32 (LDraw code), parent int32 (-1 = root), fam int8 (FAMS index), pconn / cconn int16 (flat
connector index within the parent / child part, see bricknet.graph._part_info), yaw int16 deg,
flip int8, slide int16, rot int16 (n,3) deg, mats float32 (n,4,4) *realised* poses (actions ->
bricknet.decode_graph) in LDU in the grid frame (grid index = (x/20 + W/2, -y/8, z/20 + D/2)),
boxes float16 (n,2,3) grid units of those poses.  Per model: grid_bbox / grid_mesh uint8 packbits
(H*W*D/8) from the *source* geometry, n_total int16 (bricks in the whole source model), overflow
bool, rt_err float16 (max |realised - source| in LDU, rotation error counted x 20 LDU),
miss int16 (bricks without a watertight mesh, bbox-filled), src (row id or OMR file name, '#k' for
the k-th component)."""
from __future__ import annotations

import argparse
import heapq
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1] / "tools" / "set_study"))
from bricknet_shape import GRID, boxes_to_grid, part_boxes  # noqa: E402

DATA = Path(os.environ.get("LEGO_DATA", "/Volumes/PRO-G40/lego_data"))
WATERTIGHT = Path(os.environ.get("BRICKNET_DATA", DATA / "bricknet")) / "watertight" / "meshes"
OUT = DATA / "analysis" / "m2b"
FAMS = ("stud", "hinge", "axle", "ball", "fixed")
MIN_COMP, MAX_COMP = 4, 4096                                  # bricks; components outside this range are dropped
STEP_KEYS = ("part", "color", "parent", "fam", "pconn", "cconn", "yaw", "flip", "slide", "rot", "mats", "boxes")


# ----------------------------------------------------------------------------------------- order + actions
def canonical_order(g, mats, boxes):
    """Ground-up connected order over the largest component: always place the frontier brick with the
    lowest (bottom plate, z, x); its parent is the placed neighbour whose mating connector sits lowest."""
    from bricknet.graph import _FAM_RANK, _adjacency, _part_info
    comp = set(g.components[0]); adj = _adjacency(g.edges, comp); edges = g.edges
    key = {i: (round(float(boxes[i, 0, 1]) * 2), round(float(boxes[i, 0, 2]) * 2), round(float(boxes[i, 0, 0]) * 2), i) for i in comp}
    start = min(comp, key=key.get); order = [start]; placed = {start}; parent_of = {}
    heap = [(key[v], v) for v, _, _ in adj.get(start, [])]; heapq.heapify(heap)
    while heap:
        _, v = heapq.heappop(heap)
        if v in placed:
            continue
        best = None
        for u, ei, _ in adj.get(v, []):
            if u not in placed:
                continue
            row = edges[ei]; fwd = int(row["a"]) == u
            pc = int(row["a_conn"] if fwd else row["b_conn"])
            y_ldu = float((mats[u] @ _part_info(g.part_ids[u]).frames[pc])[1, 3])   # LDraw -Y up: larger = lower
            cand = (-y_ldu, _FAM_RANK[str(row["family"])], u)
            if best is None or cand < best[0]:
                best = (cand, (u, ei, fwd))
        parent_of[v] = best[1]; placed.add(v); order.append(v)
        for w, _, _ in adj.get(v, []):
            if w not in placed:
                heapq.heappush(heap, (key[w], w))
    return order, parent_of


def actions(g, order, parent_of):
    """Tree edges of the order -> per-step action columns + the Tree (for the round trip)."""
    from bricknet.core import AxleEdge, BallEdge, HingeEdge, StudEdge
    from bricknet.graph import _build_tree
    tree = _build_tree(g, order, parent_of); n = len(order)
    cols = {"parent": np.full(n, -1, np.int32), "fam": np.zeros(n, np.int8), "pconn": np.zeros(n, np.int16),
            "cconn": np.zeros(n, np.int16), "yaw": np.zeros(n, np.int16), "flip": np.zeros(n, np.int8),
            "slide": np.zeros(n, np.int16), "rot": np.zeros((n, 3), np.int16)}
    for i, e in enumerate(tree.edges, start=1):
        u, ei, fwd = parent_of[order[i]]; row = g.edges[ei]
        cols["parent"][i] = e.parent
        cols["pconn"][i], cols["cconn"][i] = (row["a_conn"], row["b_conn"]) if fwd else (row["b_conn"], row["a_conn"])
        if isinstance(e, StudEdge):
            cols["fam"][i], cols["yaw"][i] = 0, e.yaw
        elif isinstance(e, HingeEdge):
            cols["fam"][i], cols["yaw"][i], cols["flip"][i] = 1, e.yaw, e.flip
        elif isinstance(e, AxleEdge):
            cols["fam"][i], cols["yaw"][i], cols["flip"][i], cols["slide"][i] = 2, e.yaw, e.flip, e.slide
        elif isinstance(e, BallEdge):
            cols["fam"][i], cols["rot"][i] = 3, (e.rx, e.ry, e.rz)
        else:
            cols["fam"][i] = 4
    return cols, tree


def round_trip_error(tree, mats, order) -> float:
    """Max pose error (LDU) between the realised tree and the source poses, relative to the root:
    translation error, or rotation error x 20 LDU (one stud of lever arm)."""
    from bricknet.graph import decode_graph, tree_to_graph
    T2 = np.stack(decode_graph(tree_to_graph(tree))); T1 = mats[order]
    rel = np.linalg.inv(T1[0]) @ T1
    return max(float(np.abs(rel[:, :3, 3] - T2[:, :3, 3]).max()), 20 * float(np.abs(rel[:, :3, :3] - T2[:, :3, :3]).max()))


# ----------------------------------------------------------------------------------------- grids
PITCH = 4.0                                                   # LDU; fine solid voxels of each part (1/5 stud, 1/2 plate)
SUB = np.array([[sx, sy, sz] for sx in (-0.25, 0.25) for sy in (-0.25, 0.25) for sz in (-0.25, 0.25)])  # 2x2x2 sub-samples, grid units
_vox_cache: dict[str, tuple | None] = {}


def solid_part(stem: str):
    """(filled bool (nx,ny,nz), origin LDU) solid voxels of the part's watertight mesh at PITCH (cell 0 centred
    on origin), cached on disk under OUT/partvox; None when BrickNet ships no watertight mesh for it."""
    if stem not in _vox_cache:
        cache = OUT / "partvox" / f"{stem}.npz"; src = WATERTIGHT / f"{stem}.ply"; res = None
        if cache.exists():
            z = np.load(cache); res = (z["m"], z["o"])
        elif src.exists():
            import trimesh
            v = trimesh.load(src, force="mesh").voxelized(pitch=PITCH); v.fill()
            res = (v.matrix.astype(bool), v.transform[:3, 3].copy())
            cache.parent.mkdir(parents=True, exist_ok=True); np.savez(cache, m=res[0], o=res[1])
        _vox_cache[stem] = res
    return _vox_cache[stem]


def mesh_grid(part_ids, mats, id_to_stem, grid, boxes) -> tuple[np.ndarray, int]:
    """Solid fill of the posed watertight part meshes, the training-side twin of voxelize(): a cell is
    occupied when >= 4 of its 8 sub-sample points fall inside a part (so stud caps, 4 LDU tall and 12 LDU
    wide, do not fill the cell above a brick).  Parts without a mesh fall back to their bbox.
    Returns (occ, n_fallback)."""
    H, W, D = grid; occ = np.zeros(grid, bool); miss = 0
    for i, (pid, M) in enumerate(zip(part_ids, mats)):
        sp = solid_part(id_to_stem[int(pid)])
        if sp is None:
            occ |= boxes_to_grid(boxes[i:i + 1], grid); miss += 1
            continue
        filled, org = sp; R, t = M[:3, :3], M[:3, 3]
        lo_l, hi_l = org - PITCH / 2, org + (np.array(filled.shape) - 0.5) * PITCH
        corners = np.array([[x, y, z] for x in (lo_l[0], hi_l[0]) for y in (lo_l[1], hi_l[1]) for z in (lo_l[2], hi_l[2])]) @ R.T + t
        g = corners * np.array([1 / 20.0, -1 / 8.0, 1 / 20.0]) + np.array([W / 2.0, 0.0, D / 2.0])   # grid units
        lo, hi = g.min(0), g.max(0)
        xa, xb = max(int(np.floor(lo[0] + 0.5)), 0), min(int(np.ceil(hi[0] - 0.5)), W)
        ya, yb = max(int(np.floor(lo[1] + 0.5)), 0), min(int(np.ceil(hi[1] - 0.5)), H)
        za, zb = max(int(np.floor(lo[2] + 0.5)), 0), min(int(np.ceil(hi[2] - 0.5)), D)
        if xa >= xb or ya >= yb or za >= zb:
            continue
        gy, gx, gz = np.mgrid[ya:yb, xa:xb, za:zb]
        c = np.stack([gx.ravel() + 0.5, gy.ravel() + 0.5, gz.ravel() + 0.5], 1)[:, None, :] + SUB[None]   # (k, 8, 3)
        p = np.stack([(c[..., 0] - W / 2) * 20, -c[..., 1] * 8, (c[..., 2] - D / 2) * 20], -1)          # LDU, grid frame
        idx = np.rint(((p - t) @ R - org) / PITCH).astype(int)                                           # part-local fine cell
        ok = (idx >= 0).all(-1) & (idx < np.array(filled.shape)).all(-1)
        hit = np.zeros(ok.shape, bool); hit[ok] = filled[idx[ok, 0], idx[ok, 1], idx[ok, 2]]
        occ[ya:yb, xa:xb, za:zb] |= (hit.sum(1) >= 4).reshape(gy.shape)
    return occ, miss


# ----------------------------------------------------------------------------------------- per model
def process(g, src, id_to_stem, grid, min_comp: int = MIN_COMP) -> list[dict]:
    """One record per connected component with >= min_comp bricks (each becomes its own model: the
    grammar cannot position a second component).  The grid comes from the source geometry; the stored
    poses/boxes are the *realised* ones (actions -> decode_graph), so training state matches inference."""
    from bricknet.graph import decode_graph, tree_to_graph
    from types import SimpleNamespace
    if len(g.part_ids) == 0:
        return []
    mats = g.transforms if g.transforms is not None else np.stack(decode_graph(g))
    out = []
    for k, comp in enumerate(g.components):
        if len(comp) < min_comp:
            break
        if len(comp) > MAX_COMP:                                                  # pt has 24k-brick dioramas: useless at 32 studs
            continue
        comp = list(comp); sub = _sub_graph(g, comp, mats)                                       # grid centred on this component
        boxes, shift = part_boxes(sub, id_to_stem, mats=sub.transforms, return_shift=True)
        H, W, D = grid; ext = boxes[:, 1].max(0) - boxes[:, 0].min(0)
        order, parent_of = canonical_order(sub, sub.transforms, boxes)
        if len(order) != len(comp):
            continue
        cols, tree = actions(sub, order, parent_of)
        rt = round_trip_error(tree, sub.transforms, order)
        pids = np.array(sub.part_ids, np.int32)[order]
        T0 = _shift_mats(sub.transforms[order], shift)                                               # source poses, grid frame
        T = _shift_mats(sub.transforms[order][0] @ np.stack(decode_graph(tree_to_graph(tree))), shift)  # realised poses, grid frame
        bxr, sr = part_boxes(SimpleNamespace(part_ids=pids), id_to_stem, mats=T, return_shift=True); bx = bxr + sr   # T already shifted
        gb = boxes_to_grid(boxes[order], grid); gm, miss = mesh_grid(pids, T0, id_to_stem, grid, boxes[order])
        rec = dict(cols, part=pids, color=np.array(sub.colors, np.int32)[order], mats=T.astype(np.float32), boxes=bx.astype(np.float16))
        rec.update(grid_bbox=np.packbits(gb.ravel()), grid_mesh=np.packbits(gm.ravel()), n_total=len(g.part_ids),
                   overflow=bool(ext[0] > W or ext[1] > H or ext[2] > D), rt_err=min(rt, 6e4), miss=miss, src=f"{src}#{k}" if k else str(src))
        out.append(rec)
    return out


def _shift_mats(T, shift):
    T = T.copy(); T[:, 0, 3] -= shift[0] * 20; T[:, 2, 3] -= shift[2] * 20; T[:, 1, 3] += shift[1] * 8
    return T


def _sub_graph(g, comp, mats):
    """Graph restricted to one component (node indices remapped), rooted with `mats`."""
    from bricknet.core import Graph
    if len(comp) == len(g.part_ids) and comp == sorted(comp):
        return Graph(g.part_ids, g.colors, mats, g.edges, g.components)
    remap = {old: new for new, old in enumerate(comp)}
    keep = np.array([int(a) in remap for a in g.edges["a"]], bool)
    e = g.edges[keep].copy(); e["a"] = [remap[int(a)] for a in e["a"]]; e["b"] = [remap[int(b)] for b in e["b"]]
    return Graph(tuple(g.part_ids[i] for i in comp), tuple(g.colors[i] for i in comp), mats[comp], e, (tuple(range(len(comp))),))


def _work(args):
    graphs, srcs, grid = args
    from bricknet import load_catalog
    i2s = load_catalog().id_to_stem; out = []
    for g, s in zip(graphs, srcs):
        try:
            out.extend(process(g, s, i2s, grid))
        except Exception as ex:                                                          # noqa: BLE001
            print(f"  ! {s}: {type(ex).__name__}: {ex}", flush=True)
    return out


# ----------------------------------------------------------------------------------------- sources
def omr_graphs(limit: int):
    """OMR .mpd/.ldr -> rooted Graphs via flatten (sub-models expanded) + parse_ldr (connectors)."""
    from bricknet import load_catalog
    from bricknet.data import load_aliases
    from bricknet.graph import parse_ldr
    from brickae_data import SCALE_TOL, omr_files
    from ldraw_read import flatten
    stems, aliases = set(load_catalog().id_to_stem), load_aliases()
    files = omr_files()[: limit or None]
    for f in files:
        lines = []
        for p in flatten(f):
            st = p.name.removesuffix(".dat")
            if st in aliases: st = aliases[st][0]
            if st not in stems or np.abs(np.linalg.norm(p.M[:3, :3], axis=0) - 1).max() > SCALE_TOL:
                continue
            M = p.M; lines.append(f"1 {p.colour} " + " ".join(f"{v:.6g}" for v in [*M[:3, 3], *M[:3, :3].ravel()]) + f" {p.name.removesuffix('.dat')}.dat")
        if lines:
            yield parse_ldr("\n".join(lines) + "\n"), f.name


def npz_mmap(path: Path) -> dict:
    """Memory-map the arrays of an *uncompressed* .npz (BrickNet's pt.npz is 6.6 GB stored, not deflated):
    each member is a contiguous .npy inside the zip, so np.memmap at its data offset works."""
    import zipfile
    from numpy.lib import format as npf
    arrays = {}
    with zipfile.ZipFile(path) as z:
        for info in z.infolist():
            assert info.compress_type == zipfile.ZIP_STORED, f"{info.filename} is compressed; use np.load"
            with z.open(info) as f:
                ver = npf.read_magic(f); shape, fortran, dtype = npf._read_array_header(f, ver); hdr = f.tell()
            with open(path, "rb") as raw:                                         # local header: 30 bytes + name + extra
                raw.seek(info.header_offset + 26); n_name, n_extra = np.frombuffer(raw.read(4), "<u2")
            off = info.header_offset + 30 + int(n_name) + int(n_extra) + hdr
            arrays[info.filename.removesuffix(".npy")] = np.memmap(path, dtype=dtype, mode="r", offset=off, shape=shape, order="F" if fortran else "C")
    return arrays


def load_graphs_range(arrays: dict, a: int, b: int) -> list:
    """Graphs a..b-1 of a CSR-batched BrickNet file given its (memory-mapped) arrays."""
    from bricknet.graph import _graph_from_arrays
    n0, n1 = int(arrays["node_ptr"][a]), int(arrays["node_ptr"][b]); e0, e1 = int(arrays["edge_ptr"][a]), int(arrays["edge_ptr"][b])
    nptr = np.asarray(arrays["node_ptr"][a:b + 1]) - n0; eptr = np.asarray(arrays["edge_ptr"][a:b + 1]) - e0
    cols = [np.asarray(arrays[k][n0:n1]) for k in ("part_ids", "colors")] + [np.asarray(arrays[k][e0:e1]) for k in ("edge_kind", "edge_idx", "edge_yaw", "edge_flip", "edge_rot")]
    out = []
    for i in range(b - a):
        ns, es = slice(nptr[i], nptr[i + 1]), slice(eptr[i], eptr[i + 1])
        out.append(_graph_from_arrays(cols[0][ns], cols[1][ns], cols[2][es], cols[3][es], cols[4][es], cols[5][es], cols[6][es], None))
    return out


def write_vocab(path: Path = OUT / "vocab.json") -> None:
    """Catalog tables the trainer needs without the bricknet package: part stems, connectors per part,
    LDraw colour codes (index = position), connector kind per (part, flat index) packed as strings."""
    import json
    from bricknet import load_catalog
    from bricknet.graph import _part_info
    cat = load_catalog(); n = len(cat.id_to_stem)
    kinds = {k: i for i, k in enumerate(FAMS)}
    fam_of = ["".join(str(kinds["stud" if c[0] in ("stud", "hole") else c[0]]) for c in _part_info(p).conns) for p in range(n)]
    json.dump({"stems": list(cat.id_to_stem), "n_conn": [len(f) for f in fam_of], "conn_fam": fam_of,
               "colors": sorted(cat.code_to_color)}, open(path, "w"))
    print(f"vocab -> {path}: {n} parts, {len(cat.code_to_color)} colours", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "sft", "pt", "omr", "vocab"], required=True)
    ap.add_argument("--grid", default="96,32,32")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--shard", type=int, default=5000, help="models per output shard")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    if a.split == "vocab":
        return write_vocab()
    grid = tuple(int(v) for v in a.grid.split(",")); out = a.out or OUT / a.split; out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if a.split == "omr":
        pairs = list(omr_graphs(a.limit)); graphs, srcs = [p[0] for p in pairs], [p[1] for p in pairs]
    else:                                                                            # memory-mapped, one shard at a time
        arrays = npz_mmap(DATA / "bricknet" / f"{a.split}.npz"); n_models = len(arrays["node_ptr"]) - 1
        if a.limit: n_models = min(a.limit, n_models)
        graphs = range(n_models); srcs = list(range(n_models))
    print(f"{a.split}: {len(graphs)} models in {time.time() - t0:.0f}s -> {out}  grid {grid}", flush=True)
    stats = {k: [] for k in ("n", "n_total", "rt", "pdist", "over", "miss", "fill_b", "fill_m")}
    for si in range(0, len(graphs), a.shard):
        if (out / f"m2b_{si // a.shard:03d}.npz").exists(): continue                 # resumable
        gs = load_graphs_range(arrays, si, min(si + a.shard, len(graphs))) if isinstance(graphs, range) else graphs[si:si + a.shard]
        ss = srcs[si:si + a.shard]
        per = max(1, -(-len(gs) // (a.workers * 4)))
        jobs = [(gs[i:i + per], ss[i:i + per], grid) for i in range(0, len(gs), per)]
        if a.workers > 1:
            with ProcessPoolExecutor(a.workers) as ex: recs = [r for part in ex.map(_work, jobs) for r in part]
        else:
            recs = [r for j in jobs for r in _work(j)]
        if not recs: continue
        cat = {k: np.concatenate([r[k] for r in recs]) for k in STEP_KEYS}
        cat["ptr"] = np.cumsum([0] + [len(r["part"]) for r in recs]).astype(np.int64)
        for k in ("grid_bbox", "grid_mesh"): cat[k] = np.stack([r[k] for r in recs])
        for k, dt in (("n_total", np.int16), ("overflow", bool), ("rt_err", np.float16), ("miss", np.int16)): cat[k] = np.array([r[k] for r in recs], dt)
        cat["src"] = np.array([r["src"] for r in recs]); cat["grid"] = np.array(grid, np.int32)
        np.savez_compressed(out / f"m2b_{si // a.shard:03d}.npz", **cat)
        for r in recs:
            n = len(r["part"]); stats["n"].append(n); stats["rt"].append(r["rt_err"])
            if "#" not in r["src"]: stats["n_total"].append(r["n_total"])
            stats["pdist"].extend((np.arange(n) - r["parent"])[1:].tolist()); stats["over"].append(r["overflow"]); stats["miss"].append(r["miss"])
            stats["fill_b"].append(np.unpackbits(r["grid_bbox"]).mean()); stats["fill_m"].append(np.unpackbits(r["grid_mesh"]).mean())
        print(f"  shard {si // a.shard}: {len(recs)}/{len(gs)} models  {time.time() - t0:.0f}s", flush=True)
    n, nt, rt, pd = (np.array(stats[k]) for k in ("n", "n_total", "rt", "pdist"))
    print(f"done {len(n)}/{len(graphs)} models in {time.time() - t0:.0f}s ({(time.time() - t0) / max(1, len(n)) * 1e3:.0f} ms/model)\n"
          f"  bricks kept {n.sum()}/{nt.sum()} = {n.sum() / max(1, nt.sum()):.1%} (largest component)  parts/model median {np.median(n):.0f} p90 {np.percentile(n, 90):.0f} max {n.max()}\n"
          f"  round trip: {(rt <= 1).mean():.1%} within 1 LDU, {(rt <= 4).mean():.1%} within 4  (p99 {np.percentile(rt, 99):.2f} LDU)\n"
          f"  parent distance: p50 {np.median(pd):.0f} p99 {np.percentile(pd, 99):.0f}  outside 256-window {(pd > 256).mean():.3%}  parent-above-child n/a (order is by bottom)\n"
          f"  overflow {np.mean(stats['over']):.1%}  parts w/o collider {np.sum(stats['miss'])}  fill bbox {np.mean(stats['fill_b']):.2%} mesh {np.mean(stats['fill_m']):.2%}", flush=True)


if __name__ == "__main__":
    main()
