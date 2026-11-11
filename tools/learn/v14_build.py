"""V14 dataset: 50-brick connected pieces from real sets (pos), spoiled copies, damaged sets and our generated builds.

Changes vs V11/V13:
- flatten() now yields real parts only (no primitives / subparts / hose segments), so real pieces are bricks.
- stores part NAMES (any vocabulary can be applied at train time) in compact on-disk arrays, streamed.
- no colour-only spoiler (the judge is colour-blind: those "negatives" equalled their positives).
- validation split by set (a held-out set's damaged copies are held out too).
- watermill builds and BrickNet generator outputs are kept out: they are evaluation targets.

Output dir: names.json + node_name.i32, node_cont.f16 (N,6), edge_ei.i16 (E,2), edge_ea.f16 (E,8),
g_node.i64 / g_edge.i64 (offsets, G+1), g_y.i8, g_kind.i8 (0 pos, 1 spoiled, 2 damaged set, 3 generated), g_val.i8.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "src"), str(HERE.parents[1] / "tools" / "set_study")]

TARGET_N = 50
A = Path("/Volumes/PRO-G40/lego_data/analysis")
OMR = Path("/Volumes/PRO-G40/lego_data/omr/files")
OUT = A / "vis_classifiers" / "v14_data"
SPOILS = ("types", "positions", "edges", "full")


def generated_files() -> list[Path]:
    """Our pipelines' builds (sampler, Mesh2Brick, BrickNet autoencoder outputs); never real inputs or watermills."""
    pats = ["realism*/_ours_scratch/*/model.ldr", "realism*/_ours_omr_scratch/*/model.ldr",
            "m2b/run*/gen*/*.ldr", "brickae/*/eval*/*_output.ldr"]
    fs = sorted({f for p in pats for f in A.glob(p) if not f.name.startswith("._")})
    return [f for f in fs if "watermill" not in str(f).lower()]


def damaged_files() -> list[Path]:
    fs = sorted(A.glob("vis_classifiers/vis_cls4/damaged_omr/*.ldr")) + sorted(A.glob("vis_classifiers/vis_cls/damaged/*.ldr"))
    return [f for f in fs if not f.stem.endswith("_flat") and not f.name.startswith("._")]


def is_val(key: str) -> bool:
    return zlib.crc32(key.encode()) % 20 == 0                                # 5 % of sets, stable


def set_key(path: Path) -> str:
    return path.stem.split("_")[0]                                           # "4956-1_Small-Houses" -> "4956-1"


def _items(path: Path):
    from ldraw_read import flatten, part_bbox
    from realism import _parts_with_grid, _neighbour_pairs
    parts = flatten(path); bb = {}
    for p in parts:
        if p.name not in bb:
            try: bb[p.name] = part_bbox(p.name)
            except Exception: bb[p.name] = None                              # noqa: BLE001
    items = _parts_with_grid(parts, bb)
    if len(items) < TARGET_N + 5: return None, None, None
    adj: dict = {}
    for a, b, _ in _neighbour_pairs(items):
        adj.setdefault(a, []).append(b); adj.setdefault(b, []).append(a)
    return items, adj, bb


def _pack(t: dict, names: list) -> tuple:
    return (names, t["cont"].astype(np.float16), t["ei"].T.astype(np.int16), t["ea"].astype(np.float16))


def task(path: str, n_sub: int, kind: int, spoil_p: float, seed: int, catalog: list) -> list:
    from subgraph_extract import SPOILERS, extract_subgraph, nodes_edges_to_tensors
    rng = random.Random(seed)
    try: items, adj, bb = _items(Path(path))
    except Exception: return []                                               # noqa: BLE001
    if items is None: return []
    out = []
    for _ in range(n_sub):
        sub = extract_subgraph(items, adj, TARGET_N, rng)
        if sub is None or not sub[1]: continue
        nodes, edges = sub
        t = nodes_edges_to_tensors(items, nodes, edges, bb, {}); names = [items[i][0] for i in nodes]
        out.append((_pack(t, names), 1 if kind == 0 else 0, kind))
        if kind == 0 and rng.random() < spoil_p:
            s = rng.choice(SPOILS); sp = SPOILERS[s](t, catalog, {}, bb, rng) if s != "types" else t
            sn = [rng.choice(catalog) for _ in names] if s in ("types", "full") else names
            out.append((_pack(sp, sn), 0, 1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT); ap.add_argument("--per-set", type=int, default=250)
    ap.add_argument("--per-neg", type=int, default=50); ap.add_argument("--spoil-p", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=7); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    from everything2lego.bricks import full_catalog
    catalog = sorted({p.ldraw for p in full_catalog()})
    jobs = [(f, a.per_set, 0) for f in sorted(OMR.glob("*.mpd"))] + [(f, a.per_neg, 2) for f in damaged_files()] + \
           [(f, a.per_neg, 3) for f in generated_files()]
    if a.limit: random.Random(0).shuffle(jobs); jobs = jobs[:a.limit]
    print(f"jobs: {sum(k == 0 for _, _, k in jobs)} real sets, {sum(k == 2 for _, _, k in jobs)} damaged, "
          f"{sum(k == 3 for _, _, k in jobs)} generated", flush=True)
    fh = {k: open(a.out / k, "wb") for k in ("node_name.i32", "node_cont.f16", "edge_ei.i16", "edge_ea.f16")}
    names: dict = {}; g_node = [0]; g_edge = [0]; g_y = []; g_kind = []; g_val = []; t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(task, str(f), n, k, a.spoil_p, i * 7919 + 13, catalog): (f, k) for i, (f, n, k) in enumerate(jobs)}
        for done, fut in enumerate(as_completed(futs), 1):
            f, k = futs[fut]; v = is_val(set_key(f)) if k in (0, 2) else is_val(str(f))
            for (nm, cont, ei, ea), y, kind in fut.result():
                fh["node_name.i32"].write(np.array([names.setdefault(n, len(names)) for n in nm], np.int32).tobytes())
                fh["node_cont.f16"].write(cont.tobytes()); fh["edge_ei.i16"].write(ei.tobytes()); fh["edge_ea.f16"].write(ea.tobytes())
                g_node.append(g_node[-1] + len(nm)); g_edge.append(g_edge[-1] + len(ei)); g_y.append(y); g_kind.append(kind); g_val.append(v)
            if done % 100 == 0 or done == len(jobs):
                kc = np.bincount(g_kind, minlength=4)
                print(f"  {done}/{len(jobs)} files  graphs {len(g_y)} (pos {kc[0]} spoiled {kc[1]} damaged {kc[2]} generated {kc[3]})  {time.time() - t0:.0f}s", flush=True)
    for h in fh.values(): h.close()
    for k, v, dt in (("g_node.i64", g_node, np.int64), ("g_edge.i64", g_edge, np.int64), ("g_y.i8", g_y, np.int8),
                     ("g_kind.i8", g_kind, np.int8), ("g_val.i8", g_val, np.int8)):
        np.asarray(v, dt).tofile(a.out / k)
    (a.out / "names.json").write_text(json.dumps(sorted(names, key=names.get)))
    print(f"done: {len(g_y)} graphs, {g_node[-1]} nodes, {g_edge[-1]} edges, {len(names)} part names, val {np.mean(g_val):.3f}", flush=True)


if __name__ == "__main__":
    main()
