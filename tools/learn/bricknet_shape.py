"""BrickNet graph -> solid occupancy grid in the project's voxel convention.

Matches `everything2lego.voxelize.voxelize()` / `_fit_transform`: one grid
unit is one stud on x/z and one plate on y, ground at y=0, model centred
on x/z, `occupied` is solid.  Difference from the mesh path: no scaling —
a LEGO model has an intrinsic size, so we place it at native scale and
pad/crop into a fixed (H, W, D) = (96, 32, 32) box.  At inference the
target mesh goes through `voxelize(mesh, max_studs=32)` and lands in the
same box, so the encoder sees one distribution.

Per-part geometry is the LDraw part bbox (`ldraw_read.part_bbox`, 99.8%
coverage over sft), transformed by the graph's absolute 4x4 (LDU, -Y up).

    python tools/learn/bricknet_shape.py            # self-test on val
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "set_study"))

STUD_LDU, PLATE_LDU = 20.0, 8.0
GRID = (96, 32, 32)                                                        # (H plates, W studs, D studs)

_bbox_cache: dict[str, tuple | None] = {}


def _part_bbox(stem: str):
    from ldraw_read import part_bbox
    if stem not in _bbox_cache:
        try:
            _bbox_cache[stem] = part_bbox(stem + ".dat")
        except Exception:
            _bbox_cache[stem] = None
    return _bbox_cache[stem]


def part_boxes(graph, id_to_stem, mats=None, return_shift=False) -> np.ndarray:
    """(N, 2, 3) world-space boxes in grid units [x_stud, y_plate, z_stud],
    ground-relative (min y -> 0), centred on x/z.  Parts without a bbox get
    a 1x1x1 placeholder at their origin.  `mats` overrides decode_graph;
    `return_shift` also returns the (x, y, z) grid-unit shift applied."""
    from bricknet import decode_graph
    T = np.stack(decode_graph(graph)) if mats is None else np.asarray(mats)  # (N, 4, 4) LDU, -Y up
    boxes = np.empty((len(T), 2, 3), dtype=np.float32)
    for i, (pid, M) in enumerate(zip(graph.part_ids, T)):
        bb = _part_bbox(id_to_stem[int(pid)])
        if bb is None:
            lo = hi = np.zeros(3)
        else:
            lo, hi = np.asarray(bb[0], float), np.asarray(bb[1], float)
        c = np.array([[x, y, z, 1.0] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        w = (M @ c.T).T[:, :3]
        a, b = w.min(0), w.max(0)
        # LDU -> grid units; LDraw +Y is down, so plate_y = -Y
        boxes[i, 0] = [a[0] / STUD_LDU, -b[1] / PLATE_LDU, a[2] / STUD_LDU]
        boxes[i, 1] = [b[0] / STUD_LDU, -a[1] / PLATE_LDU, b[2] / STUD_LDU]
        if bb is None:                                                     # placeholder cell
            boxes[i, 1] = boxes[i, 0] + 1.0
    lo_all, hi_all = boxes[:, 0].min(0), boxes[:, 1].max(0)
    # centre x/z by a whole number of studs so stud-aligned box edges stay on cell boundaries, ground y=0
    shift = np.array([lo_all[0] + round((hi_all[0] - lo_all[0]) / 2), lo_all[1], lo_all[2] + round((hi_all[2] - lo_all[2]) / 2)], np.float32)
    boxes -= shift
    return (boxes, shift) if return_shift else boxes


def boxes_to_grid(boxes: np.ndarray, grid=GRID) -> np.ndarray:
    """Solid-fill boxes into a bool (H, W, D) grid.  x/z are centred in the
    box (index W//2 is x=0); y=0 is the bottom plate.  Overflow is cropped."""
    H, W, D = grid
    occ = np.zeros(grid, dtype=bool)
    ox, oz = W / 2.0, D / 2.0
    e = 1e-3                                                               # box edges sit on cell boundaries: snap float noise
    for (x0, y0, z0), (x1, y1, z1) in boxes:
        # cells whose centre lies inside the box; at least one cell per axis
        xa, xb = int(np.floor(x0 + ox + 0.5 + e)), max(int(np.floor(x0 + ox + 0.5 + e)) + 1, int(np.ceil(x1 + ox - 0.5 - e)))
        ya, yb = int(np.floor(y0 + 0.5 + e)), max(int(np.floor(y0 + 0.5 + e)) + 1, int(np.ceil(y1 - 0.5 - e)))
        za, zb = int(np.floor(z0 + oz + 0.5 + e)), max(int(np.floor(z0 + oz + 0.5 + e)) + 1, int(np.ceil(z1 + oz - 0.5 - e)))
        xa, xb = max(xa, 0), min(xb, W); ya, yb = max(ya, 0), min(yb, H); za, zb = max(za, 0), min(zb, D)
        if xa < xb and ya < yb and za < zb:
            occ[ya:yb, xa:xb, za:zb] = True
    return occ


def graph_to_grid(graph, id_to_stem, grid=GRID) -> np.ndarray:
    return boxes_to_grid(part_boxes(graph, id_to_stem), grid)


if __name__ == "__main__":
    import time
    from bricknet import load_catalog, load_graphs
    cat = load_catalog(); i2s = cat.id_to_stem
    gs = load_graphs("/Volumes/PRO-G40/lego_data/bricknet/val.npz")
    g = gs[344]
    bx = part_boxes(g, i2s)
    ext = bx[:, 1].max(0) - bx[:, 0].min(0)
    occ = graph_to_grid(g, i2s)
    ys, xs, zs = np.nonzero(occ)
    print(f"val[344]: {len(g.part_ids)} parts  extent studs/plates/studs = {ext.round(1)}  (transform-only measured 10.9/14.5/2.8)")
    print(f"  grid fill: {occ.sum()} cells = {occ.mean():.2%}  bbox of filled: y[{ys.min()},{ys.max()}] x[{xs.min()},{xs.max()}] z[{zs.min()},{zs.max()}]")
    assert ys.min() == 0, "ground must be at y=0"
    assert abs(((xs.min() + xs.max()) / 2) - GRID[1] / 2) <= 1.5, "x must be centred"
    # throughput + overflow rate over 200 sft models
    sft = load_graphs("/Volumes/PRO-G40/lego_data/bricknet/sft.npz")
    import random; random.seed(0); idx = random.sample(range(len(sft)), 200)
    t0 = time.time(); over = 0; fills = []
    for k in idx:
        b = part_boxes(sft[k], i2s)
        e = b[:, 1].max(0) - b[:, 0].min(0)
        over += bool(e[0] > GRID[1] or e[1] > GRID[0] or e[2] > GRID[2])
        fills.append(boxes_to_grid(b).mean())
    dt = time.time() - t0
    print(f"200 sft models: {dt/200*1000:.1f} ms/model  overflow(>{GRID[1]}x{GRID[0]}x{GRID[2]}): {over}/200  mean fill {np.mean(fills):.2%}")
    print("self-test OK")
