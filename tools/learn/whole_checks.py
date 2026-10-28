"""Whole-build hard checks that a local judge (V14 scores 50-brick pieces) cannot see.

    largest  share of bricks in the largest connector-connected piece (BrickNet connector graph, not touching boxes)
    pieces   number of connected pieces (pieces_per_100: per 100 parts)
    floating share of bricks in pieces that do not reach the ground (report only: real sets reach 0.77 at p90 —
             vehicles on wheels, accessories, clip / pin links the connector graph misses)
    fill     IoU of the build's occupancy with a target grid at the best vertical offset (only when a target is given)
    outline  mean IoU of the top / side / front silhouettes with the target's (completeness; inner holes do not count)
    parsed   share of placed parts the connector graph could read (non-catalogue / scaled parts are dropped)

    BRICKNET_DATA=... .venv-arm/bin/python tools/learn/whole_checks.py build.ldr [--target target.npy]
    ... --calibrate          # distribution over held-out real sets (the thresholds come from here)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "set_study"), str(HERE.parents[1] / "src")]


MAX_PIECES_PER_100 = 8.0          # real held-out sets p90 (--calibrate, 65 sets, 2026-10-28)
MIN_LARGEST = 0.49                # real held-out sets p10


def passes(r: dict) -> bool:
    """Hard gate: inside the range of real sets on connectivity."""
    return r["parts"] > 0 and r["pieces_per_100"] <= MAX_PIECES_PER_100 and r["largest"] >= MIN_LARGEST


def whole_checks_text(ldr_text: str, target: np.ndarray | None = None) -> dict:
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".ldr", delete=False) as f: f.write(ldr_text)
    try: return whole_checks(Path(f.name), target)
    finally: Path(f.name).unlink()


def whole_checks(path: Path, target: np.ndarray | None = None) -> dict:
    from bricknet import load_catalog
    from bricknet_shape import boxes_to_grid, part_boxes
    from ldraw_read import flatten
    from m2b_refine import load_build
    n_all = len(flatten(path)); g = load_build(path); n = len(g.part_ids)
    if n == 0: return {"parts": 0, "parsed": 0.0, "pieces": 0, "pieces_per_100": 0.0, "largest": 0.0, "floating": 1.0}
    boxes = part_boxes(g, load_catalog().id_to_stem, mats=g.transforms)       # grid units, ground at y = 0
    comps = g.components
    grounded = sum(len(c) for c in comps if boxes[list(c), 0, 1].min() < 1.0)   # piece reaches the lowest plate
    out = {"parts": n, "parsed": n / max(1, n_all), "pieces": len(comps), "pieces_per_100": 100 * len(comps) / n,
           "largest": len(comps[0]) / n, "floating": 1 - grounded / n}
    if target is not None:                                                      # both centred on x/z by whole studs, ground at 0
        occ = boxes_to_grid(boxes, target.shape); t = target.copy()
        for ax in (1, 2):
            io, it = np.nonzero(occ.any(tuple(i for i in range(3) if i != ax)))[0], np.nonzero(t.any(tuple(i for i in range(3) if i != ax)))[0]
            if len(io) and len(it): t = np.roll(t, int(round((io.min() + io.max()) / 2 - (it.min() + it.max()) / 2)), axis=ax)
        iou = lambda tt: (occ & tt).sum() / max(1, (occ | tt).sum())                              # noqa: E731
        dy = max(range(-t.shape[0] // 2, t.shape[0] // 2), key=lambda d: iou(np.roll(t, d, axis=0)))                 # best vertical offset
        t = np.roll(t, dy, axis=0); out["fill"] = float(iou(t))
        # outline: silhouettes from the front, the side and above; holes inside a model do not change them, missing walls do
        out["outline"] = float(np.mean([(occ.any(ax) & t.any(ax)).sum() / max(1, (occ.any(ax) | t.any(ax)).sum()) for ax in (0, 1, 2)]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ldr", type=Path, nargs="*"); ap.add_argument("--target", type=Path)
    ap.add_argument("--calibrate", action="store_true", help="held-out real sets (v14_build.is_val)")
    a = ap.parse_args()
    if a.calibrate:
        from v14_build import OMR, is_val, set_key
        rows = []
        for f in sorted(OMR.glob("*.mpd")):
            if not is_val(set_key(f)): continue
            try: r = whole_checks(f)
            except Exception as e: print(f"{f.name}: {e!r}"[:120]); continue  # noqa: BLE001
            if r["parts"] >= 60: rows.append(r)
        for k in ("parsed", "pieces", "pieces_per_100", "largest", "floating"):
            v = np.array([r[k] for r in rows], float)
            print(f"{k:9s} n={len(v)}  p10 {np.percentile(v, 10):.3f}  median {np.median(v):.3f}  p90 {np.percentile(v, 90):.3f}")
        return
    tgt = np.load(a.target) if a.target else None
    for p in a.ldr:
        r = whole_checks(p, tgt)
        print(p, "PASS" if passes(r) else "FAIL", json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()}))


if __name__ == "__main__":
    main()
