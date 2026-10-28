"""Scorecard: every judge on a set of builds of one target, ranked.

    connectivity  whole_checks: pieces / 100 parts, largest piece share, PASS = inside the real-set range
    outline       silhouettes (top / side / front) vs the target: completeness; holes inside a model are fine
    fill          volume IoU with the target grid (mesh voxelised as m2b_watermill.mesh_to_grid), best vertical offset
    v14           V14 local realism, mean over 10 seeds x 20 pieces of 50 bricks
    render        graded render judge (render_grade.py), whole-model look; render_logit ranks where the 0-1 score is ~0

    BRICKNET_DATA=... .venv-arm/bin/python tools/learn/scorecard.py --target target.npy --out report/ a.ldr b.ldr ...
Writes <out>/scorecard.csv (ranked: outline >= MIN_OUTLINE first, then PASS, then render logit, then v14) and renders under <out>/png/.
"""
from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "set_study"), str(HERE.parents[1] / "src")]

_V14 = None
MIN_OUTLINE = 0.8                 # silhouettes vs the target: the finished sampler watermill 0.94, an unfinished 306-brick build 0.55


def _judge(args) -> dict:
    """connectivity + fill + V14 for one build (runs in a worker; V14 loaded once per worker)."""
    global _V14
    path, target, png = args
    import torch
    torch.set_num_threads(1)
    from ldraw_read import flatten, part_bbox
    from realism import _parts_with_grid
    from v8_score import load_v8, score_items
    from whole_checks import passes, whole_checks
    import render_judge as RJ
    row = {"build": str(path)}
    try:
        r = whole_checks(Path(path), np.load(target) if target else None); row.update(r); row["pass"] = passes(r)
        _V14 = _V14 or load_v8(RJ.VC / "v14_model.pt")
        ps = flatten(Path(path)); bb = {}
        for q in ps:
            if q.name not in bb:
                try: bb[q.name] = part_bbox(q.name)
                except Exception: bb[q.name] = None                                    # noqa: BLE001
        it = _parts_with_grid(ps, bb)
        row["v14"] = float(np.nanmean([score_items(_V14, it, bb, {}, k=20, seed=1000 + s) for s in range(10)]))
        RJ._render((Path(path), Path(png)))
    except Exception as e:                                                             # noqa: BLE001
        row["error"] = repr(e)[:200]
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ldr", type=Path, nargs="+"); ap.add_argument("--target", type=Path); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args(); (a.out / "png").mkdir(parents=True, exist_ok=True)
    names = [f"{i:03d}_" + "__".join(p.parts[-3:]).replace(".ldr", "").replace("/", "_") for i, p in enumerate(a.ldr)]
    jobs = [(str(p), str(a.target) if a.target else None, str(a.out / "png" / f"{n}.png")) for p, n in zip(a.ldr, names)]
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for k, r in enumerate(ex.map(_judge, jobs), 1):
            rows.append(r)
            if k % 20 == 0: print(f"  judged {k}/{len(jobs)}", flush=True)
    import render_grade as G
    import render_judge as RJ
    import torch
    m = RJ.make_model(); m.load_state_dict(torch.load(G.MODEL_PT, weights_only=False)["state_dict"]); m.eval()   # our own checkpoint
    for r, (_, _, png) in zip(rows, jobs):
        r["png"] = png
        if Path(png).exists():
            with torch.no_grad(): lg = m(RJ.to_tensor(RJ.views(Path(png))[None]))[0]
            r["render_logit"] = float(lg[1] - lg[0]); r["render"] = float(torch.sigmoid(lg[1] - lg[0]))   # logit still ranks where the 0-1 score is ~0
    full = lambda r: r.get("outline", 1.0) >= MIN_OUTLINE                              # noqa: E731  complete outline (inner holes are fine)
    rows.sort(key=lambda r: (not full(r), not r.get("pass", False), -r.get("render_logit", -99), -r.get("v14", -1)))
    cols = ["build", "pass", "parts", "pieces_per_100", "largest", "outline", "fill", "v14", "render", "render_logit", "png", "error"]
    with open(a.out / "scorecard.csv", "w", newline="") as f:
        w = csv.DictWriter(f, cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)                    # noqa: E731
    print(f"{'#':>3} {'pass':4} {'parts':>5} {'pc/100':>6} {'largest':>7} {'outline':>7} {'v14':>5} {'logit':>6}  build")
    for i, r in enumerate(rows):
        if "error" in r: print(f"{i:>3} ERROR {r['error'][:80]}  {r['build']}"); continue
        print(f"{i:>3} {'PASS' if r['pass'] else 'fail':4} {r['parts']:>5} {r['pieces_per_100']:>6.1f} {r['largest']:>7.2f} {fmt(r.get('outline', '-')):>7} "
              f"{r['v14']:>5.3f} {r.get('render_logit', float('nan')):>6.2f}  {'/'.join(Path(r['build']).parts[-3:])}")
    print(f"-> {a.out / 'scorecard.csv'}")


if __name__ == "__main__":
    main()
