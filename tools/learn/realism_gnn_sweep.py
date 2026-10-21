"""Sampler hyperparameter sweep guided by the GNN realism scorer.
Picks the config with the highest realism-GNN score.

    python tools/learn/realism_gnn_sweep.py examples/foo.glb out/ --iters 20000

Writes winner's build to <out>/ and sweep.json with per-config scores."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from itertools import product
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

DEFAULT_GRID = {
    "snot_boost": [0.0, 2.0, 5.0],
    "tile_boost": [0.0, 2.0, 5.0],
    "style": [1.0],
    "snot_rate": [0.0, 0.3],
}


def _run(mesh: Path, out: Path, iters: int, seed: int,
         snot_boost: float, tile_boost: float, style: float,
         snot_rate: float) -> dict:
    cmd = [sys.executable, str(ROOT / "tools" / "sampler_demo.py"),
           str(mesh), str(out),
           "--iters", str(iters), "--seed", str(seed),
           "--snot-boost", str(snot_boost), "--tile-boost", str(tile_boost),
           "--style", str(style)]
    if snot_rate > 0:
        cmd += ["--snot-decorate", "--snot-rate", str(snot_rate)]
    else:
        cmd += ["--no-snot-decorate"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if r.returncode != 0 or not (out / "model.ldr").exists():
        return {"ok": False, "err": r.stderr[-300:]}
    s = subprocess.run([sys.executable, str(ROOT / "tools" / "learn" / "realism_gnn.py"),
                        "score", str(out / "model.ldr"), "--samples", "10", "--subgraph-nodes", "50"],
                        capture_output=True, text=True, timeout=180)
    try:
        realism = float(s.stdout.strip().split("=")[1].split()[0])
    except Exception:
        realism = -1.0
    info = {}
    try:
        info = json.loads((out / "info.json").read_text())
    except Exception:
        pass
    return {"ok": True, "realism": realism, "parts": info.get("parts"),
             "shapes": info.get("shapes", {})}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grid", type=Path, help="optional JSON grid overrides")
    ap.add_argument("--keep-all", action="store_true")
    args = ap.parse_args()

    grid = dict(DEFAULT_GRID)
    if args.grid:
        grid.update(json.loads(args.grid.read_text()))

    args.out.mkdir(parents=True, exist_ok=True)
    scratch = args.out / "_sweep"
    scratch.mkdir(exist_ok=True)

    configs = list(product(grid["snot_boost"], grid["tile_boost"], grid["style"], grid["snot_rate"]))
    print(f"gnn-sweep: {len(configs)} configs on {args.mesh.name}", flush=True)
    results = []
    for i, (sb, tb, st, sr) in enumerate(configs):
        tag = f"sb{sb:g}_tb{tb:g}_st{st:g}_sr{sr:g}"
        od = scratch / tag
        if od.exists():
            shutil.rmtree(od)
        print(f"  [{i+1:2d}/{len(configs)}] {tag}...", flush=True)
        r = _run(args.mesh, od, args.iters, args.seed, sb, tb, st, sr)
        r["cfg"] = {"snot_boost": sb, "tile_boost": tb, "style": st, "snot_rate": sr}
        print(f"      -> realism={r.get('realism', 'n/a')!r:>7}  parts={r.get('parts', 'n/a')}", flush=True)
        results.append(r)

    ok = [r for r in results if r["ok"] and r["realism"] >= 0]
    if not ok:
        raise SystemExit("no configs succeeded")
    winner = max(ok, key=lambda r: r["realism"])
    tag = f"sb{winner['cfg']['snot_boost']:g}_tb{winner['cfg']['tile_boost']:g}_st{winner['cfg']['style']:g}_sr{winner['cfg']['snot_rate']:g}"
    src = scratch / tag
    for f in ("model.ldr", "info.json", "render.png"):
        if (src / f).exists():
            shutil.copy(src / f, args.out / f)
    (args.out / "sweep.json").write_text(json.dumps(
        {"winner_cfg": winner["cfg"], "winner_realism": winner["realism"],
         "configs": [{"cfg": r["cfg"], "realism": r.get("realism"), "parts": r.get("parts")}
                     for r in results]}, indent=1))
    print(f"\nwinner: {tag}  realism={winner['realism']:.3f}  parts={winner['parts']}", flush=True)
    print(f"wrote -> {args.out}", flush=True)
    if not args.keep_all:
        shutil.rmtree(scratch)


if __name__ == "__main__":
    main()
