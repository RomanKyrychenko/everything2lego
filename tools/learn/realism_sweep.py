"""Sweep sampler hyperparameters, score each with the realism model,
keep the best.  This is the honest reward integration: --realism-best-of
only varies seed and the realism signal is too polarized within one
sampler configuration to pick a winner; varying knobs (snot-boost,
tile-boost, style) actually moves realism and the sweep picks the
highest.

    python tools/learn/realism_sweep.py examples/foo.glb out/ --iters 20000

Writes the winning config's build to <out>/ and a summary of all tried
configs to <out>/sweep.json."""
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
}


def _run(mesh: Path, out: Path, iters: int, seed: int,
         snot_boost: float, tile_boost: float, style: float) -> dict:
    cmd = [sys.executable, str(ROOT / "tools" / "sampler_demo.py"),
           str(mesh), str(out),
           "--iters", str(iters), "--seed", str(seed),
           "--snot-boost", str(snot_boost), "--tile-boost", str(tile_boost),
           "--style", str(style)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
    if r.returncode != 0 or not (out / "model.ldr").exists():
        return {"ok": False, "err": r.stderr[-300:]}
    # Score
    s = subprocess.run([sys.executable, str(ROOT / "tools" / "learn" / "realism.py"),
                        "score", str(out / "model.ldr")],
                        capture_output=True, text=True, timeout=120)
    try:
        realism = float(s.stdout.strip().split("=")[-1])
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
    ap.add_argument("--keep-all", action="store_true", help="keep every config's output dir (default: only winner)")
    args = ap.parse_args()

    grid = DEFAULT_GRID
    if args.grid:
        grid = {**DEFAULT_GRID, **json.loads(args.grid.read_text())}

    args.out.mkdir(parents=True, exist_ok=True)
    scratch = args.out / "_sweep"
    scratch.mkdir(exist_ok=True)

    configs = list(product(grid["snot_boost"], grid["tile_boost"], grid["style"]))
    print(f"sweep: {len(configs)} configs on {args.mesh.name}", flush=True)
    results = []
    for i, (sb, tb, st) in enumerate(configs):
        tag = f"sb{sb:g}_tb{tb:g}_st{st:g}"
        od = scratch / tag
        if od.exists():
            shutil.rmtree(od)
        print(f"  [{i+1:2d}/{len(configs)}] {tag}...", flush=True)
        r = _run(args.mesh, od, args.iters, args.seed, sb, tb, st)
        r["cfg"] = {"snot_boost": sb, "tile_boost": tb, "style": st}
        print(f"      -> realism={r.get('realism', 'n/a')!r:>8}  parts={r.get('parts', 'n/a')}", flush=True)
        results.append(r)

    ok = [r for r in results if r["ok"] and r["realism"] >= 0]
    if not ok:
        raise SystemExit("no configs succeeded")
    winner = max(ok, key=lambda r: r["realism"])
    tag = f"sb{winner['cfg']['snot_boost']:g}_tb{winner['cfg']['tile_boost']:g}_st{winner['cfg']['style']:g}"
    src = scratch / tag
    # Copy winner's outputs to args.out
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
