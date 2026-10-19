"""Alternating classifier↔sampler training.  Each round:

1. Run realism_sweep on each example mesh → K ours builds per round
   (including the previous classifier's "best" that gamed it).
2. Append those builds' features to the X_ours bucket in features.npz.
3. Retrain the classifier.
4. Score a reference baseline + the new sweep winner, log the gap.

Converges when the gap stops closing — either the classifier can't be
improved further or the sampler can't find any config that passes.

    python tools/learn/realism_loop.py --rounds 5 --iters 15000

Writes per-round scores to $LEGO_DATA/analysis/realism/loop.json."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = Path(os.environ.get("LEGO_DATA", "/Volumes/PRO-G40/lego_data"))
OUT = DATA / "analysis" / "realism"
FEATURES_NPZ = OUT / "features.npz"
LOOP_JSON = OUT / "loop.json"


def _meshes() -> list[Path]:
    return sorted((ROOT / "examples").glob("*.glb"))


def _append_ours_features(new_paths: list[Path]) -> int:
    """Extract features from new ours builds, append to X_ours in features.npz."""
    sys.path.insert(0, str(ROOT / "tools" / "learn"))
    sys.path.insert(0, str(ROOT / "tools" / "set_study"))
    from realism import extract
    feats = []
    for p in new_paths:
        try:
            v = extract(p)
            if v is not None:
                feats.append(v)
        except Exception:
            pass
    if not feats:
        return 0
    feats = np.stack(feats)
    d = dict(np.load(FEATURES_NPZ))
    if "X_ours" in d:
        d["X_ours"] = np.concatenate([d["X_ours"], feats])
    else:
        d["X_ours"] = feats
    np.savez_compressed(FEATURES_NPZ, **d)
    return len(feats)


def _run_sweep(mesh: Path, out: Path, iters: int) -> list[Path]:
    """Run realism_sweep with --keep-all; return the per-config .ldr paths."""
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    cmd = [sys.executable, str(ROOT / "tools" / "learn" / "realism_sweep.py"),
           str(mesh), str(out), "--iters", str(iters), "--keep-all"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        print(f"sweep failed on {mesh.name}: {r.stderr[-200:]}", flush=True)
        return []
    return sorted((out / "_sweep").glob("*/model.ldr"))


def _score(ldr: Path) -> float:
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "learn" / "realism.py"),
                        "score", str(ldr)], capture_output=True, text=True, timeout=120)
    try:
        return float(r.stdout.strip().split("=")[-1])
    except Exception:
        return -1.0


def _run_baseline(mesh: Path, out: Path, iters: int, seed: int) -> Path | None:
    """Vanilla sampler run, used as the reference baseline each round."""
    if out.exists():
        shutil.rmtree(out)
    cmd = [sys.executable, str(ROOT / "tools" / "sampler_demo.py"),
           str(mesh), str(out), "--iters", str(iters), "--seed", str(seed),
           "--no-snot-decorate"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
    if r.returncode != 0 or not (out / "model.ldr").exists():
        return None
    return out / "model.ldr"


def _retrain(epochs: int) -> None:
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "learn" / "realism.py"),
                        "train", "--epochs", str(epochs), "--device", "cpu"],
                        capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        print(f"retrain failed: {r.stderr[-300:]}", flush=True)
    else:
        # last line has "best val acc X saved to ..."
        print(f"  retrain: {r.stdout.strip().splitlines()[-1]}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--iters", type=int, default=15000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()

    meshes = _meshes()
    if not meshes:
        raise SystemExit("no meshes in examples/")

    scratch = OUT / "_loop_scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    log: list[dict] = []

    for rd in range(1, args.rounds + 1):
        print(f"\n=== round {rd}/{args.rounds} ===", flush=True)
        # 1. Baseline score on each mesh with current classifier
        baselines = {}
        for mesh in meshes:
            od = scratch / f"r{rd}_baseline_{mesh.stem}"
            ldr = _run_baseline(mesh, od, args.iters, args.seed)
            baselines[mesh.stem] = _score(ldr) if ldr else -1.0
            print(f"  baseline {mesh.stem}: realism={baselines[mesh.stem]:.3f}", flush=True)

        # 2. Sweep each mesh, pick highest-realism winner, note winner score
        winners = {}
        new_ldrs: list[Path] = []
        for mesh in meshes:
            od = scratch / f"r{rd}_sweep_{mesh.stem}"
            ldrs = _run_sweep(mesh, od, args.iters)
            new_ldrs.extend(ldrs)
            scores = [(l, _score(l)) for l in ldrs]
            best_l, best_s = max(scores, key=lambda t: t[1]) if scores else (None, -1.0)
            winners[mesh.stem] = best_s
            print(f"  sweep winner {mesh.stem}: realism={best_s:.3f}  (from {len(scores)} configs)", flush=True)

        # 3. Append ALL sweep outputs (not just the winner) to X_ours and retrain
        added = _append_ours_features(new_ldrs)
        print(f"  appended {added} new ours features", flush=True)
        _retrain(args.epochs)

        # 4. Re-score baselines + winners under the new classifier to measure drop
        drop_baseline = {}
        drop_winner = {}
        for mesh in meshes:
            bld = scratch / f"r{rd}_baseline_{mesh.stem}" / "model.ldr"
            if bld.exists():
                drop_baseline[mesh.stem] = _score(bld)
            sw_dir = scratch / f"r{rd}_sweep_{mesh.stem}" / "_sweep"
            if sw_dir.exists():
                best = max(sw_dir.glob("*/model.ldr"), key=_score, default=None)
                if best:
                    drop_winner[mesh.stem] = _score(best)
        print(f"  post-retrain baseline: {drop_baseline}", flush=True)
        print(f"  post-retrain winner:   {drop_winner}", flush=True)

        log.append({"round": rd, "baseline_before": baselines, "winner_before": winners,
                     "baseline_after": drop_baseline, "winner_after": drop_winner,
                     "added_ours": added})
        LOOP_JSON.write_text(json.dumps(log, indent=1))

    print(f"\nfinal log -> {LOOP_JSON}", flush=True)


if __name__ == "__main__":
    main()
