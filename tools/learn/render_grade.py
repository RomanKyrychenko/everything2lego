"""Graded render judge: a quality SCALE, not real / not-real, so our own candidates can be ranked.
render_judge.py scores every build of ours ~0; here real sets are damaged at known levels and the judge learns
quality q = 1 - level / 0.6 from the renders (level 0 -> 1, 60 % of bricks damaged -> 0).

Damage at level L hits round(L * n) bricks in clustered patches (3-12 nearest bricks, like a bad refiner patch);
each hit brick is deleted, moved off its studs, turned, or swapped for another common part.  Level 0 goes through
the same flatten -> LDR writer, so the clean render differs from the damaged ones only by the damage.
Our generated builds are NOT trained on (they would be pinned to 0 again); they are only scored, to see the spread.

    .venv-arm/bin/python tools/learn/render_grade.py make [--sets 400 --workers 5]     # damaged LDRs + renders
    .venv-arm/bin/python tools/learn/render_grade.py train [--epochs 5]
    .venv-arm/bin/python tools/learn/render_grade.py score build.ldr ...
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "set_study")]
import render_judge as RJ  # noqa: E402

OUT = RJ.VC / "render_grade"
LEVELS = (0.0, 0.03, 0.1, 0.2, 0.35, 0.6)
MODEL_PT = OUT / "model.pt"


def quality(level: float) -> float:
    return 1.0 - level / LEVELS[-1]


def damage(parts: list, level: float, common: list, rng: random.Random) -> list[str]:
    """LDR type-1 lines of `parts` with round(level * n) bricks damaged in clustered patches."""
    n = len(parts); pos = np.array([p.M[:3, 3] for p in parts]); hit: set[int] = set(); k = round(level * n)
    while len(hit) < k:
        seed = rng.randrange(n); d = np.linalg.norm(pos - pos[seed], axis=1); d[list(hit)] = np.inf
        hit |= set(np.argsort(d)[: min(rng.randint(3, 12), k - len(hit))].tolist())
    lines = []
    for i, p in enumerate(parts):
        M, name = p.M.copy(), p.name
        if i in hit:
            op = rng.random()
            if op < 0.3: continue                                                       # delete
            if op < 0.6:                                                                # off its studs
                M[:3, 3] += rng.choice([(20, 0, 0), (-20, 0, 0), (0, 0, 20), (0, 0, -20), (10, 0, 10), (0, -8, 0), (0, -24, 0)])
            elif op < 0.8:                                                              # turned about its own vertical
                c, s = rng.choice([(0, 1), (-1, 0), (0, -1)]); M[:3, :3] = M[:3, :3] @ np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
            else: name = rng.choice(common)                                             # another common part
        lines.append(f"1 {p.colour} " + " ".join(f"{v:.6g}" for v in [*M[:3, 3], *M[:3, :3].ravel()]) + f" {name}")
    return lines


def _make(args) -> int:
    mpd, seed, common = args
    from ldraw_read import flatten
    try: parts = flatten(Path(mpd))
    except Exception: return 0                                                          # noqa: BLE001
    if not 60 <= len(parts) <= 3000: return 0
    rng = random.Random(seed); made = 0
    for lv in LEVELS:
        stem = f"{Path(mpd).stem}__L{int(lv * 100):02d}"; ldr = OUT / "ldr" / f"{stem}.ldr"; png = OUT / "png" / f"{stem}.png"
        if not ldr.exists(): ldr.write_text(f"0 {Path(mpd).stem} graded damage {lv}\n" + "\n".join(damage(parts, lv, common, rng)) + "\n")
        made += RJ._render((ldr, png)) in ("ok", "skip")
    return made


def make(n_sets: int, workers: int) -> None:
    from v14_build import OMR, is_val, set_key
    counts = json.loads((RJ.VC.parent / "realism_gnn" / "omr_part_counts_v2.json").read_text())
    common = [k for k, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:300]]
    files = sorted(OMR.glob("*.mpd")); val = [f for f in files if is_val(set_key(f))]; tr = [f for f in files if not is_val(set_key(f))]
    random.Random(0).shuffle(tr); jobs = val + tr[:n_sets]
    (OUT / "ldr").mkdir(parents=True, exist_ok=True); (OUT / "png").mkdir(parents=True, exist_ok=True); t0 = time.time(); n = 0
    print(f"{len(jobs)} sets ({len(val)} val) x {len(LEVELS)} levels", flush=True)
    with ProcessPoolExecutor(workers) as ex:
        for k, r in enumerate(ex.map(_make, [(str(f), i * 7919 + 3, common) for i, f in enumerate(jobs)]), 1):
            n += r
            if k % 25 == 0: print(f"  {k}/{len(jobs)} sets  {n} renders  {time.time() - t0:.0f}s", flush=True)
    print(f"done: {n} renders", flush=True)


def rows() -> list[tuple[Path, float, bool, str]]:
    from v14_build import is_val, set_key
    out = []
    for p in sorted((OUT / "png").glob("*.png")):
        if p.name.startswith("._"): continue
        s, lv = p.stem.rsplit("__L", 1); out.append((p, quality(int(lv) / 100), is_val(set_key(Path(s))), s))
    return out


def model():
    import torch
    m = RJ.make_model(); m.load_state_dict(torch.load(RJ.MODEL_PT)["state_dict"])               # start from the binary judge
    return m


def predict(m, X, batch: int = 16) -> np.ndarray:
    """Quality in [0, 1] = sigmoid of the judge's real-minus-not logit."""
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            lg = m(RJ.to_tensor(X[i:i + batch])); out += torch.sigmoid(lg[:, 1] - lg[:, 0]).tolist()
    return np.array(out)


def train(epochs: int, batch: int, threads: int) -> None:
    import torch
    from scipy.stats import spearmanr
    torch.set_num_threads(threads); torch.manual_seed(0); rng = random.Random(0)
    R = rows(); X = _cache(R)
    q = np.array([r[1] for r in R], np.float32); val = np.array([r[2] for r in R]); tr, va = np.flatnonzero(~val), np.flatnonzero(val)
    print(f"train {len(tr)} renders ({len({R[i][3] for i in tr})} sets)  val {len(va)} ({len({R[i][3] for i in va})} sets)", flush=True)
    m = model(); opt = torch.optim.AdamW(m.parameters(), lr=5e-5, weight_decay=1e-4); best = -1.0
    for ep in range(1, epochs + 1):
        m.train(); order = rng.sample(list(tr), len(tr)); t0 = time.time(); tl = 0.0
        for i in range(0, len(order), batch):
            ix = sorted(order[i:i + batch]); logits = m(RJ.to_tensor(X[ix], True, rng)); z = logits[:, 1] - logits[:, 0]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(z, torch.from_numpy(q[ix])); opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item() * len(ix)
        m.eval(); p = predict(m, X[va]); rho = spearmanr(p, q[va]).correlation
        per = "  ".join(f"L{int(lv * 100)} {p[np.isclose(q[va], quality(lv))].mean():.2f}" for lv in LEVELS)
        print(f"ep {ep}: loss {tl / len(tr):.4f}  val Spearman {rho:.3f}  | {per}  ({time.time() - t0:.0f}s)", flush=True)
        if rho > best: best = rho; torch.save({"state_dict": m.state_dict(), "val_spearman": float(rho), "epoch": ep}, MODEL_PT)
    print(f"best val Spearman {best:.3f} -> {MODEL_PT}", flush=True)


def _cache(R) -> np.ndarray:
    f = OUT / "tiles.npy"; keys = OUT / "tiles_keys.txt"; names = "\n".join(str(r[0]) for r in R)
    if f.exists() and keys.exists() and keys.read_text() == names: return np.load(f, mmap_mode="r")
    X = np.lib.format.open_memmap(f, mode="w+", dtype=np.uint8, shape=(len(R), RJ.NVIEW, RJ.RES, RJ.RES, 3))
    for i, r in enumerate(R):
        X[i] = RJ.views(r[0])
        if i % 300 == 0: print(f"  cached {i}/{len(R)}", flush=True)
    X.flush(); keys.write_text(names); return np.load(f, mmap_mode="r")


def score(paths: list[Path]) -> None:
    import tempfile
    import torch
    m = RJ.make_model(); m.load_state_dict(torch.load(MODEL_PT, weights_only=False)["state_dict"]); m.eval()   # our own checkpoint
    for p in paths:
        png = p
        if p.suffix != ".png": png = Path(tempfile.mkdtemp()) / "sheet.png"; RJ._render((p, png))
        print(f"{p}  graded quality {predict(m, RJ.views(png)[None])[0]:.3f}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=("make", "train", "score")); ap.add_argument("paths", type=Path, nargs="*")
    ap.add_argument("--sets", type=int, default=400); ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=5); ap.add_argument("--batch", type=int, default=16); ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    if a.cmd == "make": make(a.sets, a.workers)
    elif a.cmd == "train": train(a.epochs, a.batch, a.threads)
    else: score(a.paths)


if __name__ == "__main__":
    main()
