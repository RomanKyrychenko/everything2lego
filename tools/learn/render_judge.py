"""Render judge: is this WHOLE build a real set, from its rendered views (silhouette, orientation, overall look) —
what V14's 50-brick pieces cannot see.  ResNet-18 (ImageNet weights) on each of the 5 views of a render sheet
(render_build.py default: lon 0,45,90,180,270, 700 px tiles, 3 columns), view features mean-pooled -> real / not.

Data (all the same render spec):
  pos  vis_cls3/pos/*.png                          1635 real OMR sets
  neg  vis_cls4/damaged_renders/*.png              damaged real sets (297, older render)
       render_judge/neg/*.png                      rendered here: the other damaged sets + our generated builds
  Watermill builds and BrickNet generator outputs are kept out (evaluation targets).  Val split by set (v14_build.is_val).

    .venv-arm/bin/python tools/learn/render_judge.py render [--workers 4]
    .venv-arm/bin/python tools/learn/render_judge.py train [--epochs 6]
    .venv-arm/bin/python tools/learn/render_judge.py score build.ldr|sheet.png ...
"""
from __future__ import annotations

import argparse
import random
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(HERE.parent / "set_study"), str(ROOT / "src")]
VC = Path("/Volumes/PRO-G40/lego_data/analysis/vis_classifiers")
OUT = VC / "render_judge"
TILE, COLS, NVIEW, RES = 700, 3, 5, 224
MODEL_PT = OUT / "model.pt"


def _render(args) -> str:
    ldr, png = args
    if png.exists(): return "skip"
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "render_build.py"), str(ldr), str(png)], capture_output=True, text=True)
    return "ok" if png.exists() else f"fail {ldr.name}: {r.stderr[-200:]}"


def render(workers: int) -> None:
    from v14_build import damaged_files, generated_files
    done = {p.stem for p in (VC / "vis_cls4" / "damaged_renders").glob("*.png")}
    jobs = [(f, OUT / "neg" / f"damaged__{f.stem}.png") for f in damaged_files() if f.stem not in done]
    jobs += [(f, OUT / "neg" / ("gen__" + "__".join(f.relative_to(VC.parent).with_suffix("").parts) + ".png")) for f in generated_files()]
    (OUT / "neg").mkdir(parents=True, exist_ok=True); t0 = time.time(); n = 0
    print(f"{len(jobs)} renders", flush=True)
    with ProcessPoolExecutor(workers) as ex:
        for k, r in enumerate(ex.map(_render, jobs), 1):
            n += r == "ok"
            if r.startswith("fail"): print(r, flush=True)
            if k % 50 == 0: print(f"  {k}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
    print(f"rendered {n}", flush=True)


def views(png: Path) -> np.ndarray:
    """(NVIEW, RES, RES, 3) uint8 tiles of a render sheet."""
    from PIL import Image
    im = Image.open(png).convert("RGB"); s = im.width // COLS
    return np.stack([np.asarray(im.crop(((v % COLS) * s, (v // COLS) * s, (v % COLS + 1) * s, (v // COLS + 1) * s)).resize((RES, RES), Image.BILINEAR))
                     for v in range(NVIEW)])


def dataset() -> list[tuple[Path, int, bool]]:
    from v14_build import is_val, set_key
    rows = [(p, 1, is_val(set_key(p))) for p in sorted((VC / "vis_cls3" / "pos").glob("*.png"))]
    rows += [(p, 0, is_val(set_key(p))) for p in sorted((VC / "vis_cls4" / "damaged_renders").glob("*.png"))]
    for p in sorted((OUT / "neg").glob("*.png")):
        key = set_key(Path(p.stem.removeprefix("damaged__"))) if p.stem.startswith("damaged__") else p.stem
        rows.append((p, 0, is_val(key)))
    return [r for r in rows if not r[0].name.startswith("._")]


def cache(rows) -> np.ndarray:
    """All tiles, (N, NVIEW, RES, RES, 3) uint8, cached on disk (decoding 2100x1400 sheets every epoch is the bottleneck)."""
    f = OUT / "tiles.npy"; keys = OUT / "tiles_keys.txt"; names = "\n".join(str(p) for p, _, _ in rows)
    if f.exists() and keys.exists() and keys.read_text() == names: return np.load(f, mmap_mode="r")
    X = np.lib.format.open_memmap(f, mode="w+", dtype=np.uint8, shape=(len(rows), NVIEW, RES, RES, 3))
    for i, (p, _, _) in enumerate(rows):
        X[i] = views(p)
        if i % 200 == 0: print(f"  cached {i}/{len(rows)}", flush=True)
    X.flush(); keys.write_text(names); return np.load(f, mmap_mode="r")


def make_model():
    import torch.nn as nn
    import torchvision
    class Judge(nn.Module):
        def __init__(self):
            super().__init__()
            m = torchvision.models.resnet18(); m.load_state_dict(__import__("torch").load(Path.home() / ".cache/torch/hub/checkpoints/resnet18-f37072fd.pth"))
            m.fc = nn.Identity(); self.backbone = m; self.head = nn.Linear(512, 2)
        def forward(self, x):                                                    # x (B, V, 3, H, W) normalised
            B, V = x.shape[:2]
            return self.head(self.backbone(x.flatten(0, 1)).view(B, V, -1).mean(1))
    return Judge()


def to_tensor(x, aug=False, rng=None):
    import torch
    t = torch.from_numpy(np.ascontiguousarray(x)).float().div(255).permute(0, 1, 4, 2, 3)        # (B,V,3,H,W)
    if aug:                                                                     # mirror + brightness: not shape cues
        if rng.random() < 0.5: t = t.flip(-1)
        t = (t * (0.85 + 0.3 * rng.random())).clamp(0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
    return (t - mean) / std


def train(epochs: int, batch: int, threads: int) -> None:
    import torch
    from sklearn.metrics import roc_auc_score
    torch.set_num_threads(threads); torch.manual_seed(0); rng = random.Random(0)
    rows = dataset(); X = cache(rows); y = np.array([r[1] for r in rows]); val = np.array([r[2] for r in rows])
    tr, va = np.flatnonzero(~val), np.flatnonzero(val)
    print(f"train {len(tr)} ({y[tr].sum()} real / {(1 - y[tr]).sum()} not)  val {len(va)} ({y[va].sum()} / {(1 - y[va]).sum()})", flush=True)
    m = make_model(); opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=1e-4)
    w = torch.tensor([len(tr) / (2 * (1 - y[tr]).sum()), len(tr) / (2 * y[tr].sum())], dtype=torch.float32)   # class balance
    crit = torch.nn.CrossEntropyLoss(weight=w); best = -1.0
    for ep in range(1, epochs + 1):
        m.train(); order = rng.sample(list(tr), len(tr)); t0 = time.time(); tl = 0.0
        for i in range(0, len(order), batch):
            ix = sorted(order[i:i + batch]); xb = to_tensor(X[ix], True, rng); yb = torch.from_numpy(y[ix])
            loss = crit(m(xb), yb); opt.zero_grad(); loss.backward(); opt.step(); tl += loss.item() * len(ix)
        m.eval(); p = []
        with torch.no_grad():
            for i in range(0, len(va), batch): p += torch.softmax(m(to_tensor(X[va[i:i + batch]])), -1)[:, 1].tolist()
        p = np.array(p); auc = roc_auc_score(y[va], p)
        print(f"ep {ep}: loss {tl / len(tr):.4f}  val AUC {auc:.3f}  p(real) real {p[y[va] == 1].mean():.3f} / not {p[y[va] == 0].mean():.3f}  ({time.time() - t0:.0f}s)", flush=True)
        if auc > best: best = auc; torch.save({"state_dict": m.state_dict(), "val_auc": auc, "epoch": ep}, MODEL_PT)
    print(f"best val AUC {best:.3f} -> {MODEL_PT}", flush=True)


def score(paths: list[Path]) -> None:
    import tempfile
    import torch
    m = make_model(); m.load_state_dict(torch.load(MODEL_PT)["state_dict"]); m.eval()
    for p in paths:
        png = p
        if p.suffix != ".png":
            png = Path(tempfile.mkdtemp()) / "sheet.png"; _render((p, png))
        with torch.no_grad(): s = torch.softmax(m(to_tensor(views(png)[None])), -1)[0, 1].item()
        print(f"{p}  render judge {s:.3f}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=("render", "train", "score")); ap.add_argument("paths", type=Path, nargs="*")
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--epochs", type=int, default=6); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    if a.cmd == "render": render(a.workers)
    elif a.cmd == "train": train(a.epochs, a.batch, a.threads)
    else: score(a.paths)


if __name__ == "__main__":
    main()
