"""Score a built .ldr for realism using the trained real-vs-ours
classifier (discriminator_train.py).

    python tools/learn/discriminator_score.py path/to/model.ldr

Prints P(real) averaged over the three default views (30, 120, 210).
Values near 1.0 look like real OMR sets to the classifier; near 0.0
look like our sampler outputs.  Mid values are "classifier can't tell"
— either a very good build or the model has distribution shift from
what it saw in training."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = Path(os.environ.get("LEGO_DATA", "/Volumes/PRO-G40/lego_data"))
MODEL_PATH = DATA / "analysis" / "discriminator" / "model.pt"
VIEWS = (30, 120, 210)
SIZE = 128


def _render(ldr: Path, out_png: Path, view: int) -> bool:
    if out_png.exists():
        return True
    cmd = [sys.executable, str(ROOT / "tools" / "render_build.py"),
           str(ldr), str(out_png), "--views", str(view), "--size", str(SIZE)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0 and out_png.exists()


def score(ldr: Path, model_path: Path = MODEL_PATH, device: str = "cpu") -> float:
    """P(real) averaged over the three default views."""
    tfm = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    net = models.resnet18(weights=None)
    net.fc = nn.Linear(net.fc.in_features, 2)
    ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    net.load_state_dict(ckpt["state_dict"])
    net.eval().to(device)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        xs = []
        for v in VIEWS:
            p = tmp / f"v{v}.png"
            if not _render(ldr, p, v):
                continue
            xs.append(tfm(Image.open(p).convert("RGB")))
        if not xs:
            raise SystemExit(f"could not render any view of {ldr}")
        batch = torch.stack(xs).to(device)
        with torch.no_grad():
            logits = net(batch)
            probs = F.softmax(logits, dim=-1)[:, 1]            # P(real)
        return float(probs.mean().item())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ldr", type=Path)
    ap.add_argument("--model", type=Path, default=MODEL_PATH)
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    args = ap.parse_args()
    p = score(args.ldr, args.model, args.device)
    print(f"{args.ldr}: realism = {p:.3f}")


if __name__ == "__main__":
    main()
