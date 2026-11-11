"""Train a ResNet-18 real-vs-ours classifier on the dataset built by
discriminator_dataset.py.

    python tools/learn/discriminator_train.py --epochs 20

Expects $LEGO_DATA/analysis/discriminator/{real,ours}/*.png.  Splits by
MODEL identity (filename prefix before "_v{view}") so held-out
validation never sees any of the training set's views.  Reports held-
out accuracy and per-class confidence.  Saves the trained weights to
$LEGO_DATA/analysis/discriminator/model.pt."""
from __future__ import annotations

import argparse
import os
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

DATA = Path(os.environ.get("LEGO_DATA", "/Volumes/PRO-G40/lego_data"))
ROOT = DATA / "analysis" / "discriminator"


def _group_by_model(files: list[Path]) -> dict[str, list[Path]]:
    """Images sharing the base name before `_v<view>.png` belong to one
    model.  Keeps all views together under one key."""
    out: dict[str, list[Path]] = defaultdict(list)
    for f in files:
        stem = f.stem
        i = stem.rfind("_v")
        key = stem[:i] if i > 0 else stem
        out[key].append(f)
    return out


class SplitDataset(Dataset):
    """One image per item; label 1 = real, 0 = ours."""
    def __init__(self, items: list[tuple[Path, int]], tfm):
        self.items = items
        self.tfm = tfm

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        img = Image.open(p).convert("RGB")
        return self.tfm(img), y


def _pngs(d: Path) -> list[Path]:
    """Real .png files, skipping macOS AppleDouble sidecars (._*.png)."""
    return [p for p in sorted(d.glob("*.png")) if not p.name.startswith("._")]


def _build_splits(real_dir: Path, ours_dir: Path, val_frac: float, seed: int):
    real_models = _group_by_model(_pngs(real_dir))
    ours_models = _group_by_model(_pngs(ours_dir))
    rnd = random.Random(seed)
    rk = list(real_models); rnd.shuffle(rk)
    ok = list(ours_models); rnd.shuffle(ok)
    rv = max(1, int(len(rk) * val_frac))
    ov = max(1, int(len(ok) * val_frac))
    val_r, train_r = rk[:rv], rk[rv:]
    val_o, train_o = ok[:ov], ok[ov:]
    train = ([(p, 1) for k in train_r for p in real_models[k]]
             + [(p, 0) for k in train_o for p in ours_models[k]])
    val = ([(p, 1) for k in val_r for p in real_models[k]]
           + [(p, 0) for k in val_o for p in ours_models[k]])
    rnd.shuffle(train); rnd.shuffle(val)
    return train, val, (len(train_r), len(train_o), len(val_r), len(val_o))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    ap.add_argument("--out", type=Path, default=ROOT / "model.pt")
    args = ap.parse_args()

    real_dir = ROOT / "real"
    ours_dir = ROOT / "ours"
    tfm = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    train_items, val_items, (ntr_r, ntr_o, nval_r, nval_o) = _build_splits(
        real_dir, ours_dir, args.val_frac, args.seed)
    print(f"models: train real={ntr_r} ours={ntr_o}, val real={nval_r} ours={nval_o}", flush=True)
    print(f"images: train={len(train_items)}, val={len(val_items)}", flush=True)
    if not train_items or not val_items:
        raise SystemExit("empty split; need more data")

    train_dl = DataLoader(SplitDataset(train_items, tfm), batch_size=args.bs, shuffle=True)
    val_dl = DataLoader(SplitDataset(val_items, tfm), batch_size=args.bs, shuffle=False)

    device = torch.device(args.device)
    net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = nn.Linear(net.fc.in_features, 2)
    net = net.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    best_val = -1.0
    for ep in range(1, args.epochs + 1):
        net.train()
        tr_loss = tr_n = tr_hit = 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            logits = net(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += loss.item() * x.size(0)
            tr_hit += (logits.argmax(-1) == y).sum().item()
            tr_n += x.size(0)

        net.eval()
        vl_hit = vl_n = 0
        vl_real_hit = vl_real_n = vl_ours_hit = vl_ours_n = 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                logits = net(x)
                pred = logits.argmax(-1)
                vl_hit += (pred == y).sum().item()
                vl_n += x.size(0)
                for yi, pi in zip(y.cpu().tolist(), pred.cpu().tolist()):
                    if yi == 1:
                        vl_real_n += 1
                        vl_real_hit += int(pi == 1)
                    else:
                        vl_ours_n += 1
                        vl_ours_hit += int(pi == 0)
        va = vl_hit / max(1, vl_n)
        ta = tr_hit / max(1, tr_n)
        print(f"ep {ep:02d}  train loss {tr_loss/max(1,tr_n):.4f} acc {ta:.3f}  "
              f"val acc {va:.3f}  (real {vl_real_hit}/{vl_real_n}, ours {vl_ours_hit}/{vl_ours_n})",
              flush=True)
        if va > best_val:
            best_val = va
            torch.save({"state_dict": net.state_dict(), "val_acc": va, "epoch": ep}, args.out)
    print(f"best val acc: {best_val:.3f}  saved to {args.out}", flush=True)


if __name__ == "__main__":
    main()
