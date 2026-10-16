"""3D CNN encoder + autoregressive placement decoder trained on the
OMR dataset (tools/learn/omr_dataset.py).  First architecture that
actually sees the WHOLE voxel target instead of a 5x6x6 window.

Model:
  encoder: Conv3d(1->16, 3) -> ReLU -> Conv3d(16->32, 3) -> ReLU
           -> Conv3d(32->32, 3) -> ReLU -> embedding per (y,x,z) cell
           (fixed voxel resolution GRID=16, interpolate target in)
  decoder: for each placement in build order, concat
             - encoder[y, x, z]  (what the target looks like here)
             - previous_kindgrid[y, x, z]  (what we've built so far,
               embedded via a small Linear(n_kinds+1, 16))
             - kind_embedding (object kind one-hot -> Linear(k, 16))
           -> Linear -> ReLU -> Linear -> logits over tokens
  loss:    cross-entropy on the ground-truth token, teacher-forced

    python tools/learn/cnn_decoder.py train
        [--epochs 5] [--batch 4] [--lr 1e-3] [--grid 16]
    python tools/learn/cnn_decoder.py eval  [--seed 0]

Trained weights are saved to $LEGO_DATA/analysis/cnn_decoder.pt
(not into src/everything2lego/data/ since the current sampler engines
don't use torch — this is a parallel diagnostic path, like net_build).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from everything2lego.sampler import vocabulary  # noqa: E402
from everything2lego.style import kind_of  # noqa: E402

DATASET_DIR = Path("/Volumes/PRO-G40/lego_data/analysis/omr_dataset")
MODEL_PATH = Path("/Volumes/PRO-G40/lego_data/analysis/cnn_decoder.pt")
KNOWN_KINDS = ("building", "road_vehicle", "creature", "object", "figure", "scenery", "mechanism", "train")


def _tokens_and_kinds():
    """Build a stable token list (vocabulary ldraw names, in a fixed order)
    and the kind list.  The decoder predicts an index into tokens + an
    `other` catch-all."""
    vocab = vocabulary()
    tokens = sorted({v.part.ldraw for v in vocab})
    kinds = list(KNOWN_KINDS)
    return tokens, kinds


class OMRDataset(Dataset):
    """One sample = one OMR model.  Returns a target voxel grid resized to
    (grid, grid, grid) and the placement sequence (positions + token ids +
    kind id)."""

    def __init__(self, grid: int = 16, max_placements: int = 300):
        self.grid = grid
        self.max_placements = max_placements
        self.tokens, self.kinds = _tokens_and_kinds()
        self.tok_idx = {t: i for i, t in enumerate(self.tokens)}
        self.kind_idx = {k: i for i, k in enumerate(self.kinds)}
        self.files = sorted(f for f in DATASET_DIR.glob("*.npz") if not f.name.startswith("._"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = np.load(self.files[i], allow_pickle=True)
        voxels = d["voxels"].astype(np.float32)
        y, x, z = d["y"], d["x"], d["z"]
        ldraws = d["opt_ldraw"]
        kind = str(d["kind"])
        kind_i = self.kind_idx.get(kind, len(self.kinds))
        tok_ids = np.array([self.tok_idx.get(str(l), len(self.tokens)) for l in ldraws], np.int64)
        # resize voxel grid to (grid, grid, grid) via torch interpolate
        v = torch.from_numpy(voxels)[None, None]  # (1, 1, H, X, Z)
        v = F.interpolate(v, size=(self.grid, self.grid, self.grid), mode="trilinear", align_corners=False)
        v = (v.squeeze(0).squeeze(0) > 0.3).float()                    # (grid, grid, grid)
        # scale placement positions to grid
        y_s = (y.astype(np.float32) / max(1, voxels.shape[0])) * self.grid
        x_s = (x.astype(np.float32) / max(1, voxels.shape[1])) * self.grid
        z_s = (z.astype(np.float32) / max(1, voxels.shape[2])) * self.grid
        pos = np.stack([y_s, x_s, z_s], axis=1).clip(0, self.grid - 1).astype(np.int64)
        # cap sequence length
        if len(tok_ids) > self.max_placements:
            idx = np.random.permutation(len(tok_ids))[:self.max_placements]
            pos = pos[idx]; tok_ids = tok_ids[idx]
        # sort bottom-up for teacher forcing
        order = np.lexsort((pos[:, 2], pos[:, 1], pos[:, 0]))
        pos = pos[order]; tok_ids = tok_ids[order]
        return v, torch.from_numpy(pos), torch.from_numpy(tok_ids), kind_i


def _collate(batch):
    """Pad placement sequences to the longest in the batch."""
    voxels, poss, toks, kinds = zip(*batch)
    v = torch.stack(voxels, 0)                                        # (B, G, G, G)
    L = max(len(t) for t in toks)
    pos_pad = torch.zeros(len(batch), L, 3, dtype=torch.long)
    tok_pad = torch.full((len(batch), L), -1, dtype=torch.long)       # -1 = ignored
    for i, (p, t) in enumerate(zip(poss, toks)):
        pos_pad[i, :len(p)] = p
        tok_pad[i, :len(t)] = t
    return v, pos_pad, tok_pad, torch.tensor(kinds, dtype=torch.long)


class Model(nn.Module):
    """3D CNN encoder + sequence decoder.

    Encoder: 3 Conv3d blocks + 1 residual block for 128-ch features at
    full grid resolution.  Decoder: transformer-style self-attention
    over the placement sequence, cross-attending to the voxel feature
    at each placement's own cell via gather + concat.  Shapes are the
    same as the earlier small model for easy comparison; only internals
    change."""

    def __init__(self, n_tokens: int, n_kinds: int, grid: int = 16, ch: int = 128,
                 d_head: int = 128, n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.grid = grid
        self.ch = ch
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv3d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv3d(64, ch, 3, padding=1), nn.ReLU(),
            nn.Conv3d(ch, ch, 3, padding=1), nn.ReLU(),              # extra depth
        )
        self.kind_emb = nn.Embedding(n_kinds + 1, 32)
        self.state_emb = nn.Embedding(n_tokens + 2, 32)
        self.pos_emb = nn.Embedding(grid, 16)                        # per-axis positional embedding
        self.n_tokens = n_tokens
        d_in = ch + 32 + 32 + 48                                      # feat + kind + state + 3*pos
        self.proj = nn.Linear(d_in, d_head)
        dec_layer = nn.TransformerEncoderLayer(d_model=d_head, nhead=n_heads,
                                                 dim_feedforward=d_head * 2,
                                                 batch_first=True, dropout=0.1)
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=n_layers)
        self.head = nn.Linear(d_head, n_tokens + 1)

    def forward(self, voxels, pos, prev_tok, kind):
        f = self.encoder(voxels.unsqueeze(1))                         # (B, ch, G, G, G)
        B, ch, G, _, _ = f.shape
        y, x, z = pos[..., 0], pos[..., 1], pos[..., 2]
        flat = y * G * G + x * G + z
        f_flat = f.view(B, ch, -1).permute(0, 2, 1)
        gather = f_flat.gather(1, flat.unsqueeze(-1).expand(-1, -1, ch))
        k_emb = self.kind_emb(kind)[:, None, :].expand(-1, pos.size(1), -1)
        s_emb = self.state_emb(prev_tok.clamp(0, self.n_tokens + 1))
        p_emb = torch.cat([self.pos_emb(y.clamp(0, G - 1)),
                            self.pos_emb(x.clamp(0, G - 1)),
                            self.pos_emb(z.clamp(0, G - 1))], dim=-1)
        h = torch.cat([gather, k_emb, s_emb, p_emb], dim=-1)
        h = self.proj(h)
        # causal attention over the placement sequence
        L = pos.size(1)
        mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=h.device), diagonal=1)
        h = self.decoder(h, mask=mask)
        return self.head(h)


def train(epochs: int = 5, batch: int = 4, lr: float = 1e-3, grid: int = 16, limit: int = 0):
    ds = OMRDataset(grid=grid)
    if limit:
        ds.files = ds.files[:limit]
    n = len(ds)
    tr = int(n * 0.9)
    train_set = torch.utils.data.Subset(ds, range(tr))
    val_set = torch.utils.data.Subset(ds, range(tr, n))
    print(f"dataset: {n} models ({tr} train / {n - tr} val), grid={grid}, "
          f"tokens={len(ds.tokens)}, kinds={len(ds.kinds)}", flush=True)
    dl_tr = DataLoader(train_set, batch_size=batch, shuffle=True, collate_fn=_collate, num_workers=0)
    dl_v = DataLoader(val_set, batch_size=batch, shuffle=False, collate_fn=_collate, num_workers=0)
    # Conv3D isn't supported on MPS (torch 2.2); stay on CPU.
    device = "cpu"
    print(f"device: {device} (Conv3D not supported on MPS)", flush=True)
    m = Model(len(ds.tokens), len(ds.kinds), grid=grid).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    for ep in range(epochs):
        m.train()
        t0 = time.time()
        total = n_tokens = 0.0
        for v, pos, tok, kind in dl_tr:
            v, pos, tok, kind = v.to(device), pos.to(device), tok.to(device), kind.to(device)
            # teacher-forced previous token: shift right, use "empty" (n_tokens+1) at step 0
            prev = torch.full_like(tok, len(ds.tokens) + 1)
            prev[:, 1:] = tok[:, :-1].clamp(0, len(ds.tokens))
            logits = m(v, pos, prev, kind)                            # (B, L, n_tokens+1)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tok.view(-1), ignore_index=-1)
            opt.zero_grad(); loss.backward(); opt.step()
            n_valid = (tok != -1).sum().item()
            total += loss.item() * n_valid; n_tokens += n_valid
        tr_nll = total / max(1, n_tokens)
        m.eval()
        with torch.no_grad():
            total = n_tokens = correct = top5 = 0.0
            for v, pos, tok, kind in dl_v:
                v, pos, tok, kind = v.to(device), pos.to(device), tok.to(device), kind.to(device)
                prev = torch.full_like(tok, len(ds.tokens) + 1)
                prev[:, 1:] = tok[:, :-1].clamp(0, len(ds.tokens))
                logits = m(v, pos, prev, kind)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tok.view(-1), ignore_index=-1)
                mask = tok != -1
                nv = mask.sum().item()
                total += loss.item() * nv; n_tokens += nv
                top1 = logits.argmax(-1) == tok
                correct += (top1 & mask).sum().item()
                top5p = logits.topk(5, dim=-1).indices == tok.unsqueeze(-1)
                top5 += (top5p.any(-1) & mask).sum().item()
        val_nll = total / max(1, n_tokens)
        print(f"epoch {ep+1}: train nll {tr_nll:.3f}  val nll {val_nll:.3f}  "
              f"top-1 {correct/max(1,n_tokens):.3f}  top-5 {top5/max(1,n_tokens):.3f}  "
              f"({time.time()-t0:.1f}s)", flush=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": m.state_dict(), "n_tokens": len(ds.tokens), "n_kinds": len(ds.kinds),
                "tokens": ds.tokens, "kinds": ds.kinds, "grid": grid}, MODEL_PATH)
    print(f"-> {MODEL_PATH}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("train", "eval"))
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--grid", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="use only first N dataset files (quick test)")
    a = ap.parse_args()
    if a.cmd == "train":
        train(epochs=a.epochs, batch=a.batch, lr=a.lr, grid=a.grid, limit=a.limit)
    else:
        print("eval not implemented yet; load the model and run inference directly")
