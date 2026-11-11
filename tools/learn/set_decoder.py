"""DETR-style set-prediction decoder for LEGO placement.

Encoder: 3D UNet (reused from cnn_decoder) over the voxel target.
Decoder: N learned object queries attending (cross-attn) to the encoder
features.  Each query predicts (part_token, y, x, z, yaw) in parallel —
no autoregression, no cell ordering.  Training uses bipartite matching
between the N predictions and the ground-truth placements via the
Hungarian algorithm.  Unmatched queries are supervised to predict a
"no placement" index.
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
from scipy.optimize import linear_sum_assignment
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools" / "learn"))

from everything2lego.sampler import vocabulary  # noqa: E402
from cnn_decoder import _UNet3D, _tokens_and_kinds, KNOWN_KINDS, DATASET_DIR  # noqa: E402


MODEL_PATH = Path("/Volumes/PRO-G40/lego_data/analysis/set_decoder.pt")


class SetDataset(Dataset):
    """One sample = one OMR model: (voxels, pos [L,3], tok [L], yaw [L], kind)."""

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
        yaw = d["opt_yaw"].astype(np.int64)
        kind = str(d["kind"])
        kind_i = self.kind_idx.get(kind, len(self.kinds))
        tok_ids = np.array([self.tok_idx.get(str(l), len(self.tokens)) for l in ldraws], np.int64)
        yaw_lab = (yaw // 90) & 3
        v = torch.from_numpy(voxels)[None, None]
        v = F.interpolate(v, size=(self.grid, self.grid, self.grid), mode="trilinear", align_corners=False)
        v = (v.squeeze(0).squeeze(0) > 0.3).float()
        y_s = (y.astype(np.float32) / max(1, voxels.shape[0])) * self.grid
        x_s = (x.astype(np.float32) / max(1, voxels.shape[1])) * self.grid
        z_s = (z.astype(np.float32) / max(1, voxels.shape[2])) * self.grid
        pos = np.stack([y_s, x_s, z_s], axis=1).clip(0, self.grid - 1).astype(np.int64)
        if len(tok_ids) > self.max_placements:
            idx = np.random.permutation(len(tok_ids))[:self.max_placements]
            pos = pos[idx]; tok_ids = tok_ids[idx]; yaw_lab = yaw_lab[idx]
        return v, torch.from_numpy(pos), torch.from_numpy(tok_ids), torch.from_numpy(yaw_lab), kind_i


def _collate(batch):
    voxels, poss, toks, yaws, kinds = zip(*batch)
    v = torch.stack(voxels, 0)
    L = max(len(t) for t in toks)
    pos_pad = torch.zeros(len(batch), L, 3, dtype=torch.long)
    tok_pad = torch.full((len(batch), L), -1, dtype=torch.long)
    yaw_pad = torch.full((len(batch), L), -1, dtype=torch.long)
    for i, (p, t, y) in enumerate(zip(poss, toks, yaws)):
        pos_pad[i, :len(p)] = p
        tok_pad[i, :len(t)] = t
        yaw_pad[i, :len(y)] = y
    return v, pos_pad, tok_pad, yaw_pad, torch.tensor(kinds, dtype=torch.long)


class SetModel(nn.Module):
    """UNet encoder + Transformer decoder with N learned queries."""

    def __init__(self, n_tokens: int, n_kinds: int, grid: int = 16, ch: int = 32,
                 d_model: int = 128, n_queries: int = 200, n_dec_layers: int = 2, nhead: int = 4):
        super().__init__()
        self.grid = grid
        self.ch = ch
        self.n_tokens = n_tokens
        self.n_queries = n_queries
        self.no_placement_idx = n_tokens
        self.encoder = _UNet3D(ch=ch)
        self.proj_mem = nn.Linear(ch, d_model)
        self.cell_pos = nn.Embedding(grid * grid * grid, d_model)
        self.kind_emb = nn.Embedding(n_kinds + 1, d_model)
        self.queries = nn.Parameter(torch.randn(n_queries, d_model) * 0.02)
        layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead,
                                             dim_feedforward=d_model * 2,
                                             batch_first=True, dropout=0.0)
        self.decoder = nn.TransformerDecoder(layer, num_layers=n_dec_layers)
        self.head_part = nn.Linear(d_model, n_tokens + 1)
        self.head_py = nn.Linear(d_model, grid)
        self.head_px = nn.Linear(d_model, grid)
        self.head_pz = nn.Linear(d_model, grid)
        self.head_yaw = nn.Linear(d_model, 4)

    def forward(self, voxels, kind):
        B = voxels.size(0)
        f = self.encoder(voxels.unsqueeze(1))
        _, C, G, _, _ = f.shape
        mem = f.view(B, C, -1).permute(0, 2, 1)
        mem = self.proj_mem(mem)
        pos_ids = torch.arange(G * G * G, device=voxels.device)
        mem = mem + self.cell_pos(pos_ids)[None]
        k_emb = self.kind_emb(kind)[:, None, :]
        mem = mem + k_emb
        q = self.queries[None].expand(B, -1, -1)
        out = self.decoder(q, mem)
        return (self.head_part(out), self.head_py(out), self.head_px(out),
                self.head_pz(out), self.head_yaw(out))


def _hungarian_assign(part_logits_b, py_b, px_b, pz_b, tok_b, pos_b):
    T = tok_b.size(0)
    if T == 0:
        return np.array([], np.int64), np.array([], np.int64)
    log_p = F.log_softmax(part_logits_b, dim=-1)
    part_cost = -log_p[:, tok_b]
    y_prob = F.softmax(py_b, dim=-1)
    x_prob = F.softmax(px_b, dim=-1)
    z_prob = F.softmax(pz_b, dim=-1)
    G = y_prob.size(-1)
    axis = torch.arange(G, device=py_b.device, dtype=torch.float32)
    y_hat = (y_prob * axis).sum(-1)
    x_hat = (x_prob * axis).sum(-1)
    z_hat = (z_prob * axis).sum(-1)
    pos_cost = ((y_hat[:, None] - pos_b[:, 0].float()[None]).abs()
                + (x_hat[:, None] - pos_b[:, 1].float()[None]).abs()
                + (z_hat[:, None] - pos_b[:, 2].float()[None]).abs())
    cost = part_cost + 0.5 * pos_cost
    cost_np = cost.detach().cpu().numpy()
    q_idx, g_idx = linear_sum_assignment(cost_np)
    return q_idx.astype(np.int64), g_idx.astype(np.int64)


def _loss_and_metrics(outs, pos, tok, yaw, no_placement_idx):
    part_logits, py, px, pz, yaw_logits = outs
    B, N = part_logits.shape[:2]
    device = part_logits.device
    part_tgt = torch.full((B, N), no_placement_idx, dtype=torch.long, device=device)
    py_tgt = torch.full((B, N), -1, dtype=torch.long, device=device)
    px_tgt = torch.full((B, N), -1, dtype=torch.long, device=device)
    pz_tgt = torch.full((B, N), -1, dtype=torch.long, device=device)
    yaw_tgt = torch.full((B, N), -1, dtype=torch.long, device=device)
    for b in range(B):
        valid = tok[b] >= 0
        if not valid.any():
            continue
        tok_b = tok[b, valid]
        pos_b = pos[b, valid]
        yaw_b = yaw[b, valid]
        q_idx, g_idx = _hungarian_assign(part_logits[b], py[b], px[b], pz[b], tok_b, pos_b)
        if len(q_idx) == 0:
            continue
        qt = torch.from_numpy(q_idx).to(device)
        gt = torch.from_numpy(g_idx).to(device)
        part_tgt[b, qt] = tok_b[gt]
        py_tgt[b, qt] = pos_b[gt, 0]
        px_tgt[b, qt] = pos_b[gt, 1]
        pz_tgt[b, qt] = pos_b[gt, 2]
        yaw_tgt[b, qt] = yaw_b[gt]
    loss_part = F.cross_entropy(part_logits.reshape(-1, part_logits.size(-1)), part_tgt.reshape(-1))
    loss_py = F.cross_entropy(py.reshape(-1, py.size(-1)), py_tgt.reshape(-1), ignore_index=-1)
    loss_px = F.cross_entropy(px.reshape(-1, px.size(-1)), px_tgt.reshape(-1), ignore_index=-1)
    loss_pz = F.cross_entropy(pz.reshape(-1, pz.size(-1)), pz_tgt.reshape(-1), ignore_index=-1)
    loss_yaw = F.cross_entropy(yaw_logits.reshape(-1, yaw_logits.size(-1)), yaw_tgt.reshape(-1), ignore_index=-1)
    loss = loss_part + loss_py + loss_px + loss_pz + loss_yaw
    with torch.no_grad():
        matched = part_tgt != no_placement_idx
        part_correct = ((part_logits.argmax(-1) == part_tgt) & matched).sum().item()
        n_match = matched.sum().item()
    return (loss, loss_part.item(), loss_py.item(), loss_px.item(), loss_pz.item(), loss_yaw.item(),
            part_correct, n_match)


def _save_ckpt(m, ds, grid, n_queries, epoch):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": m.state_dict(), "n_tokens": len(ds.tokens), "n_kinds": len(ds.kinds),
                "tokens": ds.tokens, "kinds": ds.kinds, "grid": grid, "n_queries": n_queries,
                "ch": 32, "epoch": epoch}, MODEL_PATH)


def train(epochs: int = 10, batch: int = 4, lr: float = 1e-4, grid: int = 16, n_queries: int = 200,
          limit: int = 0):
    ds = SetDataset(grid=grid)
    if limit:
        ds.files = ds.files[:limit]
    n = len(ds)
    tr = int(n * 0.9)
    train_set = torch.utils.data.Subset(ds, range(tr))
    val_set = torch.utils.data.Subset(ds, range(tr, n))
    print(f"dataset: {n} models ({tr} train / {n - tr} val), grid={grid}, "
          f"tokens={len(ds.tokens)}, kinds={len(ds.kinds)}, n_queries={n_queries}", flush=True)
    dl_tr = DataLoader(train_set, batch_size=batch, shuffle=True, collate_fn=_collate, num_workers=0)
    dl_v = DataLoader(val_set, batch_size=batch, shuffle=False, collate_fn=_collate, num_workers=0)
    device = "cpu"
    m = SetModel(len(ds.tokens), len(ds.kinds), grid=grid, n_queries=n_queries).to(device)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"model params: {n_params/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    for ep in range(epochs):
        m.train()
        t0 = time.time()
        tot_loss = tot_lp = tot_lpy = tot_lpx = tot_lpz = tot_lyaw = 0.0
        n_batches = tot_correct = tot_matched = 0
        for v, pos, tok, yaw, kind in dl_tr:
            v, pos, tok, yaw, kind = v.to(device), pos.to(device), tok.to(device), yaw.to(device), kind.to(device)
            outs = m(v, kind)
            loss, lp, lpy, lpx, lpz, lyaw, correct, matched = _loss_and_metrics(
                outs, pos, tok, yaw, m.no_placement_idx)
            opt.zero_grad(); loss.backward(); opt.step()
            tot_loss += loss.item(); tot_lp += lp; tot_lpy += lpy; tot_lpx += lpx; tot_lpz += lpz; tot_lyaw += lyaw
            tot_correct += correct; tot_matched += matched; n_batches += 1
        tr_loss = tot_loss / max(1, n_batches)
        tr_part_top1 = tot_correct / max(1, tot_matched)
        m.eval()
        v_tot = v_lp = v_lpy = v_lpx = v_lpz = v_lyaw = 0.0
        v_n = v_correct = v_matched = 0
        with torch.no_grad():
            for v, pos, tok, yaw, kind in dl_v:
                v, pos, tok, yaw, kind = v.to(device), pos.to(device), tok.to(device), yaw.to(device), kind.to(device)
                outs = m(v, kind)
                loss, lp, lpy, lpx, lpz, lyaw, correct, matched = _loss_and_metrics(
                    outs, pos, tok, yaw, m.no_placement_idx)
                v_tot += loss.item(); v_lp += lp; v_lpy += lpy; v_lpx += lpx; v_lpz += lpz; v_lyaw += lyaw
                v_correct += correct; v_matched += matched; v_n += 1
        val_loss = v_tot / max(1, v_n)
        val_part_top1 = v_correct / max(1, v_matched)
        print(f"epoch {ep+1}: train {tr_loss:.3f} (part {tot_lp/n_batches:.3f} pos {(tot_lpy+tot_lpx+tot_lpz)/n_batches/3:.3f} "
              f"yaw {tot_lyaw/n_batches:.3f}) part-top1 {tr_part_top1:.3f}  |  "
              f"val {val_loss:.3f} (part {v_lp/v_n:.3f} pos {(v_lpy+v_lpx+v_lpz)/v_n/3:.3f} yaw {v_lyaw/v_n:.3f}) "
              f"part-top1 {val_part_top1:.3f}  ({time.time()-t0:.1f}s)", flush=True)
        _save_ckpt(m, ds, grid, n_queries, ep + 1)
    print(f"-> {MODEL_PATH}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("train",))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grid", type=int, default=16)
    ap.add_argument("--n-queries", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    train(epochs=a.epochs, batch=a.batch, lr=a.lr, grid=a.grid, n_queries=a.n_queries, limit=a.limit)
