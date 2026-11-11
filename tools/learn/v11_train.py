"""V11 training — 500k pos + 500k neg balanced, AUC-based selection.

Architecture: V10 (bigger colour-agnostic GNN — hidden 96, 4 layers,
dropout 0.1).  Training changes vs V10:
- Balanced dataset means NO class weighting (weights 1,1).
- Primary metric: ROC AUC on validation (sklearn).  Save best by AUC
  (threshold-independent).  Also report acc + F1.
- 20 epochs, cosine LR.
- Still colour-agnostic (no col_emb).
"""
from __future__ import annotations

import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from realism import CATEGORIES                                             # noqa: E402
from realism_gnn import PART_VOCAB_SIZE                                    # noqa: E402

EDGE_DIM = 8
NODE_EMB_DIM = 32 + 4 + 4 + 6
HIDDEN = 96
N_LAYERS = 4
EPOCHS = 20
DROPOUT = 0.1

DATA_PATH = Path("/Volumes/PRO-G40/lego_data/analysis/vis_classifiers/v11_dataset.pkl")
MODEL_PT = Path("/Volumes/PRO-G40/lego_data/analysis/vis_classifiers/v11_model.pt")


class MP(nn.Module):
    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.msg = nn.Linear(dim + EDGE_DIM, dim)
        self.upd = nn.Linear(dim * 2, dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, ei, ea):
        src = ei[0]; dst = ei[1]
        m = torch.relu(self.msg(torch.cat([x[src], ea], dim=-1)))
        m = self.drop(m)
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, m)
        deg = torch.zeros(x.size(0), device=x.device).scatter_add_(
            0, dst, torch.ones(ei.size(1), device=x.device, dtype=torch.float32))
        agg = agg / deg.clamp(min=1).unsqueeze(-1)
        return torch.relu(self.upd(torch.cat([x, agg], dim=-1)))


class V11GNN(nn.Module):
    """Same arch as V10GNN.  Shipping as V11 so the colour-agnostic
    checkpoint flag distinguishes it from V8/V9."""

    def __init__(self, hidden=HIDDEN, n_layers=N_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.part_emb = nn.Embedding(PART_VOCAB_SIZE, 32)
        self.cat_emb = nn.Embedding(len(CATEGORIES), 4)
        self.size_emb = nn.Embedding(5, 4)
        self.embed = nn.Linear(NODE_EMB_DIM, hidden)
        self.layers = nn.ModuleList([MP(hidden, dropout=dropout) for _ in range(n_layers)])
        self.head = nn.Sequential(nn.Linear(hidden, 48), nn.ReLU(),
                                   nn.Dropout(dropout), nn.Linear(48, 16),
                                   nn.ReLU(), nn.Linear(16, 2))

    def forward(self, ei, ea, nc, ncont, graph_id, n_graphs):
        part_e = self.part_emb(nc[:, 0])
        cat_e = self.cat_emb(nc[:, 1])
        size_e = self.size_emb(nc[:, 2])
        x = torch.cat([part_e, cat_e, size_e, ncont], dim=-1)
        x = torch.relu(self.embed(x))
        for layer in self.layers:
            x = layer(x, ei, ea)
        sums = torch.zeros(n_graphs, x.size(-1), device=x.device)
        sums.index_add_(0, graph_id, x)
        counts = torch.zeros(n_graphs, device=x.device).scatter_add_(
            0, graph_id, torch.ones(graph_id.size(0), device=x.device))
        pooled = sums / counts.clamp(min=1).unsqueeze(-1)
        return self.head(pooled)


def iter_batches(items, batch=128, shuffle=True):
    order = list(range(len(items)))
    if shuffle:
        rng2 = np.random.default_rng()
        order = rng2.permutation(len(items))
    for i in range(0, len(order), batch):
        batch_ix = order[i:i + batch]
        all_cat, all_cont, all_ei, all_ea, all_gid, all_y = [], [], [], [], [], []
        n_off = 0; kept = 0
        for j in batch_ix:
            d, y = items[j]
            if d["ei"].shape[1] == 0 or d["cat"].shape[0] == 0:
                continue
            n = d["cat"].shape[0]
            all_cat.append(torch.tensor(d["cat"], dtype=torch.long))
            all_cont.append(torch.tensor(d["cont"]))
            all_ei.append(torch.tensor(d["ei"], dtype=torch.long) + n_off)
            all_ea.append(torch.tensor(d["ea"]))
            all_gid.append(torch.full((n,), kept, dtype=torch.long))
            all_y.append(y)
            n_off += n; kept += 1
        if kept == 0:
            continue
        yield (torch.cat(all_ei, dim=1), torch.cat(all_ea, dim=0),
               torch.cat(all_cat, dim=0), torch.cat(all_cont, dim=0),
               torch.cat(all_gid, dim=0), kept,
               torch.tensor(all_y, dtype=torch.long))


if __name__ == "__main__":
    print(f"loading {DATA_PATH} ({DATA_PATH.stat().st_size/1e9:.1f} GB)...", flush=True)
    t0 = time.time()
    with DATA_PATH.open("rb") as f:
        data = pickle.load(f)
    print(f"loaded {len(data)} in {time.time()-t0:.0f}s", flush=True)

    rng = random.Random(42)
    rng.shuffle(data)
    split = int(len(data) * 0.95)
    train_data, val_data = data[:split], data[split:]
    del data
    n_tp = sum(1 for _, y in train_data if y == 1)
    n_tn = len(train_data) - n_tp
    print(f"train {len(train_data)} ({n_tp} pos / {n_tn} neg)  val {len(val_data)}", flush=True)

    model = V11GNN()
    print(f"V11GNN: {sum(p.numel() for p in model.parameters())/1e3:.1f}k params", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit = nn.CrossEntropyLoss()                                           # balanced → no weights

    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        roc_auc_score = None

    best_auc = 0.0
    best_f1 = 0.0
    for ep in range(1, EPOCHS + 1):
        model.train()
        tr_hit = tr_n = tr_loss = 0
        t0 = time.time()
        for step, (ei, ea, nc, ncont, gid, M, y) in enumerate(iter_batches(train_data, batch=128)):
            logits = model(ei, ea, nc, ncont, gid, M)
            loss = crit(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item() * M
            tr_hit += (logits.argmax(-1) == y).sum().item()
            tr_n += M
            if step % 1000 == 0 and step > 0:
                print(f"  step {step}  acc {tr_hit/tr_n:.3f}  loss {tr_loss/tr_n:.4f}  lr {opt.param_groups[0]['lr']:.5f}  ({time.time()-t0:.0f}s)", flush=True)
        sched.step()
        model.eval()
        vl_hit = vl_n = tp = fp = fn = tn = 0
        all_probs: list = []
        all_y: list = []
        with torch.no_grad():
            for ei, ea, nc, ncont, gid, M, y in iter_batches(val_data, batch=128, shuffle=False):
                logits = model(ei, ea, nc, ncont, gid, M)
                probs = torch.softmax(logits, dim=-1)[:, 1]
                pred = logits.argmax(-1)
                vl_hit += (pred == y).sum().item(); vl_n += M
                all_probs.extend(probs.tolist())
                all_y.extend(y.tolist())
                for p, yy in zip(pred.tolist(), y.tolist()):
                    if yy == 1 and p == 1: tp += 1
                    elif yy == 0 and p == 1: fp += 1
                    elif yy == 1 and p == 0: fn += 1
                    else: tn += 1
        va = vl_hit / max(1, vl_n)
        f1 = 2 * tp / max(1, 2 * tp + fp + fn)
        auc = roc_auc_score(all_y, all_probs) if roc_auc_score is not None else -1.0
        print(f"ep {ep:02d}: tr loss {tr_loss/tr_n:.4f} acc {tr_hit/tr_n:.3f}  val acc {va:.3f} F1 {f1:.3f} AUC {auc:.3f}  tp={tp} fn={fn} fp={fp} tn={tn}  ({time.time()-t0:.0f}s)", flush=True)
        if auc > best_auc:
            best_auc = auc; best_f1 = f1
            torch.save({"state_dict": model.state_dict(), "hidden": HIDDEN,
                        "n_layers": N_LAYERS, "dropout": DROPOUT,
                        "colour_agnostic": True, "arch": "v11",
                        "val_auc": auc, "val_f1": f1}, MODEL_PT)

    print(f"best val AUC {best_auc:.3f} F1 {best_f1:.3f}  saved -> {MODEL_PT}", flush=True)
