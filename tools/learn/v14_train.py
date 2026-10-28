"""V14 judge: V11's colour-blind GNN over ALL parts.

Node input = learned part embedding (parts placed >= 5x in real sets; 15 % dropped to "other" in training so
rare / unseen parts lean on features) + Rebrickable category + size/connector features (any part) + position.
Data: v14_build.py arrays, streamed.  Each epoch: every real piece, half as many spoiled, half as many hard
negatives (damaged sets and our generated builds, 50/50).  Model selection on validation AUC (held-out sets).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from v11_train import MP                                                     # noqa: E402
from v14_parts import FEAT_DIM, N_CATEGORY, PartTable                        # noqa: E402

DATA = Path("/Volumes/PRO-G40/lego_data/analysis/vis_classifiers/v14_data")
MODEL_PT = Path("/Volumes/PRO-G40/lego_data/analysis/vis_classifiers/v14_model.pt")
KINDS = ("real", "spoiled", "damaged", "generated")


class V14GNN(nn.Module):
    def __init__(self, n_emb: int, hidden=96, n_layers=4, dropout=0.1, id_drop=0.15):
        super().__init__()
        self.part_emb = nn.Embedding(n_emb, 32); self.cat_emb = nn.Embedding(N_CATEGORY, 8); self.id_drop = id_drop
        self.embed = nn.Linear(32 + 8 + FEAT_DIM + 6, hidden)
        self.layers = nn.ModuleList([MP(hidden, dropout=dropout) for _ in range(n_layers)])
        self.head = nn.Sequential(nn.Linear(hidden, 48), nn.ReLU(), nn.Dropout(dropout), nn.Linear(48, 16), nn.ReLU(), nn.Linear(16, 2))
        self.table: PartTable | None = None                                    # set by load_v14 for scoring

    def forward(self, ei, ea, nc, ncont, graph_id, n_graphs):
        """nc (N,2) [emb id, category]; ncont (N, 6 + FEAT_DIM) [position/size, part features]."""
        ids = nc[:, 0]
        if self.training and self.id_drop > 0:
            ids = torch.where(torch.rand(ids.shape, device=ids.device) < self.id_drop, torch.zeros_like(ids), ids)
        x = torch.relu(self.embed(torch.cat([self.part_emb(ids), self.cat_emb(nc[:, 1]), ncont], -1)))
        for layer in self.layers:
            x = layer(x, ei, ea)
        sums = torch.zeros(n_graphs, x.size(-1), device=x.device).index_add_(0, graph_id, x)
        counts = torch.zeros(n_graphs, device=x.device).scatter_add_(0, graph_id, torch.ones(graph_id.size(0), device=x.device))
        return self.head(sums / counts.clamp(min=1).unsqueeze(-1))

    def encode(self, names: list, cont: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        ids, f = self.table.encode(names)
        return torch.from_numpy(ids), torch.from_numpy(np.concatenate([cont.astype(np.float32), f], 1))


def load_v14(path: Path | str = MODEL_PT) -> V14GNN:
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    m = V14GNN(ck["n_emb"], ck["hidden"], ck["n_layers"], ck["dropout"]); m.load_state_dict(ck["state_dict"]); m.eval()
    m.table = PartTable(); m.table.emb = ck["emb"]                             # frozen ids from training
    return m


class Data:
    def __init__(self, d: Path, table: PartTable):
        r = lambda k, dt, shape=None: np.memmap(d / k, dtype=dt, mode="r") if shape is None else np.memmap(d / k, dtype=dt, mode="r").reshape(shape)  # noqa: E731
        self.name = r("node_name.i32", np.int32); self.cont = r("node_cont.f16", np.float16, (-1, 6))
        self.ei = r("edge_ei.i16", np.int16, (-1, 2)); self.ea = r("edge_ea.f16", np.float16, (-1, 8))
        self.gn = np.fromfile(d / "g_node.i64", np.int64); self.ge = np.fromfile(d / "g_edge.i64", np.int64)
        self.y = np.fromfile(d / "g_y.i8", np.int8); self.kind = np.fromfile(d / "g_kind.i8", np.int8); self.val = np.fromfile(d / "g_val.i8", np.int8)
        names = json.loads((d / "names.json").read_text()); ids, f = table.encode(names); table.save()
        self.n_ids = torch.from_numpy(ids); self.n_feat = torch.from_numpy(f)

    def batch(self, gs):
        nc, cont, ei, ea, gid, off = [], [], [], [], [], 0
        for k, g in enumerate(gs):
            a, b = self.gn[g], self.gn[g + 1]; e0, e1 = self.ge[g], self.ge[g + 1]
            nm = torch.from_numpy(self.name[a:b].astype(np.int64))
            nc.append(self.n_ids[nm]); cont.append(torch.cat([torch.from_numpy(self.cont[a:b].astype(np.float32)), self.n_feat[nm]], 1))
            ei.append(torch.from_numpy(self.ei[e0:e1].astype(np.int64)).T + off); ea.append(torch.from_numpy(self.ea[e0:e1].astype(np.float32)))
            gid.append(torch.full((b - a,), k, dtype=torch.long)); off += b - a
        return (torch.cat(ei, 1), torch.cat(ea), torch.cat(nc), torch.cat(cont), torch.cat(gid), len(gs),
                torch.from_numpy(self.y[gs].astype(np.int64)))


def epoch_ids(D: Data, rng) -> np.ndarray:
    tr = D.val == 0; by = {k: np.flatnonzero(tr & (D.kind == k)) for k in range(4)}; n = len(by[0])
    hard = [rng.choice(by[k], n // 4, replace=len(by[k]) < n // 4) for k in (2, 3) if len(by[k])]
    ids = np.concatenate([by[0], rng.choice(by[1], n // 2, replace=len(by[1]) < n // 2), *hard])
    return rng.permutation(ids)


def to(b, dev):
    return [z.to(dev) if torch.is_tensor(z) else z for z in b]


@torch.no_grad()
def evaluate(model, D: Data, batch=512, dev="cpu") -> dict:
    from sklearn.metrics import roc_auc_score
    model.eval(); ids = np.flatnonzero(D.val == 1); p = np.zeros(len(ids))
    for i in range(0, len(ids), batch):
        b = to(D.batch(ids[i:i + batch]), dev); p[i:i + batch] = torch.softmax(model(*b[:6]), -1)[:, 1].cpu().numpy()
    y, k = D.y[ids], D.kind[ids]; out = {"auc": float(roc_auc_score(y, p)), "n": len(ids)}
    for j in (1, 2, 3):                                                        # real vs each negative kind
        m = (k == 0) | (k == j)
        if (k == j).any(): out[f"auc_{KINDS[j]}"] = float(roc_auc_score(y[m], p[m])); out[f"p_{KINDS[j]}"] = float(p[k == j].mean())
    out["p_real"] = float(p[k == 0].mean())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA); ap.add_argument("--out", type=Path, default=MODEL_PT)
    ap.add_argument("--epochs", type=int, default=10); ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch", type=int, default=512); ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu"); ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=0, help="smoke test: stop each epoch after this many steps")
    a = ap.parse_args(); torch.set_num_threads(a.threads); torch.manual_seed(0); rng = np.random.default_rng(0)
    table = PartTable(); D = Data(a.data, table)
    kc = [int(((D.val == 0) & (D.kind == k)).sum()) for k in range(4)]; kv = [int(((D.val == 1) & (D.kind == k)).sum()) for k in range(4)]
    print(f"train {dict(zip(KINDS, kc))}  val {dict(zip(KINDS, kv))}  parts: {table.n_emb - 1} embedded", flush=True)
    model = V14GNN(table.n_emb).to(a.device); print(f"V14GNN {sum(p.numel() for p in model.parameters()) / 1e3:.1f}k params", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    crit = nn.CrossEntropyLoss(); best = -1.0
    for ep in range(1, a.epochs + 1):
        model.train(); ids = epoch_ids(D, rng); t0 = time.time(); tl = th = tn = 0
        for step, i in enumerate(range(0, len(ids), a.batch)):
            if a.max_steps and step >= a.max_steps: break
            *x, y = to(D.batch(ids[i:i + a.batch]), a.device); logits = model(*x); loss = crit(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item() * len(y); th += (logits.argmax(-1) == y).sum().item(); tn += len(y)
            if step % 200 == 0 and step:
                print(f"  ep {ep} step {step}/{len(ids) // a.batch}  loss {tl / tn:.4f} acc {th / tn:.3f}  ({time.time() - t0:.0f}s)", flush=True)
        sched.step(); v = evaluate(model, D, dev=a.device)
        print(f"ep {ep:02d}: train loss {tl / tn:.4f} acc {th / tn:.3f} | val " + "  ".join(f"{k} {x:.3f}" for k, x in v.items() if k != "n") + f"  ({time.time() - t0:.0f}s)", flush=True)
        if v["auc"] > best:
            best = v["auc"]
            torch.save({"state_dict": {k: t.cpu() for k, t in model.state_dict().items()}, "arch": "v14", "n_emb": table.n_emb, "hidden": 96, "n_layers": 4, "dropout": 0.1,
                        "emb": table.emb, "colour_agnostic": True, "val": v, "epoch": ep}, a.out)
    print(f"best val AUC {best:.3f} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
