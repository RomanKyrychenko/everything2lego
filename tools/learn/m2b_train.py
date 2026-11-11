"""Mesh2Brick M1 trainer: teacher-forced training on m2b_data shards (docs/mesh2brick_architecture.md §4, §6).

    /Users/romankyrychenko/model_test/.venv/bin/python tools/learn/m2b_train.py --out $LEGO_DATA/analysis/m2b/run1 \
        --train sft --limit 1000 --val val --steps 3000 --batch 8 --device mps

M1 gate: part top-1 > 0.9 teacher-forced on the training 1k (then free-run via m2b_gen.py).
A sample = one model (or a random 256-step window of it), its target grid (mesh fill with p=0.7, else bbox
fill) and the state grid rasterised from the bricks before the window.  Log: <out>/train_log.jsonl."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bricknet_shape import boxes_to_grid  # noqa: E402
from m2b_model import NCONN, N_SLIDE, Mesh2Brick, loss_and_acc  # noqa: E402

DATA = Path(os.environ.get("LEGO_DATA", "/Volumes/PRO-G40/lego_data")); M2B = DATA / "analysis" / "m2b"
WINDOW = 256
COLS = ("part", "color", "parent", "fam", "pconn", "cconn", "yaw", "flip", "slide", "rot", "mats", "boxes")


def load_split(name: str, limit: int = 0, max_err: float = 4.0) -> list[dict]:
    """Models as dicts of per-step arrays + packed grids; rt_err > max_err and overflow filtered out.
    `name` = "sft" | "sft,pt" | "pt:20" (first 20 shards of pt) ..."""
    models = []
    files = []
    for part in name.split(","):
        split, _, k = part.partition(":"); fs = sorted((M2B / split).glob("m2b_*.npz")); files += fs[: int(k)] if k else fs
    for f in files:
        z = np.load(f); ptr = z["ptr"]; cols = {k: z[k] for k in COLS}; grid = tuple(int(v) for v in z["grid"])
        gb, gm, rt, ov, src = z["grid_bbox"], z["grid_mesh"], z["rt_err"], z["overflow"], z["src"]    # read each npz array once
        for m in range(len(ptr) - 1):
            if rt[m] > max_err or ov[m]: continue
            s = slice(ptr[m], ptr[m + 1])
            models.append({**{k: cols[k][s] for k in COLS}, "grid_bbox": gb[m], "grid_mesh": gm[m], "grid": grid, "src": str(src[m])})
            if limit and len(models) >= limit: return models
    return models


class Vocab:
    def __init__(self, path: Path = M2B / "vocab.json"):
        v = json.load(open(path)); self.n_part = len(v["stems"]); self.stems = v["stems"]
        self.n_conn = np.minimum(np.array(v["n_conn"]), NCONN); self.colors = v["colors"]; self.col_idx = {c: i for i, c in enumerate(self.colors)}
        self.conn_fam = v["conn_fam"]


def sample(m: dict, vocab: Vocab, rng: random.Random, p_mesh: float = 0.7, window: int = WINDOW, max_shift: int = 6, infill_p: float = 0.0) -> dict:
    if infill_p and rng.random() < infill_p:
        r = sample_infill(m, vocab, rng, p_mesh, max_shift)
        if r is not None: return r
    H, W, D = m["grid"]; n = len(m["part"]); t0 = 0 if n <= window else rng.randrange(0, n - window + 1); t1 = min(n, t0 + window)
    bx = m["boxes"].astype(np.float32); state = boxes_to_grid(bx[:t0], (H, W, D)) if t0 else np.zeros((H, W, D), bool)
    par = m["parent"][t0:t1].astype(np.int64); par = np.where(par >= 0, par - t0, -1)
    if t0: par[0] = -1                                                             # window root: parent outside the window
    return _fields(m, vocab, rng, np.arange(t0, t1), par, state, np.ones(t1 - t0, bool), p_mesh, max_shift)


def sample_infill(m: dict, vocab: Vocab, rng: random.Random, p_mesh: float = 0.7, max_shift: int = 6, ctx: int = 96,
                  rmin: int = 5, rmax: int = 60) -> dict | None:
    """Hole filling, as m2b_refine poses it: a connected region of the tree (never the root) is removed; the state grid holds
    the remaining bricks; the sequence is up to `ctx` nearby kept bricks (sorted by y, z, x, plus every outside parent of the
    region) then the region in canonical order.  Only region steps (and the STOP after them) are supervised."""
    H, W, D = m["grid"]; n = len(m["part"]); par0 = m["parent"].astype(np.int64)
    if n < rmin + 2: return None
    nb = [[] for _ in range(n)]
    for i in range(1, n):
        if par0[i] >= 0: nb[i].append(int(par0[i])); nb[int(par0[i])].append(i)
    size = rng.randint(rmin, min(rmax, n - 1)); start = rng.randrange(1, n); region, seen, q = [], {start}, [start]
    while q and len(region) < size:
        u = q.pop(0); region.append(u)
        for v in nb[u]:
            if v and v not in seen: seen.add(v); q.append(v)                       # v = 0 is the root: never removed
    if len(region) < rmin: return None
    region = sorted(region); rset = set(region); keep = np.array([i for i in range(n) if i not in rset])
    bx = m["boxes"].astype(np.float32); lo, hi = bx[region, 0].min(0) - 2, bx[region, 1].max(0) + 2
    key = lambda i: (bx[i, 0, 1], bx[i, 0, 2], bx[i, 0, 0])                         # noqa: E731  (m2b_refine's prefix order)
    near = sorted((i for i in keep if (bx[i, 1] > lo).all() and (bx[i, 0] < hi).all()), key=key)[-ctx:]
    need = {int(par0[i]) for i in region if par0[i] not in rset} - set(near)
    pre = sorted(set(near) | need, key=key); idx = np.array(pre + region); pos = {int(o): k for k, o in enumerate(idx)}
    par = np.array([-1] * len(pre) + [pos[int(par0[i])] for i in region], np.int64)
    state = boxes_to_grid(bx[keep], (H, W, D)); sup = np.r_[np.zeros(len(pre), bool), np.ones(len(region), bool)]
    return _fields(m, vocab, rng, idx, par, state, sup, p_mesh, max_shift)


def _fields(m, vocab, rng, idx, par, state, sup, p_mesh, max_shift) -> dict:
    """Training columns for the bricks `idx` (sequence order) with sequence-relative parents `par`; sup = supervised steps."""
    H, W, D = m["grid"]
    tgt = np.unpackbits(m["grid_mesh"] if rng.random() < p_mesh else m["grid_bbox"])[: H * W * D].reshape(H, W, D).astype(bool)
    off = np.array([W / 2, 0, D / 2], np.float32); b = m["boxes"].astype(np.float32)[idx] + off
    if max_shift:                        # whole-cell x/z translation: the root pose must then be read from the volume, not memorised
        occ = np.nonzero(tgt | state); xs, zs = occ[1], occ[2]
        if len(xs):
            dx = rng.randint(max(-max_shift, -int(xs.min())), min(max_shift, W - 1 - int(xs.max())))
            dz = rng.randint(max(-max_shift, -int(zs.min())), min(max_shift, D - 1 - int(zs.max())))
            tgt = np.roll(tgt, (dx, dz), axis=(1, 2)); state = np.roll(state, (dx, dz), axis=(1, 2)); b = b + np.array([dx, 0, dz], np.float32)
    anchor = np.clip(np.floor(b[:, 0] + 1e-3).astype(np.int64), 0, np.array([W - 1, H - 1, D - 1]))
    rot = (m["rot"][idx].astype(np.int64) % 360) // 10
    return {"vol": np.stack([tgt, state]).astype(np.float32), "part": m["part"][idx].astype(np.int64),
            "color": np.array([vocab.col_idx.get(int(c), 0) for c in m["color"][idx]]), "parent": par, "fam": m["fam"][idx].astype(np.int64),
            "pconn": m["pconn"][idx].astype(np.int64), "cconn": m["cconn"][idx].astype(np.int64),
            "yaw": (m["yaw"][idx].astype(np.int64) % 360) // 10, "flip": m["flip"][idx].astype(np.int64),
            "slide": np.clip(m["slide"][idx].astype(np.int64), -20, 20) + 20, "rot": rot,
            "centre": (b[:, 0] + b[:, 1]) / 2, "extent": b[:, 1] - b[:, 0], "R": m["mats"][idx, :3, :3].astype(np.float32), "anchor": anchor,
            "root_cell": (anchor[0][[1, 0, 2]] // 2)[None], "sup": sup}


def collate(items: list[dict], device, bucket: int = 32) -> tuple[dict, torch.Tensor]:
    T = -(-max(len(it["part"]) for it in items) // bucket) * bucket; B = len(items); out = {}   # fixed-size buckets: MPS caches per shape
    pad = torch.ones(B, T, dtype=torch.bool)
    for k in items[0]:
        if k == "vol": out[k] = torch.from_numpy(np.stack([it[k] for it in items])); continue
        a = np.zeros((B, T) + items[0][k].shape[1:], items[0][k].dtype)
        for i, it in enumerate(items): a[i, : len(it[k])] = it[k]; pad[i, : len(it[k])] = False
        out[k] = torch.from_numpy(a)
    return {k: v.to(device) for k, v in out.items()}, pad.to(device)


@torch.no_grad()
def evaluate(model, models, vocab, nconn, device, n: int, batch: int, seed: int = 0, infill_p: float = 0.0) -> dict:
    model.eval(); rng = random.Random(seed); accs = {}; losses = []
    idx = list(range(len(models))); rng.shuffle(idx); idx = idx[:n]
    for i in range(0, len(idx), batch):
        b, pad = collate([sample(models[j], vocab, rng, infill_p=infill_p) for j in idx[i:i + batch]], device)
        out, rc = model(b["vol"], b, pad); loss, acc = loss_and_acc(out, b, pad, rc, nconn, model.grid); losses.append(loss.item())
        for k, v in acc.items():
            if v == v: accs.setdefault(k, []).append(v)
    model.train()
    return {"loss": float(np.mean(losses)), **{k: float(np.mean(v)) for k, v in accs.items()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--train", default="sft"); ap.add_argument("--val", default="val")
    ap.add_argument("--limit", type=int, default=1000, help="train models (0 = all)")
    ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--d", type=int, default=256); ap.add_argument("--layers", type=int, default=6); ap.add_argument("--C", type=int, default=128)
    ap.add_argument("--eval-every", type=int, default=250); ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--device", default="mps"); ap.add_argument("--init", type=Path, default=None); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--infill-p", type=float, default=0.0, help="share of samples that are hole filling (sample_infill)")
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed); rng = random.Random(a.seed); dev = torch.device(a.device)
    vocab = Vocab(); train = load_split(a.train, a.limit); val = load_split(a.val)
    grid = train[0]["grid"]; nconn = torch.from_numpy(vocab.n_conn).to(dev)
    print(f"train {len(train)} models ({a.train}), val {len(val)} ({a.val}), grid {grid}", flush=True)
    model = Mesh2Brick(vocab.n_part, len(vocab.colors), grid, d=a.d, layers=a.layers, C=a.C).to(dev)
    if a.init:
        res = model.load_state_dict(torch.load(a.init, map_location=dev, weights_only=False)["model"], strict=False)
        print(f"init from {a.init}: {len(res.missing_keys)} missing, {len(res.unexpected_keys)} unexpected keys", flush=True)
    print(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))
    log = open(a.out / "train_log.jsonl", "a"); t0 = time.time(); order = []
    use_amp = dev.type in ("cuda", "mps"); scaler = torch.amp.GradScaler(enabled=dev.type == "cuda"); n_skip = 0   # fp16 on cuda needs loss scaling
    for step in range(1, a.steps + 1):
        if len(order) < a.batch: order = list(range(len(train))); rng.shuffle(order)
        b, pad = collate([sample(train[order.pop()], vocab, rng, infill_p=a.infill_p) for _ in range(a.batch)], dev)
        with torch.autocast(dev.type, dtype=torch.float16, enabled=use_amp):
            out, rc = model(b["vol"], b, pad)
        out = {k: v.float() for k, v in out.items()}
        loss, acc = loss_and_acc(out, b, pad, rc, nconn, grid)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt); gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if torch.isfinite(gn): scaler.step(opt)
        else: n_skip += 1
        scaler.update(); sched.step()
        rec = {"step": step, "loss": round(loss.item(), 4), "gn": round(float(gn), 3), "lr": sched.get_last_lr()[0], "s/step": round((time.time() - t0) / step, 3), "skip": n_skip,
               **{f"acc_{k}": round(v, 3) for k, v in acc.items() if v == v}}
        log.write(json.dumps(rec) + "\n"); log.flush()
        if step % 25 == 0: print(json.dumps(rec), flush=True)
        if dev.type == "mps" and step % 10 == 0: torch.mps.empty_cache()
        if step % a.eval_every == 0 or step == a.steps:
            ev_tr = evaluate(model, train, vocab, nconn, dev, 64, a.batch); ev_va = evaluate(model, val, vocab, nconn, dev, 64, a.batch)
            ev_in = evaluate(model, val, vocab, nconn, dev, 64, a.batch, infill_p=1.0) if a.infill_p else {}
            rec = {"step": step, "EVAL": {"train": ev_tr, "val": ev_va, "val_infill": ev_in}}; log.write(json.dumps(rec) + "\n"); log.flush()
            with open(a.out / "evals.jsonl", "a") as ef: ef.write(json.dumps(rec) + "\n")       # closed per write: syncs on Drive mounts
            print(f"EVAL {step}: train part {ev_tr.get('part', 0):.3f} part0 {ev_tr.get('part0', 0):.3f} cell {ev_tr.get('root_cell', 0):.2f} arot {ev_tr.get('arot', 0):.2f} parent {ev_tr.get('parent', 0):.3f} pconn {ev_tr.get('pconn', 0):.3f} cconn {ev_tr.get('cconn', 0):.3f} yaw {ev_tr.get('yaw', 0):.3f} color {ev_tr.get('color', 0):.3f} | "
                  f"val part {ev_va.get('part', 0):.3f} parent {ev_va.get('parent', 0):.3f} pconn {ev_va.get('pconn', 0):.3f} loss {ev_va['loss']:.3f}"
                  + (f" | infill part {ev_in.get('part', 0):.3f} parent {ev_in.get('parent', 0):.3f} pconn {ev_in.get('pconn', 0):.3f} loss {ev_in['loss']:.3f}" if ev_in else ""), flush=True)
        if step % a.save_every == 0 or step == a.steps:
            torch.save({"model": model.state_dict(), "args": vars(a) | {"out": str(a.out), "init": str(a.init)}, "grid": grid, "step": step}, a.out / "model.pt")
    print("done", flush=True)


if __name__ == "__main__":
    main()
