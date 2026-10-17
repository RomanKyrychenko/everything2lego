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
DISTILLED_DIR = Path("/Volumes/PRO-G40/lego_data/analysis/omr_distilled")
MODEL_PATH = Path("/Volumes/PRO-G40/lego_data/analysis/cnn_decoder.pt")
MODEL_PATH_FACTORED = Path("/Volumes/PRO-G40/lego_data/analysis/cnn_decoder_factored.pt")


def _model_path(factored: bool, unet: bool, sched_sampling: float = 0.0,
                ch: int = 32, augment: bool = False, dataset: str = "omr",
                oversample: int = 0) -> Path:
    base = "cnn_decoder"
    if unet:
        base += "_unet"
    if factored:
        base += "_factored"
    if sched_sampling > 0:
        base += f"_sched{int(sched_sampling*100):02d}"
    if ch != 32:
        base += f"_ch{ch}"
    if augment:
        base += "_aug"
    if dataset != "omr":
        base += f"_{dataset}"
    if dataset == "hybrid" and oversample > 0:
        base += f"_x{oversample}"
    return Path(f"/Volumes/PRO-G40/lego_data/analysis/{base}.pt")
KNOWN_KINDS = ("building", "road_vehicle", "creature", "object", "figure", "scenery", "mechanism", "train")


def _tokens_and_kinds():
    """Build a stable token list (vocabulary ldraw names, in a fixed order)
    and the kind list.  The decoder predicts an index into tokens + an
    `other` catch-all."""
    vocab = vocabulary()
    tokens = sorted({v.part.ldraw for v in vocab})
    kinds = list(KNOWN_KINDS)
    return tokens, kinds


def _factorization():
    """Factored label spaces: category (16+unk), footprint (h,w,d) joint
    (observed-sizes + unk), yaw (4).  Returns ordered lists and lookup
    maps:   ldraw -> cat_id,   (h,w,d) -> size_id."""
    vocab = vocabulary()
    cats = sorted({v.category for v in vocab})
    sizes = sorted({tuple(o.size) for v in vocab for o in v.orients})
    cat_of = {v.part.ldraw: cats.index(v.category) for v in vocab}
    size_of = {s: i for i, s in enumerate(sizes)}
    return cats, sizes, cat_of, size_of


class OMRDataset(Dataset):
    """One sample = one OMR model.  Returns a target voxel grid resized to
    (grid, grid, grid) and the placement sequence (positions + token ids +
    kind id).  In factored mode also returns cat_id, size_id, yaw_id per
    placement."""

    def __init__(self, grid: int = 16, max_placements: int = 300, factored: bool = False,
                 augment: bool = False, dataset_dir: Path = DATASET_DIR):
        self.grid = grid
        self.max_placements = max_placements
        self.factored = factored
        self.augment = augment
        self.tokens, self.kinds = _tokens_and_kinds()
        self.tok_idx = {t: i for i, t in enumerate(self.tokens)}
        self.kind_idx = {k: i for i, k in enumerate(self.kinds)}
        if factored:
            self.cats, self.sizes, self.cat_of, self.size_of = _factorization()
        if augment:                                                        # map ldraw -> set of valid yaw rotations
            from everything2lego.sampler import vocabulary
            self.valid_yaws = {v.part.ldraw: frozenset(o.rot for o in v.orients) for v in vocabulary()}
        self.files = sorted(f for f in dataset_dir.glob("*.npz") if not f.name.startswith("._"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = np.load(self.files[i], allow_pickle=True)
        voxels = d["voxels"].astype(np.float32)
        y, x, z = d["y"].astype(np.int64), d["x"].astype(np.int64), d["z"].astype(np.int64)
        ldraws = d["opt_ldraw"]
        kind = str(d["kind"])
        kind_i = self.kind_idx.get(kind, len(self.kinds))
        tok_ids = np.array([self.tok_idx.get(str(l), len(self.tokens)) for l in ldraws], np.int64)
        if self.factored or self.augment:
            w = d["w"].astype(np.int64); h3 = d["h"].astype(np.int64); dd = d["d"].astype(np.int64)
            yaw = d["opt_yaw"].astype(np.int64)                            # degrees: 0/90/180/270
        if self.augment and np.random.random() < 0.5:
            # 180 degree rotation around Y: flip x and z of voxels + placements.
            # Footprint (h,w,d) unchanged; new yaw = yaw+180 if valid for the part,
            # else keep (part is 180-symmetric and the vocab omits the redundant yaw).
            voxels = voxels[:, ::-1, ::-1].copy()
            W_v, D_v = voxels.shape[1], voxels.shape[2]
            x = W_v - x - w
            z = D_v - z - dd
            new_yaw = (yaw + 180) % 360
            for j, l in enumerate(ldraws):
                if new_yaw[j] not in self.valid_yaws.get(str(l), {0}):
                    new_yaw[j] = yaw[j]
            yaw = new_yaw
        if self.factored:
            cat = np.array([self.cat_of.get(str(l), len(self.cats)) for l in ldraws], np.int64)
            sz = np.array([self.size_of.get((int(h3[j]), int(w[j]), int(dd[j])), len(self.sizes))
                            for j in range(len(ldraws))], np.int64)
            yaw_lab = (yaw // 90) & 3                                      # 0/90/180/270 -> 0..3
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
            if self.factored:
                cat = cat[idx]; sz = sz[idx]; yaw_lab = yaw_lab[idx]
        # sort bottom-up for teacher forcing
        order = np.lexsort((pos[:, 2], pos[:, 1], pos[:, 0]))
        pos = pos[order]; tok_ids = tok_ids[order]
        if self.factored:
            cat = cat[order]; sz = sz[order]; yaw_lab = yaw_lab[order]
            return (v, torch.from_numpy(pos), torch.from_numpy(tok_ids), kind_i,
                    torch.from_numpy(cat), torch.from_numpy(sz), torch.from_numpy(yaw_lab))
        return v, torch.from_numpy(pos), torch.from_numpy(tok_ids), kind_i


def _collate(batch):
    """Pad placement sequences to the longest in the batch.  Returns
    chunk_id = Y-layer of each placement (-1 for padding) so the model
    can attend only within the same layer."""
    voxels, poss, toks, kinds = (list(x) for x in zip(*[b[:4] for b in batch]))
    v = torch.stack(voxels, 0)
    L = max(len(t) for t in toks)
    pos_pad = torch.zeros(len(batch), L, 3, dtype=torch.long)
    tok_pad = torch.full((len(batch), L), -1, dtype=torch.long)
    chunk_pad = torch.full((len(batch), L), -1, dtype=torch.long)
    for i, (p, t) in enumerate(zip(poss, toks)):
        pos_pad[i, :len(p)] = p
        tok_pad[i, :len(t)] = t
        chunk_pad[i, :len(p)] = p[:, 0]                               # Y-layer id = chunk id
    if len(batch[0]) == 4:
        return v, pos_pad, tok_pad, torch.tensor(kinds, dtype=torch.long), chunk_pad
    # factored: also pad cat/size/yaw
    cat_pad = torch.full((len(batch), L), -1, dtype=torch.long)
    size_pad = torch.full((len(batch), L), -1, dtype=torch.long)
    yaw_pad = torch.full((len(batch), L), -1, dtype=torch.long)
    for i, b in enumerate(batch):
        c, s, y = b[4], b[5], b[6]
        cat_pad[i, :len(c)] = c
        size_pad[i, :len(s)] = s
        yaw_pad[i, :len(y)] = y
    return (v, pos_pad, tok_pad, torch.tensor(kinds, dtype=torch.long), chunk_pad,
            cat_pad, size_pad, yaw_pad)


class _UNet3D(nn.Module):
    """3D UNet: 16->8->4 encoder, 4->8->16 decoder with skip concats.
    Output has `ch` channels at full input resolution so Model.forward's
    per-cell gather stays unchanged."""

    def __init__(self, ch: int = 32):
        super().__init__()
        c1, c2, c3 = ch, ch * 2, ch * 4
        def blk(ic, oc):
            return nn.Sequential(nn.Conv3d(ic, oc, 3, padding=1), nn.ReLU(),
                                  nn.Conv3d(oc, oc, 3, padding=1), nn.ReLU())
        self.enc1 = blk(1, c1)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = blk(c1, c2)
        self.pool2 = nn.MaxPool3d(2)
        self.bot = blk(c2, c3)
        self.up2 = nn.ConvTranspose3d(c3, c2, 2, stride=2)
        self.dec2 = blk(c3, c2)                                          # cat with e2 doubles input ch
        self.up1 = nn.ConvTranspose3d(c2, c1, 2, stride=2)
        self.dec1 = blk(c2, c1)                                          # cat with e1

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bot(self.pool2(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return d1


class Model(nn.Module):
    """3D CNN encoder + per-placement MLP head.  The MLP sees the voxel
    feature at the placement's cell, plus the previously-placed token
    at that same cell (empty if none), plus kind, plus the per-axis
    positional embeddings.  No cross-placement attention — a
    transformer here was tried and found impractical on CPU for this
    dataset size.  Attention is a next-session item on GPU."""

    def __init__(self, n_tokens: int, n_kinds: int, grid: int = 16, ch: int = 32,
                 d_head: int = 128, chunk_attn: bool = True, factored: bool = False,
                 n_cats: int = 0, n_sizes: int = 0, unet: bool = False):
        """`chunk_attn=True` runs a 1-layer self-attention over placements
        WITHIN the same Y-layer (chunks are typically 10-50 placements, so
        attention stays O(50^2) per layer, CPU-practical).  `factored=True`
        swaps the single n_tokens-way head for three smaller heads over
        (category, footprint, yaw) — denser training signal per class.
        `unet=True` swaps the flat 3-conv encoder for a UNet with 2 down
        steps + skips, so each cell's feature carries whole-shape context."""
        super().__init__()
        self.grid = grid
        self.ch = ch
        self.unet = unet
        if unet:
            self.encoder = _UNet3D(ch=ch)
        else:
            self.encoder = nn.Sequential(
                nn.Conv3d(1, 16, 3, padding=1), nn.ReLU(),
                nn.Conv3d(16, ch, 3, padding=1), nn.ReLU(),
                nn.Conv3d(ch, ch, 3, padding=1), nn.ReLU(),
            )
        self.kind_emb = nn.Embedding(n_kinds + 1, 16)
        self.state_emb = nn.Embedding(n_tokens + 2, 16)
        self.pos_emb = nn.Embedding(grid, 16)
        self.n_tokens = n_tokens
        self.factored = factored
        d_in = ch + 16 + 16 + 48
        self.proj = nn.Linear(d_in, d_head)
        self.chunk_attn = chunk_attn
        if chunk_attn:
            layer = nn.TransformerEncoderLayer(d_model=d_head, nhead=4,
                                                 dim_feedforward=d_head * 2,
                                                 batch_first=True, dropout=0.0)
            self.attn = nn.TransformerEncoder(layer, num_layers=1)
        if factored:
            trunk = nn.Sequential(nn.Linear(d_head, d_head), nn.ReLU())
            self.head_trunk = trunk
            self.head_cat = nn.Linear(d_head, n_cats + 1)
            self.head_size = nn.Linear(d_head, n_sizes + 1)
            self.head_yaw = nn.Linear(d_head, 4)
        else:
            self.head = nn.Sequential(
                nn.Linear(d_head, d_head), nn.ReLU(),
                nn.Linear(d_head, n_tokens + 1),
            )

    def forward(self, voxels, pos, prev_tok, kind, chunk_id=None):
        """chunk_id (B, L): same-id positions attend to each other (within
        the same Y-layer).  When None, falls back to per-cell independence."""
        f = self.encoder(voxels.unsqueeze(1))
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
        if self.chunk_attn and chunk_id is not None:
            # True = ignore in torch attention mask.  Allow attention
            # only within the same chunk id (= Y-layer).  Always keep
            # the diagonal unmasked so padding rows don't softmax-NaN.
            valid = chunk_id >= 0
            same = chunk_id.unsqueeze(-1) == chunk_id.unsqueeze(-2)
            mask = ~(same & valid.unsqueeze(-1) & valid.unsqueeze(-2))
            L = mask.size(-1)
            diag = torch.eye(L, dtype=torch.bool, device=mask.device)
            mask = mask & ~diag                                       # self-attention always allowed
            # (B, L, L) src_mask isn't supported in torch 2.2, so loop
            out = torch.zeros_like(h)
            for b in range(B):
                out[b:b+1] = self.attn(h[b:b+1], mask=mask[b])
            h = out
        if self.factored:
            t = self.head_trunk(h)
            return self.head_cat(t), self.head_size(t), self.head_yaw(t)
        return self.head(h)


def train(epochs: int = 5, batch: int = 4, lr: float = 1e-3, grid: int = 16, limit: int = 0,
          resume: bool = False, factored: bool = False, unet: bool = False,
          sched_sampling: float = 0.0, ch: int = 32, augment: bool = False,
          dataset: str = "omr", oversample: int = 0):
    """`sched_sampling` is the final mix probability (linearly ramped 0->p
    across epochs).  Each training step then runs a 2nd forward pass with
    the model's own argmax from the first pass substituted into prev at
    randomly chosen positions, so the model learns to tolerate its own
    (wrong) history instead of only ground truth.  Non-factored only — the
    factored head doesn't predict ldraw directly."""
    if sched_sampling > 0 and factored:
        raise SystemExit("--sched-sampling currently supported only without --factored")
    if dataset == "hybrid":
        # OMR for KL fidelity + oversampled distilled for structural priors
        ds = OMRDataset(grid=grid, factored=factored, augment=augment, dataset_dir=DATASET_DIR)
        dist_files = sorted(f for f in DISTILLED_DIR.glob("*.npz") if not f.name.startswith("._"))
        n_over = oversample if oversample > 0 else max(1, len(ds.files) // max(1, len(dist_files)))
        ds.files = ds.files + dist_files * n_over
        print(f"hybrid dataset: {len(dist_files)} distilled x{n_over} + OMR = {len(ds.files)} entries", flush=True)
    else:
        dataset_dir = {"omr": DATASET_DIR, "distilled": DISTILLED_DIR}[dataset]
        ds = OMRDataset(grid=grid, factored=factored, augment=augment, dataset_dir=dataset_dir)
    if limit:
        ds.files = ds.files[:limit]
    n = len(ds)
    tr = int(n * 0.9)
    train_set = torch.utils.data.Subset(ds, range(tr))
    val_set = torch.utils.data.Subset(ds, range(tr, n))
    extra = f", cats={len(ds.cats)}, sizes={len(ds.sizes)}" if factored else ""
    print(f"dataset: {n} models ({tr} train / {n - tr} val), grid={grid}, "
          f"tokens={len(ds.tokens)}, kinds={len(ds.kinds)}{extra}, factored={factored}", flush=True)
    dl_tr = DataLoader(train_set, batch_size=batch, shuffle=True, collate_fn=_collate, num_workers=0)
    dl_v = DataLoader(val_set, batch_size=batch, shuffle=False, collate_fn=_collate, num_workers=0)
    device = "cpu"                                                     # Conv3D not supported on MPS (torch 2.2)
    print(f"device: {device} (Conv3D not supported on MPS)", flush=True)
    mkw = {"n_cats": len(ds.cats), "n_sizes": len(ds.sizes)} if factored else {}
    m = Model(len(ds.tokens), len(ds.kinds), grid=grid, ch=ch, factored=factored, unet=unet, **mkw).to(device)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"model params: {n_params/1e6:.2f}M, ch={ch}, unet={unet}", flush=True)
    out_path = _model_path(factored, unet, sched_sampling, ch, augment, dataset, oversample)
    if resume and out_path.exists():
        prev = torch.load(out_path, map_location=device, weights_only=False)
        m.load_state_dict(prev["state_dict"])
        print(f"resumed from {out_path.name}", flush=True)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    for ep in range(epochs):
        m.train()
        t0 = time.time()
        total = n_tokens = 0.0
        p_sched = sched_sampling * ep / max(1, epochs - 1)                 # linear ramp 0 -> sched_sampling
        for batch_data in dl_tr:
            if factored:
                v, pos, tok, kind, chunk, cat, sz, yaw = (b.to(device) for b in batch_data)
            else:
                v, pos, tok, kind, chunk = (b.to(device) for b in batch_data)
            prev = torch.full_like(tok, len(ds.tokens) + 1)
            prev[:, 1:] = tok[:, :-1].clamp(0, len(ds.tokens))
            if p_sched > 0 and not factored:
                with torch.no_grad():
                    tf_logits = m(v, pos, prev, kind, chunk_id=chunk)
                    pred = tf_logits.argmax(-1).clamp(0, len(ds.tokens))     # model's own guess per position
                mix = torch.rand_like(tok, dtype=torch.float) < p_sched
                prev_mix = prev.clone()
                prev_mix[:, 1:] = torch.where(mix[:, 1:], pred[:, :-1], prev[:, 1:])
                out = m(v, pos, prev_mix, kind, chunk_id=chunk)
            else:
                out = m(v, pos, prev, kind, chunk_id=chunk)
            if factored:
                lc, ls, ly = out
                loss = (F.cross_entropy(lc.view(-1, lc.size(-1)), cat.view(-1), ignore_index=-1)
                        + F.cross_entropy(ls.view(-1, ls.size(-1)), sz.view(-1), ignore_index=-1)
                        + F.cross_entropy(ly.view(-1, ly.size(-1)), yaw.view(-1), ignore_index=-1))
            else:
                loss = F.cross_entropy(out.view(-1, out.size(-1)), tok.view(-1), ignore_index=-1)
            opt.zero_grad(); loss.backward(); opt.step()
            n_valid = (tok != -1).sum().item()
            total += loss.item() * n_valid; n_tokens += n_valid
        tr_nll = total / max(1, n_tokens)
        m.eval()
        with torch.no_grad():
            if factored:
                tot_c = tot_s = tot_y = c_top1 = s_top1 = y_top1 = n_tokens = 0.0
                for batch_data in dl_v:
                    v, pos, tok, kind, chunk, cat, sz, yaw = (b.to(device) for b in batch_data)
                    prev = torch.full_like(tok, len(ds.tokens) + 1)
                    prev[:, 1:] = tok[:, :-1].clamp(0, len(ds.tokens))
                    lc, ls, ly = m(v, pos, prev, kind, chunk_id=chunk)
                    mask = tok != -1
                    nv = mask.sum().item()
                    tot_c += F.cross_entropy(lc.view(-1, lc.size(-1)), cat.view(-1), ignore_index=-1).item() * nv
                    tot_s += F.cross_entropy(ls.view(-1, ls.size(-1)), sz.view(-1), ignore_index=-1).item() * nv
                    tot_y += F.cross_entropy(ly.view(-1, ly.size(-1)), yaw.view(-1), ignore_index=-1).item() * nv
                    c_top1 += ((lc.argmax(-1) == cat) & mask).sum().item()
                    s_top1 += ((ls.argmax(-1) == sz) & mask).sum().item()
                    y_top1 += ((ly.argmax(-1) == yaw) & mask).sum().item()
                    n_tokens += nv
                print(f"epoch {ep+1}: train nll {tr_nll:.3f}  "
                      f"val cat {tot_c/max(1,n_tokens):.3f}/top1 {c_top1/max(1,n_tokens):.3f}  "
                      f"size {tot_s/max(1,n_tokens):.3f}/top1 {s_top1/max(1,n_tokens):.3f}  "
                      f"yaw {tot_y/max(1,n_tokens):.3f}/top1 {y_top1/max(1,n_tokens):.3f}  "
                      f"({time.time()-t0:.1f}s)", flush=True)
            else:
                total = n_tokens = correct = top5 = 0.0
                for v, pos, tok, kind, chunk in dl_v:
                    v, pos, tok, kind, chunk = (b.to(device) for b in (v, pos, tok, kind, chunk))
                    prev = torch.full_like(tok, len(ds.tokens) + 1)
                    prev[:, 1:] = tok[:, :-1].clamp(0, len(ds.tokens))
                    logits = m(v, pos, prev, kind, chunk_id=chunk)
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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {"state_dict": m.state_dict(), "n_tokens": len(ds.tokens), "n_kinds": len(ds.kinds),
            "tokens": ds.tokens, "kinds": ds.kinds, "grid": grid, "factored": factored, "unet": unet, "ch": ch}
    if factored:
        ckpt["cats"] = ds.cats
        ckpt["sizes"] = [list(s) for s in ds.sizes]
    torch.save(ckpt, out_path)
    print(f"-> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("train", "eval"))
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--grid", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="use only first N dataset files (quick test)")
    ap.add_argument("--resume", action="store_true", help="load weights from existing cnn_decoder.pt before training")
    ap.add_argument("--factored", action="store_true", help="three heads (category, footprint, yaw) instead of one n_tokens-way softmax")
    ap.add_argument("--unet", action="store_true", help="UNet3D encoder (16->8->4 downsample with skips) instead of a flat 3-conv encoder")
    ap.add_argument("--sched-sampling", type=float, default=0.0, help="final probability (ramped 0->p across epochs) of substituting the model's own argmax for ground-truth prev at a position; non-factored only")
    ap.add_argument("--ch", type=int, default=32, help="encoder channel width (also scales UNet width at every level)")
    ap.add_argument("--augment", action="store_true", help="180 degree rotation around Y with 50% probability per sample (2x effective dataset)")
    ap.add_argument("--dataset", choices=("omr", "distilled", "hybrid"), default="omr", help="dataset to train on: raw OMR, sampler-distilled, or hybrid (OMR + distilled oversampled to parity)")
    ap.add_argument("--oversample", type=int, default=0, help="hybrid: distilled oversample factor (default auto = len(OMR)//len(distilled))")
    a = ap.parse_args()
    if a.cmd == "train":
        train(epochs=a.epochs, batch=a.batch, lr=a.lr, grid=a.grid, limit=a.limit,
               resume=a.resume, factored=a.factored, unet=a.unet, sched_sampling=a.sched_sampling,
               ch=a.ch, augment=a.augment, dataset=a.dataset, oversample=a.oversample)
    else:
        print("eval not implemented yet; load the model and run inference directly")
