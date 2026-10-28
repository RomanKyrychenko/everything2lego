"""Mesh2Brick M1 model (docs/mesh2brick_architecture.md §3): target+state voxel grid -> 3D U-Net feature
volume; brick tokens -> causal transformer with cross-attention into the volume; factorised action head
(op, root pose, parent pointer, parent connector, part, child connector, yaw/flip/slide/rot, colour).

Mac-size defaults (d=256, 6 layers, C=128) for the M1 overfit; the doc's sizes are constructor args.
Cross-attention per brick token = a WIN^3 window of stride-2 volume cells around the previous brick + GLOB-pooled global
tokens; the root brick has a dense start-cell head over the volume and reads the feature at that cell.
shortcut: connector heads are flat over the first NCONN connectors of a part, and the part head is flat (not
hierarchical) -- upgrade in M2 when baseplates (>NCONN studs) matter.
Torch-version agnostic (runs under torch 2.2 for generation and 2.14 for MPS training)."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NCONN = 256           # connector slots per part (p99 of placements is 79; baseplates exceed it)
N_ROT = 25            # 24 proper cube rotations + "other" (hinged / off-lattice)
N_YAW, N_SLIDE, N_ANG = 36, 41, 36                 # 10 deg bins; slide in LDU clamped to [-20, 20]
OPS = ("PLACE", "STOP")
ROOT_W = 5.0          # loss weight on the root factors (anchor, rotation, first part): the only ones that must read the volume


def cube_rotations() -> np.ndarray:
    """The 24 proper rotations of the cube, (24, 3, 3), index 0 = identity."""
    mats = []
    for perm in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
        for signs in ((1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1), (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1)):
            R = np.zeros((3, 3)); R[np.arange(3), perm] = signs
            if np.linalg.det(R) > 0.5: mats.append(R)
    mats.sort(key=lambda R: -np.trace(R))
    return np.stack(mats)


ROTS = cube_rotations()


def rot_class(R: np.ndarray) -> int:
    """Nearest of the 24 cube rotations, or 24 when R is more than ~3 deg off every one of them."""
    d = np.abs(ROTS - R[None]).reshape(24, -1).max(1); k = int(d.argmin())
    return k if d[k] < 0.05 else 24


def fourier(x: torch.Tensor, n: int = 6) -> torch.Tensor:
    """x (..., k) in [0, 1] -> (..., k * 2n) sin/cos at frequencies 2^0 .. 2^(n-1) (times pi)."""
    f = (2.0 ** torch.arange(n, device=x.device, dtype=x.dtype)) * math.pi
    a = x[..., None] * f
    return torch.cat([a.sin(), a.cos()], -1).flatten(-2)


# ------------------------------------------------------------------------------------------------ encoder
def _block(ci, co):
    return nn.Sequential(nn.Conv3d(ci, co, 3, padding=1), nn.GroupNorm(8, co), nn.SiLU(), nn.Conv3d(co, co, 3, padding=1), nn.GroupNorm(8, co), nn.SiLU())


class UNet3D(nn.Module):
    """(B, in_ch, H, W, D) -> (B, C, H/2, W/2, D/2).  The input is 2x2x2-average-pooled first (the
    occupancy fractions keep the plate-level detail; full-resolution Conv3d is 8x the cost on MPS)
    and Fourier voxel positions are concatenated; levels at stride 2, 4, 8, output stride 2."""

    def __init__(self, in_ch: int = 2, C: int = 128, n_freq: int = 4):
        super().__init__()
        self.n_freq = n_freq; pos = 3 * 2 * n_freq
        w = (C // 2, C, C)
        self.e0 = _block(in_ch + pos, w[0]); self.e1 = _block(w[0], w[1]); self.e2 = _block(w[1], w[2])
        self.u1 = _block(w[2] + w[1], w[1]); self.u0 = _block(w[1] + w[0], C)
        self.out_ch = C

    def forward(self, x):
        x = F.avg_pool3d(x, 2); B, _, H, W, D = x.shape
        g = torch.stack(torch.meshgrid(torch.linspace(0, 1, H), torch.linspace(0, 1, W), torch.linspace(0, 1, D), indexing="ij"), -1).to(x)
        pos = fourier(g, self.n_freq).permute(3, 0, 1, 2)[None].expand(B, -1, -1, -1, -1)
        e0 = self.e0(torch.cat([x, pos], 1)); e1 = self.e1(F.avg_pool3d(e0, 2)); e2 = self.e2(F.avg_pool3d(e1, 2))
        u1 = self.u1(torch.cat([F.interpolate(e2, scale_factor=2, mode="nearest"), e1], 1))
        return self.u0(torch.cat([F.interpolate(u1, scale_factor=2, mode="nearest"), e0], 1))          # stride 2


# ------------------------------------------------------------------------------------------------ decoder
WIN = 6               # cross-attention window in stride-2 cells around the query's anchor (12 plates x 12 studs x 12 studs)
GLOB = 4              # global tokens = feature volume pooled by this factor (stride 8 -> 12x4x4 = 192 tokens at (96,32,32))


class Block(nn.Module):
    """Pre-LN: causal self-attention over brick tokens, windowed+global cross-attention into the volume, FFN."""

    def __init__(self, d, heads):
        super().__init__()
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.sa = nn.MultiheadAttention(d, heads, batch_first=True)
        self.q, self.kv, self.o = nn.Linear(d, d), nn.Linear(d, 2 * d), nn.Linear(d, d); self.heads = heads
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def cross(self, x, mem, widx, wmask, glob):
        """x (B,Q,d); mem (B,N,d) volume tokens; widx (B,Q,K) token index per query window, wmask (B,Q,K) True = invalid;
        glob (B,G,d).  Keys of query i = mem[widx[i]] ++ glob."""
        B, Q, d = x.shape; h = self.heads; K = widx.shape[-1]; G = glob.shape[1]
        kv = self.kv(mem); kw = torch.gather(kv, 1, widx.reshape(B, Q * K, 1).expand(-1, -1, 2 * d)).reshape(B, Q, K, 2 * d)
        kg = self.kv(glob)[:, None].expand(-1, Q, -1, -1)
        k, v = torch.cat([kw, kg], 2).chunk(2, -1)                                      # (B,Q,K+G,d)
        q = self.q(x).reshape(B, Q, h, d // h).transpose(1, 2)                           # (B,h,Q,dh)
        k = k.reshape(B, Q, K + G, h, d // h).permute(0, 3, 1, 2, 4); v = v.reshape(B, Q, K + G, h, d // h).permute(0, 3, 1, 2, 4)
        att = torch.einsum("bhqd,bhqkd->bhqk", q, k) / math.sqrt(d // h)              # (B,h,Q,K+G), no broadcast product kept
        mask = torch.cat([wmask, torch.zeros(B, Q, G, dtype=torch.bool, device=x.device)], 2)
        att = att.masked_fill(mask[:, None], float("-inf")).softmax(-1)
        return self.o(torch.einsum("bhqk,bhqkd->bhqd", att, v).transpose(1, 2).reshape(B, Q, d))

    def forward(self, h, mem, widx, wmask, glob, causal, pad):
        x = self.n1(h); h = h + self.sa(x, x, x, attn_mask=causal, key_padding_mask=pad, need_weights=False)[0]
        h = h + self.cross(self.n2(h), mem, widx, wmask, glob)
        return h + self.ff(self.n3(h))


class Mesh2Brick(nn.Module):
    def __init__(self, n_part: int, n_color: int, grid=(96, 32, 32), d: int = 256, layers: int = 6, heads: int = 8, C: int = 128, in_ch: int = 2):
        super().__init__()
        self.grid, self.d, self.C = tuple(grid), d, C
        self.enc = UNet3D(in_ch, C); self.mem_proj = nn.Linear(C + 3 * 2 * 4, d); self.glob_proj = nn.Linear(C + 3 * 2 * 4, d)
        self.part_emb = nn.Embedding(n_part, d); self.col_emb = nn.Embedding(n_color, d)
        self.pose_mlp = nn.Sequential(nn.Linear(3 * 12 + N_ROT + 9 + 3 * 12, d), nn.GELU(), nn.Linear(d, d))
        self.bos = nn.Parameter(torch.zeros(1, 1, d)); self.pos_emb = nn.Embedding(1024, d)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers)); self.norm = nn.LayerNorm(d)
        # action head (chain: each factor sees the ground-truth / chosen previous factors)
        self.op = nn.Linear(d, len(OPS))
        self.root_cell = nn.Conv3d(C, 1, 1)                                               # dense "first brick starts here" over stride-2 cells
        self.cell_proj = nn.Linear(C, d)                                                  # volume feature at the root cell -> root context
        self.root_rot = nn.Linear(d, N_ROT)
        self.ptr_q, self.ptr_k = nn.Linear(d, d), nn.Linear(d, d)
        self.no_parent = nn.Parameter(torch.zeros(d))
        self.pconn_emb = nn.Embedding(NCONN + 1, d); self.cconn_emb = nn.Embedding(NCONN + 1, d)   # last = none (root)
        mlp = lambda i, o: nn.Sequential(nn.Linear(i, d), nn.GELU(), nn.Linear(d, o))            # noqa: E731
        self.h_pconn = mlp(2 * d, NCONN); self.h_part = mlp(3 * d, n_part); self.h_cconn = mlp(4 * d, NCONN)
        self.h_geom = mlp(5 * d, N_YAW + 2 + N_SLIDE + 3 * N_ANG); self.h_color = mlp(5 * d, n_color)

    # ---- inputs
    def pose_feats(self, centre, extent, R):
        """centre / extent (B,T,3) grid units, R (B,T,3,3) -> (B,T,F); rot class one-hot computed on the host."""
        H, W, D = self.grid; scale = torch.tensor([W, H, D], dtype=centre.dtype, device=centre.device)
        rc = torch.as_tensor(np.array([[rot_class(r) for r in b] for b in R.cpu().numpy()]), device=R.device)
        return torch.cat([fourier(centre / scale), F.one_hot(rc, N_ROT).to(centre.dtype), R.flatten(-2), fourier(extent / scale)], -1), rc

    def tokens(self, part, color, centre, extent, R):
        f, rc = self.pose_feats(centre, extent, R)
        e = self.part_emb(part) + self.col_emb(color) + self.pose_mlp(f)
        return e, rc

    def memory(self, vol):
        """-> dict: Fv (B,C,h,w,d) stride-2 features, mem (B,h*w*d,d) tokens, glob (B,G,d), root_logits (B,h*w*d)."""
        Fv = self.enc(vol); B, C, h, w, dd = Fv.shape
        g = torch.stack(torch.meshgrid(torch.linspace(0, 1, h), torch.linspace(0, 1, w), torch.linspace(0, 1, dd), indexing="ij"), -1).to(Fv)
        pos = fourier(g, 4).reshape(-1, 3 * 8)[None].expand(B, -1, -1)
        mem = self.mem_proj(torch.cat([Fv.flatten(2).transpose(1, 2), pos], -1))
        Fg = F.avg_pool3d(Fv, GLOB); gg = torch.stack(torch.meshgrid(*(torch.linspace(0, 1, n) for n in Fg.shape[2:]), indexing="ij"), -1).to(Fv)
        glob = self.glob_proj(torch.cat([Fg.flatten(2).transpose(1, 2), fourier(gg, 4).reshape(-1, 24)[None].expand(B, -1, -1)], -1))
        return {"Fv": Fv, "mem": mem, "glob": glob, "root_logits": self.root_cell(Fv).flatten(1), "shape": (h, w, dd)}

    def windows(self, anchors, valid, shape):
        """anchors (B,Q,3) stride-2 cells (y,x,z) per query, valid (B,Q) -> flat token indices (B,Q,WIN^3) + invalid mask."""
        h, w, dd = shape; dev = anchors.device
        off = torch.stack(torch.meshgrid(*(torch.arange(-WIN // 2, WIN - WIN // 2, device=dev),) * 3, indexing="ij"), -1).reshape(-1, 3)   # (K,3)
        cells = anchors[:, :, None, :] + off[None, None]                                                                               # (B,Q,K,3)
        inside = (cells >= 0).all(-1) & (cells[..., 0] < h) & (cells[..., 1] < w) & (cells[..., 2] < dd) & valid[:, :, None]
        flat = (cells[..., 0].clamp(0, h - 1) * w + cells[..., 1].clamp(0, w - 1)) * dd + cells[..., 2].clamp(0, dd - 1)
        return flat, ~inside

    def decode(self, tok, M, anchors, avalid, pad):
        """tok (B,T,d) brick tokens; anchors (B,T+1,3) window anchors per state (state t predicts brick t: anchor = parent of
        brick t; the root / parentless states get global tokens only); -> states (B,T+1,d)."""
        B, T, _ = tok.shape
        h = torch.cat([self.bos.expand(B, -1, -1), tok], 1) + self.pos_emb(torch.arange(T + 1, device=tok.device))[None]
        causal = torch.triu(torch.ones(T + 1, T + 1, dtype=torch.bool, device=tok.device), 1)
        kpm = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=tok.device), pad], 1)
        widx, wmask = self.windows(anchors, avalid, M["shape"])
        for b in self.blocks: h = b(h, M["mem"], widx, wmask, M["glob"], causal, kpm)
        return self.norm(h)

    def root_context(self, s0, M, cell):
        """s0 (B,d) BOS state + the volume feature at the root cell (B,3) stride-2 (y,x,z) -> root context (B,d)."""
        h, w, dd = M["shape"]; flat = (cell[:, 0] * w + cell[:, 1]) * dd + cell[:, 2]
        fc = torch.gather(M["Fv"].flatten(2), 2, flat[:, None, None].expand(-1, self.C, 1))[:, :, 0]
        return s0 + self.cell_proj(fc)

    # ---- head (teacher forced): b = batch of target columns
    def head(self, s, b, M):
        """s (B,T+1,d) states; b targets incl. root_cell (B,3). Returns logits dict."""
        B, T1, d = s.shape; T = T1 - 1; h = s[:, :T].clone()
        h0 = self.root_context(s[:, 0], M, b["root_cell"][:, 0]); h = torch.cat([h0[:, None], h[:, 1:]], 1)
        out = {"op": self.op(s), "root_cell": M["root_logits"], "arot": self.root_rot(h0)}
        q, k = self.ptr_q(h), self.ptr_k(s[:, 1:])                                        # keys = placed bricks 0..T-1
        ptr = (q @ k.transpose(1, 2)) / math.sqrt(d)
        ptr = ptr.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool, device=s.device)), float("-inf"))   # brick t: parents < t
        out["parent"] = ptr
        par = b["parent"].clamp(min=0)
        sp = torch.where((b["parent"] >= 0)[..., None], torch.gather(s[:, 1:], 1, par[..., None].expand(-1, -1, d)), self.no_parent.expand(B, T, d))
        c1 = torch.cat([h, sp], -1); out["pconn"] = self.h_pconn(c1)
        c2 = torch.cat([c1, self.pconn_emb(torch.where(b["parent"] >= 0, b["pconn"].clamp(0, NCONN - 1), torch.full_like(b["pconn"], NCONN)))], -1)
        out["part"] = self.h_part(c2)
        c3 = torch.cat([c2, self.part_emb(b["part"])], -1); out["cconn"] = self.h_cconn(c3)
        c4 = torch.cat([c3, self.cconn_emb(torch.where(b["parent"] >= 0, b["cconn"].clamp(0, NCONN - 1), torch.full_like(b["cconn"], NCONN)))], -1)
        g = self.h_geom(c4); i = 0
        for k_, n in (("yaw", N_YAW), ("flip", 2), ("slide", N_SLIDE), ("rx", N_ANG), ("ry", N_ANG), ("rz", N_ANG)):
            out[k_] = g[..., i:i + n]; i += n
        out["color"] = self.h_color(c4)
        return out

    @staticmethod
    def anchors_from(b, pad):
        """Window anchors (B,T+1,3) stride-2 (y,x,z) per state + validity: state t (predicting brick t) looks around the
        previous brick t-1 (the frontier; known at inference too); the root state has no window (global tokens only)."""
        B, T = pad.shape; cen = b["centre"]                                               # (B,T,3) x,y,z grid units
        c = torch.cat([torch.zeros(B, 1, 3, device=cen.device, dtype=cen.dtype), cen], 1)   # state t -> brick t-1
        anchors = torch.stack([c[..., 1], c[..., 0], c[..., 2]], -1).div(2).floor().long()
        valid = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=pad.device), ~pad], 1)
        return anchors, valid

    def forward(self, vol, b, pad):
        tok, rc = self.tokens(b["part"], b["color"], b["centre"], b["extent"], b["R"])
        M = self.memory(vol); anchors, valid = self.anchors_from(b, pad)
        s = self.decode(tok, M, anchors, valid, pad)
        return self.head(s, b, M), rc


def loss_and_acc(out, b, pad, rc, nconn_part, grid):
    """Sum of CE over the factors present at each step (+ op / root). Returns (loss, {name: acc})."""
    valid = ~pad; B, T = valid.shape; H, W, D = grid
    def ce(logits, tgt, m):
        if m.sum() == 0: return logits.sum() * 0, float("nan")
        l = F.cross_entropy(logits[m], tgt[m]); a = (logits[m].argmax(-1) == tgt[m]).float().mean().item()
        return l, a
    acc, loss = {}, 0.0
    # op: PLACE at every valid step, STOP at the first padded position (state T_i)
    op_t = torch.ones(B, T + 1, dtype=torch.long, device=pad.device); n = valid.sum(1); op_t[torch.arange(B), n] = 1; op_t[:, :T][valid] = 0
    op_m = torch.cat([valid, torch.zeros(B, 1, dtype=torch.bool, device=pad.device)], 1); op_m[torch.arange(B), n] = True
    l, acc["op"] = ce(out["op"], op_t, op_m); loss = loss + l
    m0 = valid[:, 0]; h2, w2, d2 = H // 2, W // 2, D // 2; rcell = b["root_cell"][:, 0]
    l, acc["root_cell"] = ce(out["root_cell"], (rcell[:, 0] * w2 + rcell[:, 1]) * d2 + rcell[:, 2], m0); loss = loss + ROOT_W * l
    l, acc["arot"] = ce(out["arot"], rc[:, 0], m0); loss = loss + ROOT_W * l
    has_p = valid & (b["parent"] >= 0)
    l, acc["parent"] = ce(out["parent"], b["parent"].clamp(min=0), has_p); loss = loss + l
    npar = nconn_part[b["part"][torch.arange(B)[:, None], b["parent"].clamp(min=0)]]
    pc_logits = out["pconn"].masked_fill(torch.arange(NCONN, device=pad.device)[None, None] >= npar[..., None], float("-inf"))
    l, acc["pconn"] = ce(pc_logits, b["pconn"].clamp(0, NCONN - 1), has_p & (b["pconn"] < NCONN)); loss = loss + l
    l, acc["part"] = ce(out["part"], b["part"], valid); loss = loss + l
    l, acc["part0"] = ce(out["part"][:, 0], b["part"][:, 0], m0); loss = loss + (ROOT_W - 1) * l      # the root part only sees the volume
    nch = nconn_part[b["part"]]
    cc_logits = out["cconn"].masked_fill(torch.arange(NCONN, device=pad.device)[None, None] >= nch[..., None], float("-inf"))
    l, acc["cconn"] = ce(cc_logits, b["cconn"].clamp(0, NCONN - 1), has_p & (b["cconn"] < NCONN)); loss = loss + l
    fam = b["fam"]
    l, acc["yaw"] = ce(out["yaw"], b["yaw"], has_p & (fam <= 2)); loss = loss + l
    l, acc["flip"] = ce(out["flip"], b["flip"], has_p & (fam == 1) | has_p & (fam == 2)); loss = loss + l
    l, acc["slide"] = ce(out["slide"], b["slide"], has_p & (fam == 2)); loss = loss + l
    for i, k in enumerate(("rx", "ry", "rz")):
        l, acc[k] = ce(out[k], b["rot"][..., i], has_p & (fam == 3)); loss = loss + l
    l, acc["color"] = ce(out["color"], b["color"], valid); loss = loss + l
    return loss, acc
