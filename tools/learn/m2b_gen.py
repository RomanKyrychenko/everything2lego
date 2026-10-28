"""Mesh2Brick free-run generation with hard constraints (docs/mesh2brick_architecture.md §3.6):
greedy factorised decoding, catalogue mate masks, pose realised with bricknet.decode_graph, rejection on
collision (bricknet.collision) or on leaving the (dilated) target grid, top-k retries, then STOP.

    BRICKNET_DATA=/Volumes/PRO-G40/lego_data/bricknet .venv/bin/python tools/learn/m2b_gen.py \
        --ckpt $LEGO_DATA/analysis/m2b/run1/model.pt --split sft --n 16 --out $LEGO_DATA/analysis/m2b/run1/gen

Runs in the project .venv (bricknet + torch 2.2).  Writes <out>/<src>.ldr + summary.json with per-model
bricks, IoU of the box-filled build vs the target grid, rejections.  --grid-from mesh|bbox picks the target."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1] / "tools" / "set_study"))
from bricknet_shape import _part_bbox, boxes_to_grid  # noqa: E402
from m2b_data import mesh_grid  # noqa: E402
from m2b_model import NCONN, N_ROT, ROTS, Mesh2Brick  # noqa: E402
from m2b_train import Vocab, load_split  # noqa: E402

FAM_NAMES = ("stud", "hinge", "axle", "ball", "fixed")


class Catalog:
    """Connector tuples per part and the mate masks over the part vocabulary."""

    def __init__(self, n_part: int):
        from bricknet.core import _AXLE_MATES, _STUD_MATES
        from bricknet.graph import _part_info
        self.info = _part_info; self.conns = [_part_info(p).conns for p in range(n_part)]
        keys = sorted({self.key(c) for cs in self.conns for c in cs}); self.kidx = {k: i for i, k in enumerate(keys)}
        self.has = np.zeros((n_part, len(keys)), bool)
        for p, cs in enumerate(self.conns):
            for c in cs: self.has[p, self.kidx[self.key(c)]] = True
        self.stud, self.axle = _STUD_MATES, _AXLE_MATES

    @staticmethod
    def key(c):                                                             # (family, sub, pol)
        return ("stud" if c[0] in ("stud", "hole") else c[0], c[1], c[2])

    def mates(self, a, b) -> bool:
        if a[0] != b[0]: return False
        if a[0] == "stud": return (min(a[1], b[1]), max(a[1], b[1])) in self.stud
        if a[0] == "axle": return (min(a[1], b[1]), max(a[1], b[1])) in self.axle
        return a[1] == b[1] and a[2] != b[2]

    def part_mask(self, pc) -> np.ndarray:
        k = self.key(pc); cols = [i for key, i in self.kidx.items() if self.mates(k, key)]
        return self.has[:, cols].any(1) if cols else np.zeros(len(self.has), bool)

    def conn_mask(self, part: int, pc) -> np.ndarray:
        k = self.key(pc); m = np.zeros(NCONN, bool)
        for i, c in enumerate(self.conns[part][:NCONN]): m[i] = self.mates(k, self.key(c))
        return m


def make_edge(parent, child, pc, cc, yaw, flip, slide, rot):
    from bricknet.core import AxleEdge, AxleSub, BallEdge, FixedEdge, HingeEdge, StudEdge, StudSub
    fam = Catalog.key(pc)[0]
    if fam in ("stud", "axle"):
        sc = StudSub if fam == "stud" else AxleSub; base = (parent, child, sc[pc[1]], sc[cc[1]], pc[3], cc[3])
        return StudEdge(*base, yaw=yaw) if fam == "stud" else AxleEdge(*base, flip=flip, yaw=yaw, slide=slide)
    s = int(pc[1]); base = (parent, child, s * 2 + (pc[2] == "on"), s * 2 + (pc[2] == "in"), pc[3], cc[3])
    return HingeEdge(*base, flip=flip, yaw=yaw) if fam == "hinge" else BallEdge(*base, rx=rot[0], ry=rot[1], rz=rot[2]) if fam == "ball" else FixedEdge(*base)


def box_of(stem: str, M: np.ndarray, grid) -> np.ndarray:
    """(2,3) box in grid units [x, y, z] (0..W, 0..H, 0..D) of a part posed by M (LDU, grid frame)."""
    H, W, D = grid; bb = _part_bbox(stem)
    lo, hi = (np.zeros(3), np.ones(3) * 8) if bb is None else (np.asarray(bb[0], float), np.asarray(bb[1], float))
    c = np.array([[x, y, z, 1.0] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]); w = (M @ c.T).T[:, :3]
    a, b = w.min(0), w.max(0)
    return np.array([[a[0] / 20 + W / 2, -b[1] / 8, a[2] / 20 + D / 2], [b[0] / 20 + W / 2, -a[1] / 8, b[2] / 20 + D / 2]], np.float32)


class VoxelScene:
    """Collision test on the solid part voxels (m2b_data.solid_part, 4 LDU): a placement collides when more
    than `tol` of its fine cells are already occupied.  shortcut: stands in for bricknet.collision while
    pybullet is stubbed in this venv; swap back when a real pybullet is installed."""

    def __init__(self, grid, stems, tol: float = 0.1):
        from m2b_data import PITCH, solid_part
        H, W, D = grid; self.pitch, self.solid, self.stems, self.tol = PITCH, solid_part, stems, tol
        self.occ = np.zeros((int(W * 20 / PITCH) + 2, int(H * 8 / PITCH) + 2, int(D * 20 / PITCH) + 2), bool)   # world fine cells (x, y, z)
        self.off = np.array([W * 10, H * 8, D * 10]) / PITCH + 1                                               # LDU -> fine index offset

    def cells(self, part: int, M: np.ndarray):
        sp = self.solid(self.stems[part])
        if sp is None:                                                                                          # no watertight mesh: solid bbox
            bb = _part_bbox(self.stems[part]); lo, hi = (np.zeros(3), np.ones(3) * 8) if bb is None else (np.asarray(bb[0], float), np.asarray(bb[1], float))
            n = np.maximum(1, np.rint((hi - lo) / self.pitch)).astype(int); sp = (np.ones(n, bool), lo + self.pitch / 2)
        filled, org = sp; idx = np.argwhere(filled); p = idx * self.pitch + org                                  # local LDU centres
        w = p @ M[:3, :3].T + M[:3, 3]; f = np.rint(w / self.pitch + self.off).astype(int)                      # y: LDU -Y up, flip sign
        f[:, 1] = np.rint(-w[:, 1] / self.pitch + self.off[1]).astype(int)
        ok = (f >= 0).all(1) & (f < np.array(self.occ.shape)).all(1)
        return f[ok]

    def check(self, part: int, M: np.ndarray, **_) -> bool:
        f = self.cells(part, M)
        return bool(f is not None and len(f) and self.occ[f[:, 0], f[:, 1], f[:, 2]].mean() > self.tol)

    def add(self, part: int, M: np.ndarray) -> None:
        f = self.cells(part, M)
        if f is not None: self.occ[f[:, 0], f[:, 1], f[:, 2]] = True


def root_pose(stem: str, rc: int, anchor, grid) -> np.ndarray:
    """Pose whose box-min corner sits at grid cell `anchor` with cube rotation class rc."""
    H, W, D = grid; R = ROTS[rc] if rc < 24 else np.eye(3); bb = _part_bbox(stem)
    lo, hi = (np.zeros(3), np.ones(3) * 8) if bb is None else (np.asarray(bb[0], float), np.asarray(bb[1], float))
    c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]) @ R.T
    M = np.eye(4); M[:3, :3] = R
    M[0, 3] = 20 * (anchor[0] - W / 2) - c[:, 0].min(); M[2, 3] = 20 * (anchor[2] - D / 2) - c[:, 2].min(); M[1, 3] = -8 * anchor[1] - c[:, 1].max()
    return M


@torch.no_grad()
def generate(model, vocab, cat, target: np.ndarray, grid, device, max_steps=256, retries=8, topk=8, tau_out=0.15, seed=0, truth_root=None,
             restart=False, max_comps=64):
    """Greedy factorised decoding with constraints.  truth_root = (part_id, color, M) forces the first brick (debug).
    restart=True: when a chain dies (STOP, or `retries` consecutive rejections) start a new component from the densest
    unfilled target cell instead of ending -- covers big targets with several sub-assemblies (not one connected build)."""
    from bricknet.core import Part, Tree
    from bricknet.graph import decode_graph, tree_to_graph
    from scipy.ndimage import binary_dilation
    H, W, D = grid; rng = np.random.default_rng(seed); dil = binary_dilation(target, iterations=1); off = np.array([W / 2, 0, D / 2], np.float32)
    vol = torch.from_numpy(np.stack([target, np.zeros_like(target)]).astype(np.float32))[None].to(device)
    M = model.memory(vol); h2, w2, d2 = M["shape"]; max_steps = min(max_steps, model.pos_emb.num_embeddings - 1)   # positional table
    try:
        import pybullet; pybullet.getConnectionInfo                                   # real pybullet? (stubbed in some venvs)
        from bricknet import collision; scene = collision.CollisionScene()
    except Exception:                                                                 # noqa: BLE001
        scene = VoxelScene(grid, vocab.stems)
    parts, mats, boxes, stems = [], [], [], []; rejected = {"collision": 0, "outside": 0, "nomate": 0}; used = set()   # (brick, flat conn)
    comps = []                                                                         # [{"idx": [global brick ids], "edges": [], "root": M}]
    build = np.zeros(grid, bool)

    def states():
        T = len(parts)
        if T == 0:
            tok = torch.zeros(1, 0, model.d, device=device); cen = torch.zeros(1, 0, 3, device=device)
        else:
            bx = np.stack(boxes); cen = torch.tensor((bx[:, 0] + bx[:, 1]) / 2)[None].to(device); extent = torch.tensor(bx[:, 1] - bx[:, 0])[None].to(device)
            R = torch.tensor(np.stack(mats)[:, :3, :3], dtype=torch.float32)[None].to(device)
            tok, _ = model.tokens(torch.tensor([[p.part_id for p in parts]], device=device), torch.tensor([[vocab.col_idx.get(p.color, 0) for p in parts]], device=device), cen, extent, R)
        c = torch.cat([torch.zeros(1, 1, 3, device=device), cen], 1)
        anchors = torch.stack([c[..., 1], c[..., 0], c[..., 2]], -1).div(2).floor().long()
        valid = torch.cat([torch.zeros(1, 1, dtype=torch.bool, device=device), torch.ones(1, T, dtype=torch.bool, device=device)], 1)
        return model.decode(tok, M, anchors, valid, torch.zeros(1, T, dtype=torch.bool, device=device))

    def pick(logits, mask=None, greedy=True):
        l = logits.clone()
        if mask is not None: l[~torch.as_tensor(mask, device=l.device)] = float("-inf")
        if not torch.isfinite(l).any(): return None
        if greedy: return int(l.argmax())
        top = torch.topk(l, min(topk, int(torch.isfinite(l).sum()))); p = torch.softmax(top.values, 0).cpu().numpy()
        return int(top.indices[rng.choice(len(p), p=p / p.sum())])

    new_comp = True
    for _ in range(3 * max_steps):
        t = len(parts)
        if t >= max_steps: break
        s = states(); h = s[0, t]                                                     # t = bricks placed so far (restarts do not advance it)
        if not new_comp and int(model.op(h).argmax()) == 1:
            if not restart: break
            new_comp = True
        if new_comp:
            if len(comps) >= max_comps: break
            free = (target & ~build).reshape(h2, 2, w2, 2, d2, 2).any((1, 3, 5)).ravel()      # stride-2 cells with unfilled target
            if not free.any(): break
        placed = False; root_parts = None
        if new_comp:                                                                   # root: dense start cell -> context -> rot
            ci = pick(M["root_logits"][0], free); cell = np.array(np.unravel_index(ci, (h2, w2, d2)))
            h0 = model.root_context(s[:, t], M, torch.tensor(cell[None], device=device))[0]; rc = int(model.root_rot(h0).argmax())
            subs = [np.array(o) for o in np.ndindex(2, 2, 2)]
            subs.sort(key=lambda o: -int((target & ~build)[cell[0] * 2 + o[0], cell[1] * 2 + o[1], cell[2] * 2 + o[2]]))
        comp = comps[-1] if comps and not new_comp else None
        for attempt in range(32 if new_comp else retries):                             # root: 4 parts x 8 sub-cells
            greedy = attempt == 0 or new_comp
            if new_comp:
                ctx = torch.cat([h0, model.no_parent, model.pconn_emb.weight[NCONN]]); parent = pc = cc = None
                if root_parts is None: root_parts = torch.topk(model.h_part(ctx), 4).indices.tolist()
                part = root_parts[attempt // 8]; o = subs[attempt % 8]; ay, ax, az = cell * 2 + o
            else:
                ptr = (model.ptr_q(h) @ model.ptr_k(s[0, 1:t + 1]).T) / model.d ** 0.5
                in_comp = np.zeros(t, bool); in_comp[comp["idx"]] = True                   # parents only inside the current component
                parent = pick(ptr, in_comp, greedy=greedy)
                sp = s[0, 1 + parent]; ctx = torch.cat([h, sp]); pl = model.h_pconn(ctx)
                fr = np.arange(NCONN) < min(len(cat.conns[parts[parent].part_id]), NCONN)
                for (b_, c_) in used:
                    if b_ == parent and c_ < NCONN: fr[c_] = False
                pci = pick(pl, fr, greedy=greedy)
                if pci is None: rejected["nomate"] += 1; continue
                pc = cat.conns[parts[parent].part_id][pci]; ctx = torch.cat([ctx, model.pconn_emb.weight[pci]])
                part = pick(model.h_part(ctx), cat.part_mask(pc), greedy=greedy)
                if part is None: rejected["nomate"] += 1; continue
            ctx = torch.cat([ctx, model.part_emb.weight[part]])
            if new_comp:
                ctx = torch.cat([ctx, model.cconn_emb.weight[NCONN]]); yaw = flip = slide = 0; rot = (0, 0, 0)
            else:
                cci = pick(model.h_cconn(ctx), cat.conn_mask(part, pc), greedy=greedy)
                if cci is None: rejected["nomate"] += 1; continue
                cc = cat.conns[part][cci]; ctx = torch.cat([ctx, model.cconn_emb.weight[cci]])
                g = model.h_geom(ctx); yaw = int(g[:36].argmax()) * 10; flip = bool(g[36:38].argmax()); slide = int(g[38:79].argmax()) - 20
                rot = tuple(int(g[79 + 36 * i:79 + 36 * (i + 1)].argmax()) * 10 for i in range(3))
            color = vocab.colors[int(model.h_color(ctx).argmax())]
            if new_comp:
                if truth_root is not None and t == 0: part, color, Mp = truth_root[0], truth_root[1], truth_root[2]
                else: Mp = root_pose(vocab.stems[part], rc, (ax, ay, az), grid)
                new_edges = []
            else:
                local = {g_: i for i, g_ in enumerate(comp["idx"])}
                new_edges = comp["edges"] + [make_edge(local[parent], len(comp["idx"]), pc, cc, yaw, flip, slide, rot)]
                cparts = [parts[i] for i in comp["idx"]] + [Part(part, color)]
                try:
                    rel = decode_graph(tree_to_graph(Tree(tuple(cparts), tuple(new_edges))))[len(comp["idx"])]
                except Exception:                                                     # noqa: BLE001
                    rejected["nomate"] += 1; continue
                Mp = comp["root"] @ rel
            bx = box_of(vocab.stems[part], Mp, grid)
            cells, _ = mesh_grid(np.array([part]), Mp[None], vocab.stems, grid, bx[None] - off)          # same fill rule as the target
            if cells.sum() == 0: cells = boxes_to_grid(bx[None] - off, grid)                              # thin/round part: judge by its bbox
            if cells.sum() == 0 or (cells & ~dil).sum() / cells.sum() > tau_out: rejected["outside"] += 1; continue
            if scene.check(part, Mp, first_only=True): rejected["collision"] += 1; continue
            parts.append(Part(part, color)); mats.append(Mp); boxes.append(bx); stems.append(vocab.stems[part]); scene.add(part, Mp); build |= cells
            if new_comp: comps.append({"idx": [t], "edges": [], "root": Mp}); print(f"  component {len(comps)} starts at brick {t}  built cells {int(build.sum())}/{int(target.sum())}", flush=True)
            else: comp["idx"].append(t); comp["edges"] = new_edges; used.add((parent, pci)); used.add((t, cci))
            placed = True; new_comp = False
            break
        if not placed:
            if not restart or new_comp: break                                          # could not even start a component
            new_comp = True                                                            # chain died: restart elsewhere
    iou = (build & target).sum() / max(1, (build | target).sum())
    ldr = "".join(f"1 {p.color} " + " ".join(f"{v:.6g}" for v in [*Mm[:3, 3], *Mm[:3, :3].ravel()]) + f" {st}.dat\n" for p, Mm, st in zip(parts, mats, stems))
    return {"n": len(parts), "iou": float(iou), "rejected": rejected, "components": len(comps), "ldr": ldr}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True); ap.add_argument("--split", default="sft"); ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--device", default="cpu"); ap.add_argument("--grid-from", default="mesh")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--truth-root", action="store_true")
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True); dev = torch.device(a.device)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False); args = ck["args"]; grid = tuple(ck["grid"])
    vocab = Vocab(); model = Mesh2Brick(vocab.n_part, len(vocab.colors), grid, d=args["d"], layers=args["layers"], C=args["C"]).to(dev)
    model.load_state_dict(ck["model"]); model.eval(); cat = Catalog(vocab.n_part)
    models = load_split(a.split, a.n); H, W, D = grid; rows = []; t0 = time.time()
    for m in models:
        target = np.unpackbits(m["grid_mesh" if a.grid_from == "mesh" else "grid_bbox"])[: H * W * D].reshape(H, W, D).astype(bool)
        tr = (int(m["part"][0]), int(m["color"][0]), m["mats"][0].astype(float)) if a.truth_root else None
        r = generate(model, vocab, cat, target, grid, dev, seed=a.seed, truth_root=tr)
        (a.out / f"{m['src']}.ldr").write_text(r["ldr"]); truth_n = len(m["part"])
        rows.append({"src": m["src"], "truth_n": truth_n, **{k: v for k, v in r.items() if k != "ldr"}})
        print(f"{m['src']}: {r['n']} bricks (truth {truth_n})  IoU {r['iou']:.3f}  rejected {r['rejected']}  {time.time() - t0:.0f}s", flush=True)
    json.dump(rows, open(a.out / "summary.json", "w"), indent=1)
    print(f"mean IoU {np.mean([r['iou'] for r in rows]):.3f}  mean bricks {np.mean([r['n'] for r in rows]):.1f} / truth {np.mean([r['truth_n'] for r in rows]):.1f}", flush=True)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------------------------------- whole-target cover
def _rel_pose(e, parent_pid: int, child_pid: int) -> np.ndarray:
    """Child pose in the parent's frame for one tree edge (same maths as bricknet's sample_collision_free_tree)."""
    from bricknet.graph import _EDGE_DT, _edge_matrix, _tree_edge_row
    r = np.array([_tree_edge_row(e, 0, 1, parent_pid, child_pid)], dtype=_EDGE_DT)[0]
    if r["a"] == 0: return _edge_matrix(r, parent_pid, child_pid)
    return np.linalg.inv(_edge_matrix(r, child_pid, parent_pid))              # ball edges are stored in-side first


@torch.no_grad()
def generate_cover(model, vocab, cat, target: np.ndarray, grid, device, colours: np.ndarray | None = None, max_bricks=4000,
                   comp_len=255, retries=16, k_parts=32, tau_out=0.15, seed=0, log_every=100, attach=True, ctx=96,
                   flat_area=32.0, out_cap=8, max_roots=1):
    """Fill a large target with many sub-assemblies.  Each sub-assembly is its own sequence (<= comp_len bricks) and
    re-encodes the volume with everything built so far in the state channel (exactly how training windows see their
    prefix).  Parts are re-ranked by how many still-empty target cells they cover; roots start on the lowest empty layer.
    colours: (H,W,D) LDraw codes (-1 empty) from the mesh -> each brick takes the majority code of the cells it covers.

    attach: after the first root, a sub-assembly grows from up to `ctx` placed bricks that border empty target cells (the
    sequence prefix, as m2b_refine / sample_infill pose it), so every new brick hangs on a real connector; a free root is
    only allowed on the ground layers when nothing can attach, and at most `max_roots` free roots in all (without this,
    every sub-assembly was its own loose piece; allowing ground roots freely gave ~180 islands, since bricks on the same
    layer cannot connect to each other).
    flat_area: thin parts (<= 1.6 plates) with a footprint >= this many studs^2 must lie studs-up (real sets: 90-98 %).
    out_cap: at most this many cells of a part may stick out of the target (tau_out alone let a 16x16 plate overhang 38)."""
    from bricknet.core import Part
    from scipy.ndimage import binary_dilation
    H, W, D = grid; rng = np.random.default_rng(seed); dil = binary_dilation(target, iterations=1); off = np.array([W / 2, 0, D / 2], np.float32)
    scene = VoxelScene(grid, vocab.stems); build = np.zeros(grid, bool)
    out_parts, out_mats, out_stems, out_boxes = [], [], [], []
    rejected = {"collision": 0, "outside": 0, "nogain": 0, "orient": 0}; n_comp = 0; n_ground = 0; fails = 0
    dead = None                                                                        # stride-2 cells where no root fits
    dims = {}

    def orient_ok(part, Mp) -> bool:
        if part not in dims:
            bb = _part_bbox(vocab.stems[part])
            dims[part] = (0.0, 9.0) if bb is None else ((bb[1][0] - bb[0][0]) * (bb[1][2] - bb[0][2]) / 400, (bb[1][1] - bb[0][1]) / 8)
        area, h = dims[part]
        return not (h <= 1.6 and area >= flat_area) or Mp[1, 1] > 0.99

    def pick(logits, mask=None, greedy=True):
        l = logits.clone()
        if mask is not None: l[~torch.as_tensor(mask, device=l.device)] = float("-inf")
        if not torch.isfinite(l).any(): return None
        if greedy: return int(l.argmax())
        top = torch.topk(l, min(4, int(torch.isfinite(l).sum()))); p = torch.softmax(top.values, 0).cpu().numpy()
        return int(top.indices[rng.choice(len(p), p=p / p.sum())])

    def realise_check(part, Mp):
        """-> (box, cells, gain) or None when outside / colliding / wrongly oriented."""
        if not orient_ok(part, Mp): rejected["orient"] += 1; return None
        bx = box_of(vocab.stems[part], Mp, grid); cells = boxes_to_grid(bx[None] - off, grid)
        mc, _ = mesh_grid(np.array([part]), Mp[None], vocab.stems, grid, bx[None] - off)
        test = mc if mc.sum() else cells; out = int((test & ~dil).sum())
        if test.sum() == 0 or out / test.sum() > tau_out or out > out_cap: rejected["outside"] += 1; return None
        if scene.check(part, Mp): rejected["collision"] += 1; return None
        gain = int((cells & target & ~build).sum())
        if gain == 0: rejected["nogain"] += 1; return None
        return bx, cells, gain

    def borders_free(bx) -> bool:
        """Does the box (grid units, centred x/z) grown by one cell touch an empty target cell?"""
        (x0, y0, z0), (x1, y1, z1) = bx[0] + off - 1, bx[1] + off + 1
        sl = (slice(max(0, int(np.floor(y0))), max(0, int(np.ceil(y1)))), slice(max(0, int(np.floor(x0))), max(0, int(np.ceil(x1)))),
              slice(max(0, int(np.floor(z0))), max(0, int(np.ceil(z1)))))
        return bool((target[sl] & ~build[sl]).any())

    def colour_of(cells, fallback):
        if colours is None: return fallback
        c = colours[cells & (colours >= 0)]
        return int(np.bincount(c).argmax()) if len(c) else fallback

    def place(part, col, Mp, bx, cells):
        scene.add(part, Mp); build[...] = build | cells
        out_parts.append(Part(part, col)); out_mats.append(Mp); out_stems.append(vocab.stems[part]); out_boxes.append(bx)

    def grow(cp, cm, cb, M) -> int:
        """Extend the sequence (cp, cm, cb) brick by brick; parents must border empty target cells.  -> bricks added."""
        used, added = set(), 0
        while len(cp) < comp_len and len(out_parts) < max_bricks:
            T = len(cp); b = np.stack(cb)
            front = np.array([borders_free(x) for x in cb])
            if not front.any(): break
            tok, _ = model.tokens(torch.tensor([[p.part_id for p in cp]], device=device), torch.tensor([[vocab.col_idx.get(p.color, 0) for p in cp]], device=device),
                                  torch.tensor((b[:, 0] + b[:, 1]) / 2)[None].to(device), torch.tensor(b[:, 1] - b[:, 0])[None].to(device),
                                  torch.tensor(np.stack(cm)[:, :3, :3], dtype=torch.float32)[None].to(device))
            c = torch.cat([torch.zeros(1, 1, 3, device=device), torch.tensor((b[:, 0] + b[:, 1]) / 2, dtype=torch.float32)[None].to(device)], 1)
            anchors = torch.stack([c[..., 1], c[..., 0], c[..., 2]], -1).div(2).floor().long()
            valid = torch.cat([torch.zeros(1, 1, dtype=torch.bool, device=device), torch.ones(1, T, dtype=torch.bool, device=device)], 1)
            s = model.decode(tok, M, anchors, valid, torch.zeros(1, T, dtype=torch.bool, device=device)); h = s[0, T]
            ptr = (model.ptr_q(h) @ model.ptr_k(s[0, 1:T + 1]).T) / model.d ** 0.5
            best = None
            for attempt in range(retries):
                parent = pick(ptr, front, greedy=attempt == 0)
                if parent is None: break
                sp = s[0, 1 + parent]; ctx_ = torch.cat([h, sp])
                fr = np.arange(NCONN) < min(len(cat.conns[cp[parent].part_id]), NCONN)
                for (b_, c_) in used:
                    if b_ == parent and c_ < NCONN: fr[c_] = False
                pci = pick(model.h_pconn(ctx_), fr, greedy=attempt == 0)
                if pci is None: continue
                pc = cat.conns[cp[parent].part_id][pci]; ctx1 = torch.cat([ctx_, model.pconn_emb.weight[pci]])
                pl = model.h_part(ctx1); pl[~torch.as_tensor(cat.part_mask(pc))] = float("-inf")
                for part in torch.topk(pl, k_parts).indices.tolist():
                    if not torch.isfinite(pl[part]): break
                    ctx2 = torch.cat([ctx1, model.part_emb.weight[part]])
                    cci = pick(model.h_cconn(ctx2), cat.conn_mask(part, pc))
                    if cci is None: continue
                    cc = cat.conns[part][cci]; ctx3 = torch.cat([ctx2, model.cconn_emb.weight[cci]])
                    g = model.h_geom(ctx3); yaw = int(g[:36].argmax()) * 10; flip = bool(g[36:38].argmax()); slide = int(g[38:79].argmax()) - 20
                    rot = tuple(int(g[79 + 36 * i:79 + 36 * (i + 1)].argmax()) * 10 for i in range(3))
                    try:
                        Mq = cm[parent] @ _rel_pose(make_edge(0, 1, pc, cc, yaw, flip, slide, rot), cp[parent].part_id, part)
                    except Exception:                                                  # noqa: BLE001
                        continue
                    r = realise_check(part, Mq)
                    if r and (best is None or r[2] > best[3]): best = (part, Mq, r[0], r[2], r[1], parent, pci, cci, ctx3)
                if best is not None: break
            if best is None: break
            part, Mq, bx, _, cells, parent, pci, cci, ctx3 = best
            col = colour_of(cells, vocab.colors[int(model.h_color(ctx3).argmax())])
            place(part, col, Mq, bx, cells); cp.append(out_parts[-1]); cm.append(Mq); cb.append(bx); used |= {(parent, pci), (T, cci)}; added += 1
            if len(out_parts) % log_every == 0:
                print(f"  {len(out_parts)} bricks, {n_comp} free roots ({n_ground} on the ground), filled {int((build & target).sum())}/{int(target.sum())}", flush=True)
        return added

    while len(out_parts) < max_bricks and fails < 200:
        free = target & ~build
        if free.sum() < 4: break
        vol = torch.from_numpy(np.stack([target, build]).astype(np.float32))[None].to(device)
        M = model.memory(vol); h2, w2, d2 = M["shape"]
        # ---- attach: grow from placed bricks that border the empty target (lowest first, as the refiner's prefix)
        if attach and out_parts:
            pre = sorted((i for i in range(len(out_parts)) if borders_free(out_boxes[i])), key=lambda i: (out_boxes[i][0][1], out_boxes[i][0][2], out_boxes[i][0][0]))
            if any(grow([out_parts[i] for i in ch], [out_mats[i] for i in ch], [out_boxes[i] for i in ch], M)            # lowest border first
                   for ch in (pre[k:k + ctx] for k in range(0, len(pre), ctx))):
                fails = 0; continue
        fc = free.reshape(h2, 2, w2, 2, d2, 2).any((1, 3, 5))
        if dead is None: dead = np.zeros_like(fc)
        fc &= ~dead; ys = np.nonzero(fc.any((1, 2)))[0]
        if len(ys) == 0: break                                                         # every empty cell tried
        if attach and out_parts and (ys.min() > 0 or n_comp >= max_roots): break     # nothing left that may start a new piece
        band = np.zeros_like(fc); band[ys.min():ys.min() + 2] = fc[ys.min():ys.min() + 2]          # lowest empty layers first
        # ---- root: best (part, sub-cell) by coverage among the model's top parts
        ci = pick(M["root_logits"][0], band.ravel()); cell = np.array(np.unravel_index(ci, (h2, w2, d2)))
        s = model.decode(torch.zeros(1, 0, model.d, device=device), M, torch.zeros(1, 1, 3, dtype=torch.long, device=device), torch.zeros(1, 1, dtype=torch.bool, device=device), torch.zeros(1, 0, dtype=torch.bool, device=device))
        h0 = model.root_context(s[:, 0], M, torch.tensor(cell[None], device=device))[0]; rc = int(model.root_rot(h0).argmax())
        ctx0 = torch.cat([h0, model.no_parent, model.pconn_emb.weight[NCONN]])
        best = None
        rots = [rc] + [r_ for r_ in (0, 1, 2, 3) if r_ != rc]                           # model's rotation, then yaw-only classes
        for part in torch.topk(model.h_part(ctx0), 2 * k_parts).indices.tolist():
            for rr in rots[:2]:
                for o in np.ndindex(2, 2, 2):
                    ay, ax, az = cell * 2 + np.array(o)
                    if not free[ay, ax, az]: continue
                    Mp = root_pose(vocab.stems[part], rr, (ax, ay, az), grid); r = realise_check(part, Mp)
                    if r and (best is None or r[2] > best[3]): best = (part, Mp, r[0], r[2], r[1])
        if best is None: dead[tuple(cell)] = True; fails += 1; continue
        fails = 0; n_comp += 1; n_ground += int(cell[0] == 0)
        part, Mp, bx, _, cells = best
        col = colour_of(cells, vocab.colors[int(model.h_color(torch.cat([ctx0, model.part_emb.weight[part], model.cconn_emb.weight[NCONN]])).argmax())])
        place(part, col, Mp, bx, cells)
        grow([out_parts[-1]], [Mp], [bx], M)
    iou = (build & target).sum() / max(1, (build | target).sum())
    ldr = "".join(f"1 {p.color} " + " ".join(f"{v:.6g}" for v in [*Mm[:3, 3], *Mm[:3, :3].ravel()]) + f" {st}.dat\n" for p, Mm, st in zip(out_parts, out_mats, out_stems))
    return {"n": len(out_parts), "iou": float(iou), "filled": int((build & target).sum()), "target": int(target.sum()), "components": n_comp,
            "ground_roots": n_ground, "rejected": rejected, "ldr": ldr}
