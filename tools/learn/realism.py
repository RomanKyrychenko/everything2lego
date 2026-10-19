"""Direct-LDraw realism classifier: real-OMR vs ours (sampler) +
degraded-real.  Features are hand-crafted from the LDraw parse (no
rendering), so building a dataset takes seconds and the classifier
can't fall back on vocab / palette / silhouette shortcuts the way an
image-based classifier can.

    python tools/learn/realism.py build    # extract features from OMR + ours
    python tools/learn/realism.py train    # train the MLP
    python tools/learn/realism.py score path/to/model.ldr

Degraded-real negatives: random part replacement, colour shuffle,
position swap.  Makes the task harder than real-vs-ours and forces
the classifier to learn STRUCTURAL realism (connectivity, pair
patterns, size distribution), not just "uses different parts than
OMR".

Output: $LEGO_DATA/analysis/realism/{features.npz,model.pt}."""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tools" / "set_study"))
sys.path.insert(0, str(ROOT / "tools" / "partsdb"))

DATA = Path(os.environ.get("LEGO_DATA", "/Volumes/PRO-G40/lego_data"))
OMR = DATA / "omr" / "files"
OUT = DATA / "analysis" / "realism"
FEATURES_NPZ = OUT / "features.npz"
MODEL_PT = OUT / "model.pt"
SET_KINDS = DATA / "analysis" / "set_kinds.csv"

# Coarse part-category buckets: ldraw name prefix -> bucket index.
# Covers the common studs-up building parts; others fall into "other".
CATEGORIES = ["brick", "plate", "tile", "slope", "round", "tech", "other"]
_CAT_RE = {
    "brick": re.compile(r"^(300[0-9]|301[0-9]|30[2-9]|3062b|4733|6111|87087|87552|88393|98100|30144)"),
    "plate": re.compile(r"^(302[0-5]|302[7-9]|30[3-6][0-9]|3666|3710|3794|4477|6141|6177|11212|87079|4032|60479|92593|41539|2420|87580|63326|85861|4274|4624|11458)"),
    "tile":  re.compile(r"^(306[0-9]|3068b|3069b|3070b|6636|2412|2431|63864|4150|4162|25269|98138|11203|11477|14719)"),
    "slope": re.compile(r"^(3037|3038|3039|3040|3044|3045|3049|3297|4286|4287|4854|6091|6227|11477|50746|52501|54200|60481|85984|87087|11477|18653|22888)"),
    "round": re.compile(r"^(4073|4274|4740|6141|87087|3062b|6116|87580|30157|30208|44375|49668|79196|98100|87079)"),
    "tech":  re.compile(r"^(3700|3701|3702|3703|3704|3705|3706|3707|3708|3709|32028|32034|32039|32073|32140|32316|32526|41677)"),
}


def _category(ldraw: str) -> int:
    for i, cat in enumerate(CATEGORIES[:-1]):
        if _CAT_RE[cat].match(ldraw):
            return i
    return len(CATEGORIES) - 1


# Top-30 LEGO colour codes (common OMR + ours set).  Everything else -> "other".
COLOUR_BINS = [0, 1, 2, 3, 4, 5, 7, 8, 10, 11, 14, 15, 19, 25, 26, 27, 28, 70, 71, 72, 74, 77, 78, 82, 84, 85, 96, 272, 308, 484]
_COLOUR_IX = {c: i for i, c in enumerate(COLOUR_BINS)}


def _colour_bin(c: int) -> int:
    return _COLOUR_IX.get(c, len(COLOUR_BINS))


# Size buckets by part footprint volume (studs² × plates).
def _size_bin(ldraw: str, bbox_cache: dict) -> int:
    bb = bbox_cache.get(ldraw)
    if bb is None:
        return 4
    (lo, hi) = bb
    w = max(1, round((hi[0] - lo[0]) / 20))
    d = max(1, round((hi[2] - lo[2]) / 20))
    h = max(1, round((hi[1] - lo[1]) / 8))
    vol = w * d * h
    if vol <= 1:
        return 0
    if vol <= 3:
        return 1
    if vol <= 6:
        return 2
    if vol <= 24:
        return 3
    return 4


def _is_snot(M: np.ndarray) -> bool:
    """Y basis vector not aligned with world ±Y means studs are not up."""
    y_axis = M[:3, :3] / (np.linalg.norm(M[:3, :3], axis=0) + 1e-9)
    up = y_axis @ np.array([0.0, -1.0, 0.0])
    return abs(up[1]) < 0.9


def extract(path: Path) -> np.ndarray:
    """Hand-crafted feature vector for one LDraw build.  Returns float32
    numpy array; see FEATURE_NAMES for layout."""
    from ldraw_read import flatten, part_bbox
    try:
        parts = flatten(path)
    except Exception:
        return None
    if not parts:
        return None
    # pre-cache bboxes
    bbox_cache = {}
    for p in parts:
        if p.name not in bbox_cache:
            try:
                bbox_cache[p.name] = part_bbox(p.name)
            except Exception:
                bbox_cache[p.name] = None
    # per-part features
    cat = np.zeros(len(CATEGORIES), dtype=np.float32)
    col = np.zeros(len(COLOUR_BINS) + 1, dtype=np.float32)
    sz = np.zeros(5, dtype=np.float32)
    snot = 0
    centres = []
    for p in parts:
        cat[_category(p.name)] += 1
        col[_colour_bin(p.colour)] += 1
        sz[_size_bin(p.name, bbox_cache)] += 1
        if _is_snot(p.M):
            snot += 1
        centres.append(p.M[:3, 3])
    n = len(parts)
    cat /= n; col /= n; sz /= n
    centres = np.array(centres)
    bbox_min = centres.min(0)
    bbox_max = centres.max(0)
    extent = bbox_max - bbox_min
    volume = max(1.0, extent[0] * extent[1] * extent[2])
    # global stats
    unique = len(set(p.name for p in parts))
    stats = np.array([
        n / 1000.0,                                           # total parts (scaled)
        unique / max(1, n),                                   # vocab diversity
        snot / n,                                             # SNOT fraction
        float(np.mean([_size_bin(p.name, bbox_cache) for p in parts])),  # mean size bucket
        extent[0] / 100.0, extent[1] / 100.0, extent[2] / 100.0,          # bbox LDU/100
        extent[1] / max(1.0, extent[0]),                       # height / width aspect
        n / (volume / 1e6),                                    # parts per bbox volume (density)
    ], dtype=np.float32)
    return np.concatenate([cat, col, sz, stats])


FEATURE_LEN = len(CATEGORIES) + len(COLOUR_BINS) + 1 + 5 + 9


def _degrade(parts, kind: str, rng: random.Random):
    """One in-place degradation of a parts list (sampler/OMR-ish)."""
    if not parts:
        return parts
    out = list(parts)
    if kind == "replace":
        other = out[rng.randrange(len(out))]
        n = max(1, len(out) // 5)                             # 20% replaced
        for _ in range(n):
            i = rng.randrange(len(out))
            # replace with a random other part's name (keep position / colour)
            j = rng.randrange(len(out))
            if j == i:
                continue
            orig = out[i]
            out[i] = type(orig)(out[j].name, orig.colour, orig.M, orig.step, orig.path)
    elif kind == "colour":
        colours = [p.colour for p in out]
        rng.shuffle(colours)
        for i, p in enumerate(out):
            out[i] = type(p)(p.name, colours[i], p.M, p.step, p.path)
    elif kind == "position":
        matrices = [p.M for p in out]
        rng.shuffle(matrices)
        for i, p in enumerate(out):
            out[i] = type(p)(p.name, p.colour, matrices[i], p.step, p.path)
    return out


def _extract_degraded(parts, kind: str, rng: random.Random) -> np.ndarray | None:
    """Same features as extract() but on a degraded parts list in memory."""
    from ldraw_read import part_bbox
    parts = _degrade(parts, kind, rng)
    bbox_cache = {}
    for p in parts:
        if p.name not in bbox_cache:
            try: bbox_cache[p.name] = part_bbox(p.name)
            except Exception: bbox_cache[p.name] = None
    cat = np.zeros(len(CATEGORIES), dtype=np.float32)
    col = np.zeros(len(COLOUR_BINS) + 1, dtype=np.float32)
    sz = np.zeros(5, dtype=np.float32)
    snot = 0
    centres = []
    for p in parts:
        cat[_category(p.name)] += 1
        col[_colour_bin(p.colour)] += 1
        sz[_size_bin(p.name, bbox_cache)] += 1
        if _is_snot(p.M):
            snot += 1
        centres.append(p.M[:3, 3])
    n = len(parts)
    cat /= n; col /= n; sz /= n
    centres = np.array(centres)
    bbox_min = centres.min(0); bbox_max = centres.max(0)
    extent = bbox_max - bbox_min
    volume = max(1.0, extent[0] * extent[1] * extent[2])
    unique = len(set(p.name for p in parts))
    stats = np.array([
        n / 1000.0, unique / max(1, n), snot / n,
        float(np.mean([_size_bin(p.name, bbox_cache) for p in parts])),
        extent[0] / 100.0, extent[1] / 100.0, extent[2] / 100.0,
        extent[1] / max(1.0, extent[0]),
        n / (volume / 1e6),
    ], dtype=np.float32)
    return np.concatenate([cat, col, sz, stats])


def _building_omr_ids() -> set:
    if not SET_KINDS.exists():
        return set()
    import csv
    out = set()
    with SET_KINDS.open() as f:
        for row in csv.DictReader(f):
            if row.get("kind") == "building":
                out.add(row["id"])
    return out


def _ours_meshes() -> list[Path]:
    out = list((ROOT / "examples").glob("*.glb"))
    return sorted(set(out))


def _generate_ours(mesh: Path, seed: int, out_dir: Path, iters: int = 20000) -> Path | None:
    sess = out_dir / f"{mesh.stem}_s{seed}"
    sess.mkdir(parents=True, exist_ok=True)
    ldr = sess / "model.ldr"
    if ldr.exists():
        return ldr
    cmd = [sys.executable, str(ROOT / "tools" / "sampler_demo.py"),
           str(mesh), str(sess), "--iters", str(iters), "--seed", str(seed),
           "--no-snot-decorate"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0 or not ldr.exists():
        return None
    return ldr


def cmd_build(a) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    bids = _building_omr_ids()
    omr = []
    for f in sorted(OMR.glob("*.mpd")):
        if f.stem.split("_")[0] in bids:
            omr.append(f)
    random.Random(a.seed).shuffle(omr)
    omr = omr[:a.n_real]
    print(f"real: parsing {len(omr)} OMR building sets", flush=True)
    X_real = []
    sample_parts = []                                         # keep raw parts for degradation
    for i, f in enumerate(omr):
        v = extract(f)
        if v is None:
            continue
        X_real.append(v)
        if len(sample_parts) < a.n_degraded:
            from ldraw_read import flatten
            try:
                sample_parts.append(flatten(f))
            except Exception:
                pass
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(omr)}", flush=True)
    print(f"real: {len(X_real)} features", flush=True)

    # Ours
    meshes = _ours_meshes()
    print(f"ours: {len(meshes)} meshes available, generating {a.n_ours} builds", flush=True)
    X_ours = []
    scratch = OUT / "_ours_scratch"
    for i in range(a.n_ours):
        mesh = meshes[i % len(meshes)]
        seed = a.seed * 1000 + i
        ldr = _generate_ours(mesh, seed, scratch, iters=a.iters)
        if ldr is None:
            continue
        v = extract(ldr)
        if v is not None:
            X_ours.append(v)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{a.n_ours}", flush=True)
    print(f"ours: {len(X_ours)} features", flush=True)

    # Degraded-real
    rng = random.Random(a.seed + 1)
    X_deg = []
    kinds = ["replace", "colour", "position"]
    for i, parts in enumerate(sample_parts):
        kind = kinds[i % len(kinds)]
        v = _extract_degraded(parts, kind, rng)
        if v is not None:
            X_deg.append(v)
        if (i + 1) % 50 == 0:
            print(f"  degraded {i+1}/{len(sample_parts)}", flush=True)
    print(f"degraded-real: {len(X_deg)} features", flush=True)

    np.savez_compressed(FEATURES_NPZ,
                        X_real=np.stack(X_real), X_ours=np.stack(X_ours),
                        X_deg=np.stack(X_deg))
    print(f"saved -> {FEATURES_NPZ}", flush=True)


def cmd_train(a) -> None:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    d = np.load(FEATURES_NPZ)
    Xr, Xo, Xd = d["X_real"], d["X_ours"], d["X_deg"]
    # labels: real=1, everything else=0
    X = np.concatenate([Xr, Xo, Xd]); y = np.concatenate([
        np.ones(len(Xr)), np.zeros(len(Xo)), np.zeros(len(Xd))])
    # per-source tag so we can report per-group accuracy
    src = np.array(["real"]*len(Xr) + ["ours"]*len(Xo) + ["deg"]*len(Xd))
    rng = np.random.default_rng(a.seed)
    idx = rng.permutation(len(X))
    X, y, src = X[idx], y[idx], src[idx]
    val_n = int(len(X) * a.val_frac)
    Xv, yv, sv = X[:val_n], y[:val_n], src[:val_n]
    Xt, yt = X[val_n:], y[val_n:]
    print(f"train={len(Xt)}, val={len(Xv)}  (real {sum(src=='real')}, ours {sum(src=='ours')}, deg {sum(src=='deg')})", flush=True)

    device = torch.device(a.device)
    net = nn.Sequential(
        nn.Linear(FEATURE_LEN, 64), nn.ReLU(),
        nn.Linear(64, 32), nn.ReLU(),
        nn.Linear(32, 2),
    ).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    Xt_t = torch.tensor(Xt, dtype=torch.float32, device=device)
    yt_t = torch.tensor(yt, dtype=torch.long, device=device)
    Xv_t = torch.tensor(Xv, dtype=torch.float32, device=device)
    yv_t = torch.tensor(yv, dtype=torch.long, device=device)
    best = -1.0
    for ep in range(1, a.epochs + 1):
        net.train()
        # mini-batch shuffle
        perm = torch.randperm(len(Xt_t), device=device)
        tr_loss = tr_hit = tr_n = 0
        for b in range(0, len(perm), a.bs):
            ii = perm[b:b+a.bs]
            logits = net(Xt_t[ii])
            loss = F.cross_entropy(logits, yt_t[ii])
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item() * len(ii)
            tr_hit += (logits.argmax(-1) == yt_t[ii]).sum().item()
            tr_n += len(ii)
        net.eval()
        with torch.no_grad():
            vl = net(Xv_t)
            vp = vl.argmax(-1)
            va = (vp == yv_t).float().mean().item()
            # per source
            per = {}
            for s in ("real", "ours", "deg"):
                m = sv == s
                if m.any():
                    per[s] = (vp.cpu().numpy()[m] == yv[m]).mean()
        tag = "  ".join(f"{k}={per[k]:.3f}" for k in ("real", "ours", "deg") if k in per)
        print(f"ep {ep:02d}  tr loss {tr_loss/max(1,tr_n):.4f} acc {tr_hit/max(1,tr_n):.3f}  val acc {va:.3f}  [{tag}]", flush=True)
        if va > best:
            best = va
            import torch as _t
            _t.save({"state_dict": net.state_dict(), "val_acc": va, "epoch": ep,
                      "feature_len": FEATURE_LEN}, MODEL_PT)
    print(f"best val acc {best:.3f}  saved -> {MODEL_PT}", flush=True)


def cmd_score(a) -> None:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    v = extract(a.path)
    if v is None:
        raise SystemExit(f"could not extract features from {a.path}")
    ckpt = torch.load(MODEL_PT, map_location="cpu", weights_only=True)
    net = nn.Sequential(
        nn.Linear(ckpt["feature_len"], 64), nn.ReLU(),
        nn.Linear(64, 32), nn.ReLU(),
        nn.Linear(32, 2),
    )
    net.load_state_dict(ckpt["state_dict"]); net.eval()
    with torch.no_grad():
        logits = net(torch.tensor(v).unsqueeze(0))
        p_real = F.softmax(logits, dim=-1)[0, 1].item()
    print(f"{a.path}: realism = {p_real:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--n-real", type=int, default=300)
    b.add_argument("--n-ours", type=int, default=50)
    b.add_argument("--n-degraded", type=int, default=200)
    b.add_argument("--iters", type=int, default=20000)
    b.add_argument("--seed", type=int, default=0)
    b.set_defaults(func=cmd_build)

    t = sub.add_parser("train")
    t.add_argument("--epochs", type=int, default=40)
    t.add_argument("--bs", type=int, default=32)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--val-frac", type=float, default=0.2)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default="cpu")
    t.set_defaults(func=cmd_train)

    s = sub.add_parser("score")
    s.add_argument("path", type=Path)
    s.set_defaults(func=cmd_score)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
