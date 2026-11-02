"""Label the target's cells with the LEGO part category a feature of the mesh
asks for -- a window, a door, a wheel, a fence, plants -- where the mesh only
paints it or barely models it, for the sampler's --semantic.

    python tools/semantic_labels.py mesh.glb --studs 32 --out labels.npz [--sheet sheet.png]

  - candidate regions at every scale, as selective search groups an image: the
    leaves are connected faces of one texture-colour cluster (k-means in Lab over
    each face's texel, smoothed over neighbours) and one facing axis; adjacent
    regions merge most similar first, and each merge of --min-area..--max-area
    stud^2 is a candidate (a window pane, the pane with its frame, a whole wheel);
  - each candidate is drawn along its mean normal (a painter's-algorithm render
    in the texture's colours, its surroundings dimmed) and labelled zero-shot by
    CLIP against semantic classes (window, door, wheel, fence, ladder, plants:
    their Rebrickable category in the part vocabulary) and plain ones (wall,
    roof, stone, ground, grass, water, waterfall, wood, chimney: nothing special);
  - kept when the best class is semantic, p >= --p-min (0.7), it leads the
    runner-up by >= --margin (0.2), and drawn ALONE (no surroundings) the region
    is still that class first with p >= --p-alone (0.3) -- with context alone,
    every beam and sill beside a window is "a window" (CLIP reads the context);
    calibrated on the watermill (32 studs, --sheet): real windows p 0.78-1.00,
    alone 0.50-0.91; slivers beside them alone <= 0.25; a fence post p 0.58;
  - each kept region's surface is sampled, the samples go to the nearest wanted
    cell of the sampler's own target (sampler_demo.target_from_mesh, same
    --studs), and the showing wanted cells in the box they span take the label
    (smaller regions first): `cell_cat` (H, X, Z) int8 into `categories`, -1 none.

CLIP (laion/CLIP-ViT-B-32-laion2B-s34B-b79K) is read from the local HF cache only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]

from everything2lego.colors import _rgb_to_lab  # noqa: E402

CLIP_ID = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
PLATE = 0.4                                      # a plate's height in studs
# (class, what CLIP is shown, Rebrickable category; None = a plain surface, no part asked for)
CLASSES = [("window", "a window", "Windows and Doors"), ("door", "a door", "Windows and Doors"),
           ("wheel", "a wheel", "Wheels and Tyres"), ("fence", "a fence", "Bars, Ladders and Fences"),
           ("ladder", "a ladder", "Bars, Ladders and Fences"), ("plant", "plants with leaves", "Plants & Trees"),
           ("wall", "a plain wall", None), ("roof", "a tiled roof", None), ("stone", "stones and rocks", None),
           ("ground", "the ground", None), ("grass", "grass", None), ("water", "water", None), ("waterfall", "a waterfall", None),
           ("wood", "wooden planks", None), ("chimney", "a chimney", None)]
TEMPLATES = ("a 3D render of {}.", "a close-up of {} on a model.", "{}")


def face_colours(mesh) -> np.ndarray:
    """(F, 3) uint8: the texture (or vertex colour) at each face's centre."""
    from everything2lego.voxelize import _face_color_lookup
    return _face_color_lookup(mesh, mesh.triangles_center, np.arange(len(mesh.faces)))


def regions(mesh, grid_T: np.ndarray, k: int, min_area: float, max_area: float, seed: int = 0) -> list[dict]:
    """Candidate regions [{id, faces, area (stud^2)}] at every scale, as selective search (Uijlings et al. 2013)
    groups an image: the leaves are connected faces of one colour cluster and one facing axis; adjacent regions merge
    most similar first (close mean colour, both small), and each merge of `min_area`..`max_area` stud^2 that grew its
    larger part by half is a candidate -- a window pane, the pane with its frame, a whole wheel of many spokes."""
    import heapq
    from scipy.sparse import csr_matrix
    from sklearn.cluster import KMeans
    from segment_mesh import _components
    lab = _rgb_to_lab(face_colours(mesh).astype(np.float32) / 255.0)
    e = mesh.face_adjacency
    A = csr_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(len(lab),) * 2)
    deg = np.asarray(A.sum(1)).ravel() + 1
    for _ in range(2):                                   # texel noise: mean with the neighbours, twice
        lab = (A @ lab + lab) / deg[:, None]
    cl = KMeans(n_clusters=k, n_init=4, random_state=seed).fit_predict(lab)
    n = mesh.face_normals
    axis = np.argmax(np.abs(n), 1) * 2 + (n[np.arange(len(n)), np.argmax(np.abs(n), 1)] > 0)
    comp = np.unique(_components(mesh, cl * 6 + axis), return_inverse=True)[1]
    tri = stud_space(mesh.triangles.reshape(-1, 3), grid_T).reshape(-1, 3, 3)
    area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) / 2
    m = comp.max() + 1
    ar = np.bincount(comp, area, m)
    col = np.stack([np.bincount(comp, lab[:, i] * area, m) for i in range(3)], 1) / np.maximum(ar, 1e-12)[:, None]
    leaves = {r: [r] for r in range(m)}
    nb = {r: set() for r in range(m)}
    for a, b in zip(comp[e[:, 0]], comp[e[:, 1]]):
        if a != b:
            nb[a].add(b); nb[b].add(a)
    total = ar.sum()
    sim = lambda a, b: np.exp(-np.linalg.norm(col[a] - col[b]) / 20) + 1 - (ar[a] + ar[b]) / total
    heap = [(-sim(a, b), a, b) for a in range(m) for b in nb[a] if a < b]
    heapq.heapify(heap)
    ar, col = list(ar), list(col)
    alive, out = set(range(m)), []
    for r in range(m):
        if min_area <= ar[r] <= max_area:
            out.append(np.flatnonzero(comp == r))
    while heap:
        _, a, b = heapq.heappop(heap)
        if a not in alive or b not in alive:
            continue
        r = len(ar)
        ar.append(ar[a] + ar[b]); col.append((col[a] * ar[a] + col[b] * ar[b]) / ar[r])
        leaves[r] = leaves.pop(a) + leaves.pop(b)
        nb[r] = (nb.pop(a) | nb.pop(b)) - {a, b}
        alive -= {a, b}; alive.add(r)
        for c in nb[r]:
            nb[c] -= {a, b}; nb[c].add(r)
            heapq.heappush(heap, (-sim(r, c), r, c))
        if min_area <= ar[r] <= max_area and ar[r] >= 1.5 * max(ar[a], ar[b]):
            out.append(np.flatnonzero(np.isin(comp, leaves[r])))
    return [{"id": i, "faces": f, "area": float(area[f].sum())} for i, f in enumerate(out)]


def stud_space(p: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Mesh points -> grid units (x stud, y plate, z stud) -> studs on every axis (isotropic, for drawing and areas)."""
    g = p @ T[:3, :3].T + T[:3, 3]
    return g * np.array([1.0, PLATE, 1.0])


def render(P: np.ndarray, faces: np.ndarray, fcol: np.ndarray, fn: np.ndarray, region: np.ndarray, normal: np.ndarray,
           size: int = 224, ctx: float = 1.3, outline: bool = False, alone: bool = False):
    """The region seen along -normal (orthographic, stud-space vertices P), framed with `ctx` times its extent of
    surroundings, the surroundings dimmed; faces painted far to near.  `alone`: the region by itself, nothing around it."""
    from PIL import Image, ImageDraw
    f = -np.asarray(normal, float)
    up = np.array([0.0, 1.0, 0.0]) if abs(f[1]) < 0.9 else np.array([0.0, 0.0, -1.0])
    r = np.cross(up, f); r /= np.linalg.norm(r); u = np.cross(f, r)
    q = P @ np.stack([r, u, f], 1)                       # (V, 3): right, up, depth
    qr = q[faces[region]].reshape(-1, 3)
    c = (qr.min(0) + qr.max(0)) / 2
    half = max(np.ptp(qr[:, 0]), np.ptp(qr[:, 1]), 1.0) * ctx / 2 + 0.5
    cen = q[faces].mean(1)
    sel = (np.abs(cen[:, 0] - c[0]) < half * 1.3) & (np.abs(cen[:, 1] - c[1]) < half * 1.3) & (fn @ f < 0.05)
    inreg = np.zeros(len(faces), bool); inreg[region] = True
    img = Image.new("RGB", (size, size), (255, 255, 255)); dr = ImageDraw.Draw(img)
    s = size / (2 * half)
    light = np.array([0.3, 0.8, -0.5]); light /= np.linalg.norm(light)
    if alone:
        sel &= inreg
    for i in np.flatnonzero(sel)[np.argsort(-cen[sel, 2])]:           # far first (depth along f, larger is farther)
        k = (0.75 + 0.25 * abs(fn[i] @ light)) * (1.0 if inreg[i] else 0.4)
        xy = [(size / 2 + (q[v, 0] - c[0]) * s, size / 2 - (q[v, 1] - c[1]) * s) for v in faces[i]]
        dr.polygon(xy, fill=tuple(int(min(255, x * k)) for x in fcol[i]))
    if outline:
        lo, hi = qr.min(0), qr.max(0)
        dr.rectangle([size / 2 + (lo[0] - c[0]) * s, size / 2 - (hi[1] - c[1]) * s,
                      size / 2 + (hi[0] - c[0]) * s, size / 2 - (lo[1] - c[1]) * s], outline=(255, 0, 255), width=2)
    return img


@lru_cache(maxsize=1)
def load_clip():
    """(CLIPModel, CLIPProcessor) from the local cache.  The cache holds only open_clip's weights of CLIP_ID (no
    transformers config or tokenizer), so they are mapped onto a transformers CLIPModel of the same ViT-B/32 shape;
    the tokenizer is CLIP's BPE (the same 49408 tokens in every CLIP), from any cached copy of it."""
    os.environ.setdefault("HF_HOME", "/Volumes/PRO-G40/huggingface")
    import torch
    from huggingface_hub import try_to_load_from_cache
    from safetensors.torch import load_file
    from transformers import CLIPConfig, CLIPImageProcessor, CLIPModel, CLIPProcessor, CLIPTokenizer
    sd = load_file(try_to_load_from_cache(CLIP_ID, "open_clip_model.safetensors"))
    cfg = CLIPConfig(text_config=dict(hidden_size=512, intermediate_size=2048, num_attention_heads=8, num_hidden_layers=12,
                                      hidden_act="gelu"),
                     vision_config=dict(hidden_size=768, intermediate_size=3072, num_attention_heads=12, num_hidden_layers=12,
                                        patch_size=32, image_size=224, hidden_act="gelu"), projection_dim=512)
    out = {"logit_scale": sd["logit_scale"], "text_projection.weight": sd["text_projection"].T,
           "visual_projection.weight": sd["visual.proj"].T,
           "text_model.embeddings.token_embedding.weight": sd["token_embedding.weight"],
           "text_model.embeddings.position_embedding.weight": sd["positional_embedding"],
           "text_model.final_layer_norm.weight": sd["ln_final.weight"], "text_model.final_layer_norm.bias": sd["ln_final.bias"],
           "vision_model.embeddings.class_embedding": sd["visual.class_embedding"],
           "vision_model.embeddings.position_embedding.weight": sd["visual.positional_embedding"],
           "vision_model.embeddings.patch_embedding.weight": sd["visual.conv1.weight"]}
    for a, b in (("ln_pre", "pre_layrnorm"), ("ln_post", "post_layernorm")):
        for p in ("weight", "bias"):
            out[f"vision_model.{b}.{p}"] = sd[f"visual.{a}.{p}"]
    for src, dst in (("transformer", "text_model"), ("visual.transformer", "vision_model")):
        for i in range(12):
            s, d = f"{src}.resblocks.{i}.", f"{dst}.encoder.layers.{i}."
            for p in ("weight", "bias"):
                for q, w in zip(("q", "k", "v"), sd[s + f"attn.in_proj_{p}"].chunk(3)):
                    out[d + f"self_attn.{q}_proj.{p}"] = w
                for a, b in (("attn.out_proj", "self_attn.out_proj"), ("ln_1", "layer_norm1"), ("ln_2", "layer_norm2"),
                             ("mlp.c_fc", "mlp.fc1"), ("mlp.c_proj", "mlp.fc2")):
                    out[d + f"{b}.{p}"] = sd[s + f"{a}.{p}"]
    model = CLIPModel(cfg)
    missing, unexpected = model.load_state_dict(out, strict=False)
    assert not unexpected and all("position_ids" in k for k in missing), (missing, unexpected)
    hub = Path(os.environ["HF_HOME"]) / "hub"
    vocab = next(c.parent for pat in ("models--*/snapshots/*/tokenizer_config.json", "models--*/snapshots/*/*/tokenizer/tokenizer_config.json")
                 for c in hub.glob(pat) if not c.name.startswith("._") and (c.parent / "merges.txt").exists()
                 and "CLIPTokenizer" in c.read_text(errors="ignore"))
    return model.eval(), CLIPProcessor(image_processor=CLIPImageProcessor(), tokenizer=CLIPTokenizer.from_pretrained(vocab))


def clip_probs(images: list) -> np.ndarray:
    """(n_images, n_classes) softmax over CLASSES (prompt ensemble)."""
    import torch
    model, proc = load_clip()
    with torch.no_grad():
        txt = [t.format(p) for _, p, _ in CLASSES for t in TEMPLATES]
        te = model.text_projection(model.text_model(**proc(text=txt, return_tensors="pt", padding=True)).pooler_output)
        te = te / te.norm(dim=-1, keepdim=True)
        te = te.reshape(len(CLASSES), len(TEMPLATES), -1).mean(1)
        te = te / te.norm(dim=-1, keepdim=True)
        out = []
        for b in range(0, len(images), 32):
            ie = model.visual_projection(model.vision_model(**proc(images=images[b:b + 32], return_tensors="pt")).pooler_output)
            ie = ie / ie.norm(dim=-1, keepdim=True)
            out.append((model.logit_scale.exp() * ie @ te.T).softmax(-1).numpy())
    return np.concatenate(out) if out else np.zeros((0, len(CLASSES)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh"); ap.add_argument("--studs", type=int, default=32)
    ap.add_argument("--out", required=True); ap.add_argument("--sheet", default=None)
    ap.add_argument("--k", type=int, default=16, help="colour clusters (k-means in Lab)")
    ap.add_argument("--min-area", type=float, default=0.5, help="smallest candidate region, stud^2")
    ap.add_argument("--max-area", type=float, default=48.0, help="largest candidate region, stud^2")
    ap.add_argument("--p-min", type=float, default=0.7, help="keep a semantic label at this CLIP probability or more")
    ap.add_argument("--margin", type=float, default=0.2, help="... leading the runner-up class by this much")
    ap.add_argument("--p-alone", type=float, default=0.3, help="... and, the region seen alone, that class still first with this probability")
    a = ap.parse_args()
    from everything2lego.sampler import vocabulary
    from everything2lego.semantic_v2.mesh_target import MeshTarget
    from sampler_demo import target_from_mesh
    t = target_from_mesh(a.mesh, a.studs, shell=2)       # the sampler's own target: its cells, as --studs builds them
    mt = MeshTarget.from_path(a.mesh, max_studs=a.studs, n_samples=40000, min_island_cells=8)
    mesh, T = mt.mesh, mt.mesh_to_voxel.astype(np.float64)
    regs = regions(mesh, T, a.k, a.min_area, a.max_area)
    print(f"{len(regs)} candidate regions of {a.min_area}..{a.max_area} stud^2 ({len(mesh.faces)} faces)", flush=True)
    P, fcol = stud_space(mesh.vertices, T), face_colours(mesh)
    tri = P[mesh.faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    fa = np.linalg.norm(fn, axis=1)
    fn /= fa[:, None] + 1e-12
    for r in regs:                                       # the way it faces, area-weighted, in stud space
        nm = (fn[r["faces"]] * fa[r["faces"], None]).sum(0); r["normal"] = nm / (np.linalg.norm(nm) + 1e-12)
    prob = clip_probs([render(P, mesh.faces, fcol, fn, r["faces"], r["normal"]) for r in regs])
    present = {v.category for v in vocabulary()}
    cats = sorted({c for _, _, c in CLASSES if c in present})
    sem = np.array([c in cats for _, _, c in CLASSES])
    for r, p in zip(regs, prob):
        o = np.argsort(-p)
        r.update(cls=CLASSES[o[0]][0], k=int(o[0]), p=float(p[o[0]]), margin=float(p[o[0]] - p[o[1]]), alone=0.0,
                 p_sem=float(p[sem].max()), sem_cls=CLASSES[int(np.flatnonzero(sem)[np.argmax(p[sem])])][0])
        r["keep"] = bool(sem[o[0]] and r["p"] >= a.p_min and r["margin"] >= a.margin)
    # the region itself must be the feature, not a sliver beside it: seen alone, it is still most likely that class
    cand = [r for r in regs if r["keep"]]
    alone = clip_probs([render(P, mesh.faces, fcol, fn, r["faces"], r["normal"], alone=True) for r in cand])
    for r, g in zip(cand, alone):
        r["alone"] = float(g[r["k"]])
        r["keep"] = bool(g.argmax() == r["k"] and r["alone"] >= a.p_alone)
    # kept regions -> the sampler's cells: surface samples (~16 per stud^2) to the nearest wanted cell within a cell,
    # then the box those cells span, its showing wanted cells (a window's pane and frame, a wheel's disc, not just the
    # patches the region's texels hit); smaller regions first, a cell keeps its first label
    import trimesh
    from scipy.spatial import cKDTree
    H, X, Z = t.want.shape
    wc = np.argwhere(t.want)                             # (y, x, z)
    tree = cKDTree(wc[:, [1, 0, 2]] + 0.5)
    show = t.want & t.visible
    cell_cat = np.full((H, X, Z), -1, np.int8)
    for r in sorted((r for r in regs if r["keep"]), key=lambda r: r["area"]):
        sub = trimesh.Trimesh(mesh.vertices, mesh.faces[r["faces"]], process=False)
        pts, _ = trimesh.sample.sample_surface_even(sub, max(64, int(16 * r["area"])), seed=0)
        d, j = tree.query(pts @ T[:3, :3].T + T[:3, 3], distance_upper_bound=1.0)
        hit = wc[j[np.isfinite(d)]]
        if not len(hit):
            continue
        lo, hi = hit.min(0), hit.max(0) + 1
        box = np.zeros_like(show); box[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = True
        box &= show & (cell_cat < 0)
        cell_cat[box] = cats.index(dict((n, c) for n, _, c in CLASSES)[r["cls"]])
        r["cells"] = int(box.sum())
    table = [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k not in ("faces", "normal")}
             for r in regs]
    np.savez_compressed(a.out, cell_cat=cell_cat, categories=np.array(cats), regions=json.dumps(table))
    for r in sorted((r for r in regs if r["keep"]), key=lambda r: (r["cls"], -r["area"])):
        print(f"  region {r['id']:>3}  {r['area']:6.1f} stud^2  {r['cls']:>7} p {r['p']:.2f} margin {r['margin']:.2f}"
              f" alone {r['alone']:.2f}  -> {r.get('cells', 0)} cells")
    print(f"-> {a.out}: cells per category " + ", ".join(f"{c} {int((cell_cat == i).sum())}" for i, c in enumerate(cats)))
    if a.sheet:
        from PIL import Image, ImageDraw
        show = sorted(regs, key=lambda r: (not r["keep"], r["alone"] == 0, -r["p_sem"]))[:96]
        S, cols = 160, 8
        sheet = Image.new("RGB", (cols * S, -(-len(show) // cols) * (S + 28)), "white")
        for i, r in enumerate(show):
            im = render(P, mesh.faces, fcol, fn, r["faces"], r["normal"], size=S, outline=True)
            x, y = (i % cols) * S, (i // cols) * (S + 28)
            sheet.paste(im, (x, y))
            dr = ImageDraw.Draw(sheet)
            dr.text((x + 3, y + S + 1), f"#{r['id']} {r['cls']} {r['p']:.2f} m{r['margin']:.2f} a{r['alone']:.2f}", fill="green" if r["keep"] else "black")
            dr.text((x + 3, y + S + 13), f"{r['sem_cls']} {r['p_sem']:.2f}  {r['area']:.1f}st2", fill="gray")
        sheet.save(a.sheet)
        print(f"-> {a.sheet}")


if __name__ == "__main__":
    main()
