"""LEGO build (.ldr) -> real triangle mesh, and how it corresponds to the input mesh.

The LDraw library holds each part's actual geometry (type 3 / 4 lines, sub-files and primitives); this walks it,
places every part, colours it, and writes a .glb.  With --compare, the input mesh is placed the way the sampler places
it (max horizontal extent = --studs studs, ground at 0, centred on x / z) and the two surfaces are compared:

    on the VISIBLE surfaces (rays from 25 directions; hidden inner walls don't count), aligned by the best translation:
    chamfer      two-way mean surface distance, studs and % of model size
    hausdorff    the largest distance either way (the worst mismatch); hausdorff95: its 95th percentile
    covered      share of the input surface within 1 stud of the build (shape the build has)
    extra        share of the build surface more than 1 stud from the input (shape the build adds)

and writes <out>_overlay.png (front / side / top: green = both, red = input only, blue = build only) and
<out>_both.glb (the build plus the input, half transparent).

    .venv-arm/bin/python tools/ldr_to_mesh.py build.ldr out.glb [--compare input.glb --studs 32]
"""
from __future__ import annotations

import argparse
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools" / "set_study")]
STUD = 20.0                                                                      # LDU per stud


def _matrix(t) -> np.ndarray:
    x, y, z, a, b, c, d, e, f, g, h, i = map(float, t)
    return np.array([[a, b, c, x], [d, e, f, y], [g, h, i, z], [0, 0, 0, 1]])


@lru_cache(maxsize=None)
def _part(name: str) -> tuple[np.ndarray, np.ndarray]:
    """A library file's triangles in its own frame: (n, 3, 3) vertices, (n,) colour codes (16 = the caller's colour)."""
    from ldraw_read import _find
    f = _find(name)
    if f is None: return np.zeros((0, 3, 3)), np.zeros(0, int)
    tris, cols = [], []
    for ln in f.read_text(errors="replace").splitlines():
        t = ln.split()
        if not t: continue
        if t[0] == "1" and len(t) >= 15:
            v, c = _part(" ".join(t[14:]).lower().replace("\\", "/"))
            if len(v):
                M = _matrix(t[2:14]); tris.append(v @ M[:3, :3].T + M[:3, 3]); cc = int(t[1]) if t[1].lstrip("-").isdigit() else 16
                cols.append(np.where(c == 16, cc, c))
        elif t[0] in ("3", "4"):
            n = int(t[0]); p = np.array(t[2:2 + 3 * n], float).reshape(n, 3); c = int(t[1]) if t[1].lstrip("-").isdigit() else 16
            for tri in ([p[[0, 1, 2]]] if n == 3 else [p[[0, 1, 2]], p[[0, 2, 3]]]):
                tris.append(tri[None]); cols.append(np.array([c]))
    if not tris: return np.zeros((0, 3, 3)), np.zeros(0, int)
    return np.concatenate(tris), np.concatenate(cols)


def _rgb() -> dict:
    """LDraw colour code -> RGB from the LDConfig the renders use."""
    from ldraw_read import LDRAW_OFFICIAL
    out = {}
    for ln in (LDRAW_OFFICIAL / "LDConfig.ldr").read_text(errors="replace").splitlines():
        t = ln.split()
        if "!COLOUR" in t and "CODE" in t and "VALUE" in t:
            v = t[t.index("VALUE") + 1].lstrip("#"); out[int(t[t.index("CODE") + 1])] = tuple(int(v[k:k + 2], 16) for k in (0, 2, 4))
    return out


def build_mesh(ldr: Path):
    """The build as one coloured trimesh, in LDU with +y UP (LDraw's -y), ground at 0, centred on x / z."""
    import trimesh
    from ldraw_read import flatten
    rgb = _rgb(); V, C = [], []
    for p in flatten(ldr):
        v, c = _part(p.name)
        if not len(v): continue
        V.append(v @ p.M[:3, :3].T + p.M[:3, 3]); C.append(np.where(c == 16, p.colour, c))
    V = np.concatenate(V) * np.array([1, -1, 1]); C = np.concatenate(C)
    V = V[:, [0, 2, 1]]                                                          # y flip mirrors: swap two vertices to keep faces outward
    V -= np.array([(V[..., 0].min() + V[..., 0].max()) / 2, V[..., 1].min(), (V[..., 2].min() + V[..., 2].max()) / 2])
    m = trimesh.Trimesh(vertices=V.reshape(-1, 3), faces=np.arange(len(V) * 3).reshape(-1, 3), process=False)
    m.visual.face_colors = np.array([(*rgb.get(int(c), (160, 160, 160)), 255) for c in C], np.uint8)
    return m


def input_mesh(path: Path, studs: int):
    """The input mesh placed as the sampler places it: turned square to the grid (align_to_axes), max horizontal
    extent = studs * 20 LDU, ground 0, centred (compare() then refines the offset)."""
    from everything2lego.voxelize import align_to_axes, load_mesh
    m = load_mesh(path)                                                          # the sampler's chain (MeshTarget.from_path):
    m.process(validate=True); m = align_to_axes(m); v = m.vertices.copy()        # cleaned, squared to the grid, then scaled
    ext = v.max(0) - v.min(0); s = studs * STUD / max(ext[0], ext[2])
    v = v * s; v -= np.array([(v[:, 0].min() + v[:, 0].max()) / 2, v[:, 1].min(), (v[:, 2].min() + v[:, 2].max()) / 2])
    m.vertices = v
    return m


def visible_points(m, spacing: float = STUD / 4, n: int = 800000) -> tuple[np.ndarray, np.ndarray]:
    """The surface seen from outside, as a depth buffer of dense surface samples: for each of 13 directions (from
    above and the sides, none from below) keep the nearest sample per `spacing`-LDU pixel.  Hidden inner walls of a
    LEGO build drop out; works on meshes that are not closed solids.  Returns (points, face index)."""
    import trimesh
    pts, fi = trimesh.sample.sample_surface(m, n, seed=0)
    dirs = [np.array(v, float) for v in {(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)}
            if v != (0, 0, 0) and v[1] <= 0]                                      # looking down or level
    keep = np.zeros(len(pts), bool)
    for d in dirs:
        d = d / np.linalg.norm(d); u = np.cross(d, [0, 0, 1.0] if abs(d[2]) < 0.9 else [1.0, 0, 0]); u /= np.linalg.norm(u); v = np.cross(d, u)
        pu, pv, depth = pts @ u, pts @ v, pts @ d
        key = np.floor(pu / spacing).astype(np.int64) * 1_000_003 + np.floor(pv / spacing).astype(np.int64)
        order = np.lexsort((depth, key))                                           # per pixel, nearest first
        first = np.ones(len(order), bool); first[1:] = key[order][1:] != key[order][:-1]
        keep[order[first]] = True
    return pts[keep], fi[keep]


def compare(build, inp, out: Path, n: int = 60000) -> dict:
    """Shape match on the VISIBLE surfaces, after aligning by the best translation (ICP).
    chamfer   two-way mean distance;  hausdorff  the largest;  hausdorff95  the 95th percentile (one stray point
    doesn't set it); covered / extra  share of input / LEGO visible surface within / beyond 1 stud of the other."""
    import trimesh  # noqa: F401
    from PIL import Image
    from scipy.spatial import cKDTree
    pi, fi = visible_points(inp); pb, fb = visible_points(build)
    shift = np.zeros(3); ti = cKDTree(pi)
    for _ in range(30):
        tb = cKDTree(pb + shift); _, j1 = tb.query(pi); _, j2 = ti.query(pb + shift)
        step = np.concatenate([pi - (pb + shift)[j1], pi[j2] - (pb + shift)]).mean(0); shift += step
        if np.linalg.norm(step) < 0.05: break
    pb = pb + shift; build.apply_translation(shift)
    d_in = cKDTree(pb).query(pi)[0] / STUD; d_b = cKDTree(pi).query(pb)[0] / STUD
    size = float(np.max(inp.bounds[1] - inp.bounds[0])) / STUD
    res = {"chamfer_studs": float((d_in.mean() + d_b.mean()) / 2), "chamfer_pct": float(100 * (d_in.mean() + d_b.mean()) / 2 / size),
           "hausdorff_studs": float(max(d_in.max(), d_b.max())), "hausdorff95_studs": float(max(np.percentile(d_in, 95), np.percentile(d_b, 95))),
           "covered": float((d_in <= 1).mean()), "extra": float((d_b > 1).mean()), "shift_ldu": [round(float(v), 1) for v in shift],
           "points": [len(pi), len(pb)]}
    # overlay: occupancy of surface samples per view at half-stud pixels
    lo = np.minimum(pb.min(0), pi.min(0)); hi = np.maximum(pb.max(0), pi.max(0)); px = STUD / 2
    tiles = []
    for a, b in ((0, 1), (2, 1), (0, 2)):                                        # front (x, y), side (z, y), top (x, z)
        W, H = int((hi[a] - lo[a]) / px) + 2, int((hi[b] - lo[b]) / px) + 2
        def occ(p):
            g = np.zeros((H, W), bool); g[((p[:, b] - lo[b]) / px).astype(int), ((p[:, a] - lo[a]) / px).astype(int)] = True
            from scipy.ndimage import binary_closing
            return binary_closing(g, iterations=1)
        gi, gb = occ(pi), occ(pb); img = np.full((H, W, 3), 255, np.uint8)
        img[gi & gb] = (60, 170, 80); img[gi & ~gb] = (215, 60, 50); img[~gi & gb] = (60, 110, 220)
        im = Image.fromarray(img[::-1] if b == 1 else img).resize((W * 6, H * 6), Image.NEAREST); tiles.append(im)
    Wt, Ht = sum(t.width for t in tiles) + 40, max(t.height for t in tiles)
    sheet = Image.new("RGB", (Wt, Ht), "white"); x = 0
    for t in tiles: sheet.paste(t, (x, 0)); x += t.width + 20
    sheet.save(out.with_name(out.stem + "_overlay.png"))
    ghost = inp.copy(); ghost.visual.face_colors = np.array([200, 200, 200, 90], np.uint8)
    trimesh.Scene([build, ghost]).export(str(out.with_name(out.stem + "_both.glb")))
    return res


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("ldr", type=Path); ap.add_argument("out", type=Path)
    ap.add_argument("--compare", type=Path); ap.add_argument("--studs", type=int, default=32)
    a = ap.parse_args()
    m = build_mesh(a.ldr); m.export(str(a.out)); print(f"{a.out}: {len(m.faces)} triangles")
    if a.compare:
        r = compare(m, input_mesh(a.compare, a.studs), a.out)
        print("  ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in r.items()))
        print(f"overlay -> {a.out.with_name(a.out.stem + '_overlay.png')}  both -> {a.out.with_name(a.out.stem + '_both.glb')}")


if __name__ == "__main__":
    main()
