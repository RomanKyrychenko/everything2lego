"""Each grid part's shape on the build grid, from its LDraw geometry: how
full each cell of its box is, and where it connects — studs on top, stud
holes underneath, studs and holes on its sides, with the face they point
out of.  A generator fits a part to a target by these fractions (a 45°
slope half-fills its sloped cells, so it is the part for a sloped surface)
and joins parts only where a stud meets a hole.

    python tools/partsdb/shapes.py        # -> table part_shape in parts.sqlite

The shape is the part's visual hull from its three axis views (top, front,
side): exact for slopes, wedges, curves, rounds and corners, and blind to
the hollow underside every brick has.  A part qualifies when its body (the
studs aside) sits on the grid: footprint a whole number of studs, height a
whole number of plates.

`bits` is each cell's shape at 4x4x4 samples (sampler.pack); `occ` its share.
Cells are indexed [y from the bottom, x, z] in the part's own frame,
x and z from the box's minimum corner; `origin_x/z` is the box's centre in
the part frame (LDU), which is where a placement puts the footprint's
centre.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.ndimage import binary_fill_holes

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "set_study"))
from build import DB  # noqa: E402
from ldraw_read import _find, _matrix  # noqa: E402

sys.path.insert(0, str(HERE.parents[1] / "src"))
from everything2lego.sampler import SUB, pack  # noqa: E402  (one bit order for parts and targets)

CATEGORIES = ("Bricks", "Plates", "Tiles", "Bricks Sloped", "Bricks Curved", "Bricks Round and Cones",
              "Tiles Round and Curved", "Plates Round Curved and Dishes", "Bricks Wedged", "Plates Wedged",
              "Bricks Special", "Plates Special", "Tiles Special",
              "Hinges, Arms and Turntables", "Technic Bricks", "Technic Beams")
MIN_SETS = 30
TOL = 1.0                     # LDU a body may miss the grid by


# top studs (connectors, not body): stud.dat, stud2a.dat, stug-2x2.dat, ...;
# the underside's tubes (stud3, stud4, stug3-, stug4-) are inside the body
TOP_STUD = re.compile(r"^(?:.*/)?(stud|stug)(?![34])[^/]*\.dat$")


def triangles(name: str, M=np.eye(4), depth: int = 0, _cache={}) -> np.ndarray:
    """A part's body triangles (n, 3, 3), LDU, in its own frame — its top
    studs left out."""
    if depth > 14:
        return np.zeros((0, 3, 3))
    key = name.lower().replace("\\", "/")
    if depth and TOP_STUD.match(key):
        return np.zeros((0, 3, 3))
    local = _cache.get(key)
    if local is None:
        f = _find(key)
        tris = []
        if f is not None:
            for ln in f.read_text(errors="replace").splitlines():
                t = ln.split()
                if not t:
                    continue
                if t[0] == "3" and len(t) >= 11:
                    tris.append(np.array(t[2:11], float).reshape(3, 3))
                elif t[0] == "4" and len(t) >= 14:
                    q = np.array(t[2:14], float).reshape(4, 3)
                    tris += [q[[0, 1, 2]], q[[0, 2, 3]]]
                elif t[0] == "1" and len(t) >= 15:
                    sub = triangles(" ".join(t[14:]), _matrix(t[2:14]), depth + 1)
                    if len(sub):
                        tris += list(sub)
        local = _cache[key] = np.array(tris).reshape(-1, 3, 3)
    if not len(local):
        return local
    h = np.concatenate([local, np.ones(local.shape[:2] + (1,))], axis=2)
    return (h @ M.T)[..., :3]


def _silhouette(pts2: np.ndarray, lo, hi) -> np.ndarray:
    """Filled projection of triangles (n, 3, 2) on a 1/4-LDU raster over [lo, hi]."""
    s = 4.0
    W, H = int(np.ceil((hi[0] - lo[0]) * s)) + 2, int(np.ceil((hi[1] - lo[1]) * s)) + 2
    img = Image.new("1", (W, H), 0)
    dr = ImageDraw.Draw(img)
    for t in pts2:
        dr.polygon([((x - lo[0]) * s, (y - lo[1]) * s) for x, y in t], fill=1, outline=1)
    # the outer envelope: a hollow (an open stud, a tube seen end-on) is
    # filled; an opening reaching the edge (an arch) is not a hole
    return binary_fill_holes(np.array(img, dtype=bool).T)      # [u, v]


def _grid_box(v: np.ndarray, line: float) -> tuple[float, float]:
    """The smallest stud-grid span (grid lines at `line` + 20k) holding v."""
    lo = line + 20 * np.floor((v.min() - line + TOL) / 20)
    hi = line + 20 * np.ceil((v.max() - line - TOL) / 20)
    return float(lo), float(hi)


def _snap(v: float, step: float, off: float = 0.0):
    k = round((v - off) / step)
    return k * step + off if abs(k * step + off - v) <= TOL else None


def shape(row, T: np.ndarray | None = None, relaxed: bool = False, parity: int = 1) -> dict | None:
    """`T`: the triangles, when not the part file's own (an assembly).
    `relaxed`: a part off the grid (a wheel: its diameter no whole number of
    studs) is centred in the smallest grid box holding it, not refused —
    a box `parity` (1 odd, 0 even) studs across, so its centre (a hub) is on
    a cell's centre or between two cells, where pins and holders put it."""
    part_num, ldraw = row[:2]
    studs = row[2] if len(row) > 2 else None             # (x, z) of its top studs: where its grid is
    if T is None:
        try:
            T = triangles(ldraw)
        except Exception:
            return None
    if not len(T):
        return None
    P = T.reshape(-1, 3)
    if relaxed:
        lo, hi = P.min(0), P.max(0)
        c = (lo + hi) / 2
        n = np.ceil((hi - lo) / np.array([20.0, 8.0, 20.0]) - 0.05).astype(int)
        a = 0 if (hi - lo)[0] >= (hi - lo)[2] else 2         # across the wheel (its diameter), not along its axle
        n[a] += (n[a] + parity) % 2
        x0, x1 = c[0] - 10 * n[0], c[0] + 10 * n[0]
        y0, y1 = c[1] - 4 * n[1], c[1] + 4 * n[1]
        z0, z1 = c[2] - 10 * n[2], c[2] + 10 * n[2]
    else:
        x0, x1 = _snap(P[:, 0].min(), 10), _snap(P[:, 0].max(), 10)
        z0, z1 = _snap(P[:, 2].min(), 10), _snap(P[:, 2].max(), 10)
        y0, y1 = _snap(P[:, 1].min(), 8), _snap(P[:, 1].max(), 8)    # the body's top and bottom
        if None in (y0, y1) and studs:
            # the plate grid starts at its studs' base (y 0, body growing
            # downward); the box takes in what hangs below, and ignores a
            # finger or clip sticking above (hinge plates)
            y0 = 0.0
            body_max = P[P[:, 1] > -TOL, 1]
            y1 = 8 * np.ceil(((body_max.max() if len(body_max) else P[:, 1].max()) - TOL) / 8)
        if None in (x0, x1, z0, z1) and studs:
            # something sticks out past the grid (a holder's pins, a clip): the
            # grid is where its studs are; the box takes in what sticks out
            (gx, gz) = studs[0]
            x0, x1 = _grid_box(P[:, 0], gx - 10)
            z0, z1 = _grid_box(P[:, 2], gz - 10)
    if None in (x0, x1, z0, z1, y0, y1) or y1 <= y0:
        return None
    w, d, h = round((x1 - x0) / 20), round((z1 - z0) / 20), round((y1 - y0) / 8)
    if w < 1 or d < 1 or abs((x1 - x0) - 20 * w) > TOL or abs((z1 - z0) - 20 * d) > TOL or h > 30:
        return None
    if w * d * h > 4000:
        return None
    lo3, hi3 = np.array([x0, y0, z0]), np.array([x1, y1, z1])
    top = _silhouette(T[:, :, [0, 2]], lo3[[0, 2]], hi3[[0, 2]])
    front = _silhouette(T[:, :, [0, 1]], lo3[[0, 1]], hi3[[0, 1]])
    side = _silhouette(T[:, :, [2, 1]], lo3[[2, 1]], hi3[[2, 1]])
    # sample points: SUB per cell per axis, at sub-cell centres
    xs = x0 + (np.arange(w * SUB) + 0.5) * 20 / SUB
    zs = z0 + (np.arange(d * SUB) + 0.5) * 20 / SUB
    ys = y1 - (np.arange(h * SUB) + 0.5) * 8 / SUB          # from the bottom up
    ix = np.clip(((xs - x0) * 4).astype(int), 0, top.shape[0] - 1)
    iz = np.clip(((zs - z0) * 4).astype(int), 0, top.shape[1] - 1)
    iy = np.clip(((ys - y0) * 4).astype(int), 0, front.shape[1] - 1)
    inside = (top[ix][:, iz][None, :, :]
              & front[ix][:, iy].T[:, :, None]
              & side[iz][:, iy].T[:, None, :])             # [y, x, z]
    occ = inside.reshape(h, SUB, w, SUB, d, SUB).mean(axis=(1, 3, 5))
    bits = pack(inside, (h, w, d))
    return {"part_num": part_num, "ldraw": ldraw, "w": w, "d": d, "h": h,
            "origin_x": (x0 + x1) / 2, "origin_z": (z0 + z1) / 2, "origin_y": y0, "x0": x0, "z0": z0, "y0": y0,
            "occ": np.round(occ, 3).tolist(), "bits": bits.tolist(), "fill": float(occ.mean())}


SIDE_KINDS = ("stud", "antistud", "pin", "pinhole", "axle", "axlehole", "bar", "barhole", "clip",
              "finger", "generic")


def connections(con, shapes: dict[str, dict]) -> None:
    """Studs on top, holes underneath, and every connector on the sides
    (studs, holes, pins, axles, bars, clips, ...) in the cell of the face it
    opens on and that face.  An assembly takes its first part's connectors."""
    c = pd.read_sql(f"SELECT part_num, kind, gender, x, y, z, fx, fy, fz FROM connector "
                    f"WHERE kind IN ({','.join(repr(k) for k in SIDE_KINDS)})", con)
    for pn, g in c.groupby("part_num"):
        for s in [v for v in shapes.values() if v["part_num"] == pn]:
            _connect(s, g)


def _connect(s: dict, g) -> None:
    """Top studs and bottom holes as cells; side connectors as points on
    the box's faces: (kind:gender, face, t, hy) — t along the face in half
    studs from the box's minimum corner, hy in half plates from its bottom.
    Two parts join at their sides where their faces touch at the same point
    with kinds that fit (sampler.FITS)."""
    top, bottom, sides, off = set(), set(), set(), 0
    occ = s["occ"]
    W, D, H = s["w"], s["d"], s["h"]
    for r in g.itertuples(index=False):
        ry = r.y - s["y0"]                                   # from the body's top
        if abs(r.fy) > 0.9:
            if r.kind not in ("stud", "antistud"):
                continue
            fi, fk = (r.x - s["x0"]) / 20, (r.z - s["z0"]) / 20
            i, k = int(np.floor(fi)), int(np.floor(fk))
            if abs(fi - i - 0.5) > 0.1 or abs(fk - k - 0.5) > 0.1 or not (0 <= i < W and 0 <= k < D):
                off += 1                                     # a jumper's centred stud: off the grid
                continue
            # a stud or hole only where the part is solid at that face
            if r.kind == "stud" and r.fy < 0 and abs(ry) < 1 and occ[-1][i][k] >= 0.5:
                top.add((i, k))
            elif r.kind == "antistud" and r.fy > 0 and abs(ry - H * 8) < 1 and occ[0][i][k] >= 0.5:
                bottom.add((i, k))
            continue
        if abs(r.fx) > 0.9:
            face, t = ("+x" if r.fx > 0 else "-x"), (r.z - s["z0"]) / 10
        elif abs(r.fz) > 0.9:
            face, t = ("+z" if r.fz > 0 else "-z"), (r.x - s["x0"]) / 10
        else:
            continue                                         # a slanted connector
        hy = (H * 8 - ry) / 4                                # half plates from the bottom
        if abs(t - round(t)) > 0.2:
            off += 1                                         # between half studs: off the grid
            continue
        t, hy = int(round(t)), int(np.clip(round(hy), 0, 2 * H))
        if 0 <= t <= 2 * (D if face[1] == "x" else W):
            sides.add((f"{r.kind}:{r.gender}", face, t, hy))
    s["studs_top"] = sorted(top)
    s["holes_bottom"] = sorted(bottom)
    s["sides"] = sorted(sides)
    s["offgrid"] = off


def wheel_assemblies(con) -> dict[str, dict]:
    """Each rim with the tyre official models put on it most (wheels.py),
    as one part: the pair's shape, off the grid so centred in its box; the
    rim's connectors (its hub); the tyre written with it (`companion`)."""
    from build import DATA, _ld_to_part
    f = DATA / "analysis" / "wheel_pairs.json"
    if not f.exists():
        return {}
    ldmap = _ld_to_part(con)
    out = {}
    for p in json.loads(f.read_text()):                 # most used first: the first tyre per rim wins
        pn = ldmap.get(p["rim"], p["rim"][:-4])
        if f"{pn}|odd" in out:
            continue
        T = np.concatenate([triangles(p["rim"]), triangles(p["tyre"])])
        for parity, name in ((1, "odd"), (0, "even")):
            s = shape((pn, p["rim"]), T, relaxed=True, parity=parity)
            if s is not None:
                s["companion"], s["variant"] = p["tyre"], name
                out[f"{pn}|{name}"] = s
    return out


def main() -> None:
    con = sqlite3.connect(DB)
    q = f"""SELECT p.part_num, MIN(g.ldraw) FROM part p JOIN geometry g ON g.part_num = p.part_num
            WHERE p.category IN ({",".join("?" * len(CATEGORIES))}) AND p.n_sets_recent >= ?
            AND g.ldraw IS NOT NULL GROUP BY p.part_num"""
    studs = {}
    for pn, x, z in con.execute("SELECT part_num, x, z FROM connector "
                                "WHERE (kind='stud' AND fy<-0.9) OR (kind='antistud' AND fy>0.9)"):
        studs.setdefault(pn, []).append((x, z))
    rows = [(pn, ld, studs.get(pn)) for pn, ld in con.execute(q, (*CATEGORIES, MIN_SETS)).fetchall()]
    with Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:
        got = [s for s in pool.imap_unordered(shape, rows, chunksize=4) if s]
    shapes = {s["part_num"]: s for s in got}
    shapes.update(wheel_assemblies(con))
    connections(con, shapes)
    df = pd.DataFrame([{**s, "occ": json.dumps(s["occ"]), "bits": json.dumps(s["bits"]),
                        "studs_top": json.dumps(s.get("studs_top", [])),
                        "holes_bottom": json.dumps(s.get("holes_bottom", [])),
                        "sides": json.dumps(s.get("sides", []))} for s in shapes.values()])
    df.to_sql("part_shape", con, if_exists="replace", index=False)
    con.execute("DROP INDEX IF EXISTS ix_shape")
    con.execute("CREATE INDEX ix_shape ON part_shape(part_num)")
    con.commit()
    print(f"part_shape: {len(df)} parts ({len(rows)} candidates on the grid, "
          f"{int(df.companion.notna().sum()) if 'companion' in df else 0} wheel assemblies)")


if __name__ == "__main__":
    main()
