"""Express the target in LEGO's geometry: every smooth surface of the mesh turned to a pitch LEGO builds.

Colours are snapped to the palette before building; this does the same for shape.  One rule for every surface:
    segment   region growing over face adjacency: a face joins a region when its normal is within `angle` of the
              region's area-weighted mean normal (so a curve can't creep around: the mean stays put)
    fit       the region's plane: area-weighted mean normal through its area-weighted centroid
    snap      tilt from horizontal to the nearest of PITCHES; the normal's heading to the nearest grid axis
    move      each vertex to the least-squares point of the snapped planes of the regions around it (one plane:
              projected onto it; two: onto their line, so a ridge or an eave stays a sharp edge; the smallest such
              move, planes too close to parallel to cross (rcond) counting as one) -- vertices stay shared, the mesh stays as closed
              as it was, and texture coordinates are untouched
A region is left as it is when it is small (under `min_area` stud tops), curved (vertex RMS off its plane over
`flat` x sqrt(area)), or would have to turn more than `max_turn` (a diagonal wall: LEGO builds it as a staircase
anyway, and turning it would distort the model).

PITCHES are the row pitches of the vocabulary's slopes, from their LDraw geometry (`derive_pitches`): a row of
slopes climbs one part height per sloped run, e.g. a "33" slope (3298) rises 24 LDU over its 40 LDU face, 31.0 deg;
a "45" (3039) 24 over 20, 50.2 deg.  A pitch counts when at least two parts give it.

    .venv-arm/bin/python -m everything2lego.target_snap            # self-check on a gable box
    .venv-arm/bin/python -m everything2lego.target_snap --derive   # re-derive PITCHES (needs LDraw on PRO-G40)
"""
from __future__ import annotations

import numpy as np
import trimesh

# deg from horizontal; derive_pitches() on select_vocabulary("building", 2000) gives the slope ones
PITCHES = (0.0, 11.3, 21.8, 31.0, 39.6, 50.2, 56.3, 67.4, 74.5, 90.0)


def derive_pitches(kind: str = "building", n: int = 2000, min_parts: int = 2) -> list[tuple[float, list[str]]]:
    """(pitch, parts) for every row pitch given by >= min_parts single-direction slopes of the vocabulary:
    atan(part height / horizontal run of its main sloped face)."""
    import collections, sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(root / "tools"), str(root / "tools" / "set_study")]
    from ldr_to_mesh import _part
    from .sampler import select_vocabulary
    out = collections.defaultdict(list)
    for v in select_vocabulary(kind, n):
        if v.part.shape != "slope":
            continue
        tri, _ = _part(v.part.ldraw.lower())
        if not len(tri):
            continue
        nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        a = np.linalg.norm(nrm, axis=1); nrm /= np.maximum(a, 1e-9)[:, None]
        tilt = np.round(np.degrees(np.arccos(np.clip(-nrm[:, 1], -1, 1))), 1)        # LDraw y is down
        ax = np.argmax(np.abs(nrm[:, [0, 2]]), 1) * 2
        fall = np.sign(nrm[np.arange(len(nrm)), ax])
        ok = (tilt > 2) & (tilt < 88) & (np.minimum(abs(nrm[:, 0]), abs(nrm[:, 2])) < 0.02)
        if not ok.any():
            continue
        keys = list(zip(tilt[ok], ax[ok], fall[ok]))                                   # one face: tilt, axis, way
        w = collections.Counter()
        for k, ar in zip(keys, a[ok]):
            w[k] += ar
        if len({(x, s) for _, x, s in w}) > 1 and max(w.values()) < 0.9 * sum(w.values()):
            continue                                                                   # double / convex: no rows
        t, x, s = max(w, key=w.get)
        sel = np.flatnonzero(ok)[[k == (t, x, s) for k in keys]]
        run = np.ptp(tri[sel][:, :, x])
        out[round(float(np.degrees(np.arctan2(v.part.height_units * 8.0, run))), 1)].append(v.part.ldraw)
    return sorted((p, ps) for p, ps in out.items() if len(ps) >= min_parts)


def _snap_normal(n: np.ndarray, pitches) -> np.ndarray:
    """The buildable direction nearest n: tilt to a pitch, heading to an axis (y up)."""
    tilt = np.degrees(np.arccos(np.clip(abs(n[1]), 0, 1)))
    t = np.radians(min(pitches, key=lambda p: abs(p - tilt)))
    h = np.zeros(3)
    k = 0 if abs(n[0]) >= abs(n[2]) else 2
    h[k] = 1.0 if n[k] >= 0 else -1.0
    return h * np.sin(t) + np.array([0.0, 1.0 if n[1] >= 0 else -1.0, 0.0]) * np.cos(t)


def snap(mesh: trimesh.Trimesh, studs: int = 32, pitches=PITCHES, angle: float = 20.0, flat: float = 0.02,
         min_area: float = 1.0, max_turn: float = 25.0, rcond: float = 0.05) -> tuple[trimesh.Trimesh, dict]:
    """The mesh with its smooth regions snapped to buildable planes, and a report.  The mesh is first squared to
    the grid (voxelize.align_to_axes, as MeshTarget does), so headings snap in the grid's frame.  `studs` sets the
    scale (longest horizontal side, as the voxeliser), for `min_area` in stud tops.
    Defaults: angle 20 deg -- a mesh's own facet noise is a few degrees, while the creases that part a model's
    surfaces (wall / roof ~50, the two sides of a ridge ~70) are far wider; flat 0.02 -- a cylinder patch spanning
    +-20 deg sits ~0.03 sqrt(area) off its plane, a flat one ~0; max_turn 25 deg -- the widest heading change
    short of the diagonal (45), and above the largest tilt change (half the 15.5 deg 74.5 -> 90 gap)."""
    from scipy.sparse import coo_matrix
    from .voxelize import align_to_axes
    m = align_to_axes(mesh).copy()
    V = np.asarray(m.vertices, float)
    stud = max(np.ptp(V[:, 0]), np.ptp(V[:, 2])) / studs
    # weld by position (texture seams split vertices) for adjacency and for moving shared corners together
    _, weld = np.unique(np.round(V / (1e-6 * stud * studs)), axis=0, return_inverse=True)
    weld = weld.ravel(); F = weld[m.faces]; nv = weld.max() + 1
    N, A, C = np.asarray(m.face_normals), np.asarray(m.area_faces), np.asarray(m.triangles_center)
    adj = trimesh.graph.face_adjacency(faces=F)
    G = coo_matrix((np.ones(2 * len(adj)), (np.r_[adj[:, 0], adj[:, 1]], np.r_[adj[:, 1], adj[:, 0]])),
                   shape=(len(F),) * 2).tocsr()
    region = np.full(len(F), -1); nr = 0
    cos_t = np.cos(np.radians(angle))
    for seed in np.argsort(-A):                                    # big faces first: seeds on the broad surfaces
        if region[seed] >= 0:
            continue
        r = nr; nr += 1; region[seed] = r; acc = N[seed] * A[seed]; stack = [seed]
        while stack:
            f = stack.pop()
            for g in G.indices[G.indptr[f]:G.indptr[f + 1]]:
                if region[g] < 0 and N[g] @ acc >= cos_t * np.linalg.norm(acc):
                    region[g] = r; acc = acc + N[g] * A[g]; stack.append(g)
    total = A.sum()
    Vw = np.zeros((nv, 3)); Vw[weld] = V
    planes, rows = {}, []
    for r in range(nr):
        fs = np.flatnonzero(region == r)
        area = A[fs].sum(); n = (N[fs] * A[fs, None]).sum(0); n /= np.linalg.norm(n) + 1e-12
        c = (C[fs] * A[fs, None]).sum(0) / area
        vs = np.unique(F[fs])
        rms = float(np.sqrt(np.mean(((Vw[vs] - c) @ n) ** 2)))
        s = _snap_normal(n, pitches)
        turn = float(np.degrees(np.arccos(np.clip(n @ s, -1, 1))))
        tilt = lambda u: float(np.degrees(np.arccos(np.clip(abs(u[1]), 0, 1))))
        why = ("small" if area < min_area * stud ** 2 else "curved" if rms > flat * np.sqrt(area)
               else "turn" if turn > max_turn else "")
        if not why:
            planes[r] = (s, float(s @ c))
        rows.append(dict(region=r, area_share=float(area / total), area_studs=float(area / stud ** 2),
                         tilt=tilt(n), tilt_snapped=tilt(s), turn=turn, rms_studs=rms / stud, left=why))
    # each welded vertex: the least move putting it on the snapped planes of its regions (pseudo-inverse)
    pair = np.unique(np.c_[F.ravel(), np.repeat(region, 3)], axis=0)
    pair = pair[[p in planes for p in pair[:, 1]]]
    M = np.zeros((nv, 3, 3)); b = np.zeros((nv, 3))
    if len(pair):
        nn = np.array([planes[p][0] for p in pair[:, 1]]); dd = np.array([planes[p][1] for p in pair[:, 1]])
        off = dd - (nn * Vw[pair[:, 0]]).sum(1)                       # signed distance to each plane
        np.add.at(M, pair[:, 0], nn[:, :, None] * nn[:, None, :]); np.add.at(b, pair[:, 0], nn * off[:, None])
    Vn = Vw + (np.linalg.pinv(M, rcond=rcond) @ b[..., None])[..., 0]
    move = np.linalg.norm(Vn - Vw, axis=1) / stud
    m.vertices = Vn[weld]
    moved = move[np.unique(pair[:, 0])] if len(pair) else np.zeros(1)
    return m, dict(regions=rows, face_region=region, n_regions=nr, n_snapped=len(planes),
                   snapped_area=float(sum(r["area_share"] for r in rows if not r["left"])),
                   move_mean_studs=float(moved.mean()), move_max_studs=float(moved.max()))


def _gable(pitch: float = 37.0) -> trimesh.Trimesh:
    """A closed 8 x 6 box, 4 high, with a gable roof of `pitch` deg falling along +-z (units: studs)."""
    W, D, H = 8.0, 6.0, 4.0
    R = H + D / 2 * np.tan(np.radians(pitch))
    v = np.array([[0, 0, 0], [W, 0, 0], [W, 0, D], [0, 0, D], [0, H, 0], [W, H, 0], [W, H, D], [0, H, D],
                  [0, R, D / 2], [W, R, D / 2]], float)
    f = [[0, 1, 2], [0, 2, 3], [0, 4, 5], [0, 5, 1], [1, 5, 6], [1, 6, 2], [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0],
         [4, 8, 9], [4, 9, 5], [7, 6, 9], [7, 9, 8], [4, 7, 8], [5, 9, 6]]
    m = trimesh.Trimesh(v, f, process=False); m.fix_normals()
    return trimesh.Trimesh(*trimesh.remesh.subdivide_to_size(m.vertices, m.faces, 0.5), process=False)


if __name__ == "__main__":
    import sys
    if "--derive" in sys.argv:
        for p, ps in derive_pitches():
            print(f"{p:5.1f}  {len(ps):2d}  {' '.join(ps[:8])}")
        sys.exit()
    m0 = _gable(37.0)
    m1, rep = snap(m0, studs=8)
    for r in rep["regions"]:
        print({k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()})
    roof = [r for r in rep["regions"] if 10 < r["tilt"] < 80]
    assert roof and all(abs(r["tilt_snapped"] - 39.6) < 1e-6 and not r["left"] for r in roof), roof
    n = m1.face_normals; up = n[:, 1] > 0.1
    assert np.allclose(np.degrees(np.arccos(n[up, 1])), 39.6, atol=1e-3), "roof faces not at 39.6"
    walls = np.abs(m0.face_normals[:, 1]) < 1e-6
    assert np.allclose(m1.face_normals[walls], m0.face_normals[walls], atol=1e-9), "walls turned"
    assert m1.is_watertight and abs(m1.volume - m0.volume) / m0.volume < 0.05
    print(f"ok: roof 37 -> 39.6 deg, walls unchanged, volume {m0.volume:.1f} -> {m1.volume:.1f}, "
          f"max move {rep['move_max_studs']:.3f} studs")
