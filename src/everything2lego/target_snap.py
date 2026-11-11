"""Express the target in LEGO's geometry: the mesh's planes turned to pitches LEGO builds, the rest following smoothly.

Colours are snapped to the palette before building; this does the same for shape.  After Zhou et al. 2019 ("Automatic
generation of vivid LEGO architectural sculptures", planarity detection + least-squares deformation; ideas only):
    scale     everything is measured in studs (longest horizontal side / `studs`, as the voxeliser), because the
              grid can't show detail finer than its cell: a shingle, a plank or a 2-degree warp all voxelise as
              noise on the plane they sit on.  So planes are found at stud scale, not at the mesh's own detail:
    normals   each face's normal averaged (area-weighted) over the faces within `radius` (1 stud = one grid cell)
              -- over a closed staircase that sum is exactly the normal of the plane it climbs, so shingles
              smaller than a cell read as their roof (where the two skins of a thin part cancel, the raw normal
              still counts)
    segment   region growing over face adjacency, flattest neighbourhoods first: a face joins a region when its
              stud-scale (or raw) normal is within `angle` of the region's mean and its corners lie within `tol`
              (half a stud: the grid rounds a surface to the nearest cell, so anything closer than that IS the
              plane at LEGO scale) of the region's plane.  Planes under a cell wide or a stud top in area are
              dropped.  Then the scale test both ways: a plane's face not facing its way (a riser; the foot of a
              boulder on the ground) leaves it when shape outside every plane rises more than `tol` off it within a
              cell; and a connected piece of faces outside every plane (a crease, whose averaged normal blends two
              planes; a strip under a cell wide) joins the planes it touches, each face the nearest, when all of it
              lies within `tol` of them -- a piece rising further (the boulder) is shape and stays as it is
    snap      each region's plane (area-weighted least squares through its corners, so a staircase fits its
              middle): tilt to the nearest of `pitches`, heading to the nearest grid axis
    deform    one sparse linear least squares over all vertex displacements D (in studs):
                  w_p * n_p . (v + D_v) = w_p * d_p        every vertex v of every snapped plane p (planarity)
                  L D_v = 0                                  every vertex on no plane (uniform Laplacian: free
                                                             parts follow their planes smoothly, keeping shape)
                  w_a * D_v = 0                              every vertex (a weak anchor: D fades over ~`fade` studs
                                                             away from the planes and the system has one solution)
              so a vertex on one plane lands on it, one on two planes on their line (ridges and eaves stay
              sharp; planes within `angle` of each other, or meeting more than a cell away, keep only the bigger
              one there), and a shingle staircase is flattened onto its roof plane.  Vertices are welded by position
              for this (texture seams), so the mesh stays as closed as it was and UVs are untouched.
A region is left as it is when it is small (under `min_area` stud tops or a cell wide) or would have to turn more than `max_turn`
(a diagonal wall: LEGO builds it as a staircase anyway, and turning it would distort the model).

PITCHES are the row pitches of the vocabulary's slopes, from their LDraw geometry (`derive_pitches`): a row of
slopes climbs one part height per sloped run, e.g. a "33" slope (3298) rises 24 LDU over its 40 LDU face, 31.0 deg;
a "45" (3039) 24 over 20, 50.2 deg.  A pitch counts when at least two parts give it.  PRESETS["roof"] drops 39.6:
its parts (54200, 85984, 7835, cheese "31" slopes 2/3 brick high) have no stud on top, so they only cap a step and
can't be stacked into rows; a roof surface at 39.6 would be all caps.  Every other pitch has studded parts.

    .venv-arm/bin/python -m everything2lego.target_snap            # self-check on a shingled gable box
    .venv-arm/bin/python -m everything2lego.target_snap --derive   # re-derive PITCHES (needs LDraw on PRO-G40)
"""
from __future__ import annotations

import numpy as np
import trimesh

# deg from horizontal; derive_pitches() on select_vocabulary("building", 2000) gives the slope ones
PITCHES = (0.0, 11.3, 21.8, 31.0, 39.6, 50.2, 56.3, 67.4, 74.5, 90.0)
PRESETS = {"all": PITCHES, "roof": tuple(p for p in PITCHES if p != 39.6)}     # see the module docstring


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


def snap(mesh: trimesh.Trimesh, studs: int = 32, pitches="all", angle: float = 20.0, tol: float = 0.5,
         radius: float = 1.0, min_area: float = 1.0, max_turn: float = 25.0, w_plane: float = 100.0,
         fade: float = 4.0) -> tuple[trimesh.Trimesh, dict]:
    """The mesh with its stud-scale planes snapped to buildable ones and the rest deformed smoothly, and a report.
    The mesh is first squared to the grid (voxelize.align_to_axes, as MeshTarget does), so headings snap in the grid's
    frame.  `studs` sets the unit (longest horizontal side, as the voxeliser); `pitches` is a list or a PRESETS key.
    Lengths are in studs, all fixed by the grid (see the module docstring): tol 0.5 (half a cell), radius 1 (a cell),
    min_area 1 stud top, fade 4 (a displacement dies out over a 1x4 brick's length).  angle 20 deg -- a mesh's facet
    noise is a few degrees, the creases that part surfaces (wall / roof ~50, the two sides of a ridge ~70) far wider;
    max_turn 25 deg -- the widest heading change short of the diagonal (45), and above the largest tilt change (half
    the 15.5 deg 74.5 -> 90 gap); w_plane 100 -- planarity ~1e4 x stronger than shape keeping, so planes come out
    flat to ~1e-4 stud."""
    import collections
    from scipy.sparse import coo_matrix, identity, kron, vstack
    from scipy.sparse.linalg import spsolve
    from scipy.spatial import cKDTree
    from .voxelize import align_to_axes
    pitches = PRESETS[pitches] if isinstance(pitches, str) else tuple(pitches)
    m = align_to_axes(mesh).copy()
    V = np.asarray(m.vertices, float)
    stud = max(np.ptp(V[:, 0]), np.ptp(V[:, 2])) / studs
    # weld by position (texture seams split vertices) for adjacency and for moving shared corners together
    _, weld = np.unique(np.round(V / (1e-6 * stud * studs)), axis=0, return_inverse=True)
    weld = weld.ravel(); F = weld[m.faces]; nv = weld.max() + 1
    P = np.zeros((nv, 3)); P[weld] = V / stud                                        # welded vertices, studs
    cr = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])
    A = np.linalg.norm(cr, axis=1) / 2; N = cr / np.maximum(2 * A, 1e-12)[:, None]; C = P[F].mean(1)
    # stud-scale normals: area-weighted over the faces within `radius` -- over a closed staircase that sum is exactly
    # the normal of the plane it climbs; where the two skins of a thin part cancel, the raw normal still counts below
    nb = cKDTree(C).sparse_distance_matrix(cKDTree(C), radius, output_type="coo_matrix")
    i, j = nb.row, nb.col
    Ns = N * A[:, None]; np.add.at(Ns, i, N[j] * A[j, None])
    wsum = A.copy(); np.add.at(wsum, i, A[j])
    flat = np.linalg.norm(Ns, axis=1) / np.maximum(wsum, 1e-12)                      # 1 = a flat neighbourhood
    Ns /= np.maximum(np.linalg.norm(Ns, axis=1), 1e-12)[:, None]
    adj = trimesh.graph.face_adjacency(faces=F)
    G = coo_matrix((np.ones(2 * len(adj)), (np.r_[adj[:, 0], adj[:, 1]], np.r_[adj[:, 1], adj[:, 0]])),
                   shape=(len(F),) * 2).tocsr()
    region = np.full(len(F), -1); nr = 0
    cos_t = np.cos(np.radians(angle))
    for seed in np.lexsort((-A, -np.round(flat, 2))):                # flattest, then biggest, neighbourhoods first
        if region[seed] >= 0:
            continue
        r = nr; nr += 1; region[seed] = r
        acc, sA, sC = Ns[seed] * A[seed], A[seed], C[seed] * A[seed]; queue = collections.deque([seed])
        while queue:
            f = queue.popleft()
            n = acc / (np.linalg.norm(acc) + 1e-12); c = sC / max(sA, 1e-12)
            for g in G.indices[G.indptr[f]:G.indptr[f + 1]]:
                # the raw normal too: a face on the plane next to a crease has its stud-scale normal blended
                if (region[g] < 0 and max(Ns[g] @ n, N[g] @ n) >= cos_t
                        and np.abs((P[F[g]] - c) @ n).max() <= tol):
                    region[g] = r; acc = acc + Ns[g] * A[g]; sA += A[g]; sC = sC + C[g] * A[g]; queue.append(g)
    total = A.sum()
    tilt = lambda u: float(np.degrees(np.arccos(np.clip(abs(u[1]), 0, 1))))
    nrm = lambda fs: (N[fs] * A[fs, None]).sum(0)

    def fit(fs):
        """Area-weighted least-squares plane of the faces' corners (a staircase cut short still fits its middle),
        and the region's width: the narrower in-plane side of the rectangle with the same spread."""
        w = np.repeat(A[fs] / 3, 3); X = P[F[fs]].reshape(-1, 3); c = w @ X / w.sum()
        ev, vec = np.linalg.eigh(((X - c) * w[:, None]).T @ (X - c) / w.sum())
        n = vec[:, 0] * (1.0 if vec[:, 0] @ nrm(fs) >= 0 else -1.0)
        return n, c, float(np.sqrt(12 * max(ev[1], 0.0)))

    def verdict(fs):
        n, c, width = fit(fs); s = _snap_normal(n, pitches)
        turn = float(np.degrees(np.arccos(np.clip(n @ s, -1, 1))))
        # a plane narrower than a cell can't show in the grid as a plane
        why = "small" if A[fs].sum() < min_area or width < 1.0 else "turn" if turn > max_turn else ""
        return n, c, s, turn, width, why

    fits = {}
    for r in range(nr):
        fs = np.flatnonzero(region == r)
        if A[fs].sum() >= min_area:
            n, c, *_, why = verdict(fs)
            if not why:
                fits[r] = (n, c)
    from scipy.sparse.csgraph import connected_components

    def pieces(fs):
        """The connected pieces of a face set, as arrays of face ids."""
        _, lab = connected_components(G[fs][:, fs], directed=False)
        o = np.argsort(lab, kind="stable")
        return [fs[i] for i in np.split(o, np.flatnonzero(np.diff(lab[o])) + 1)]

    # the other way round: a plane's face that doesn't face its way (a shingle's riser; the foot of a boulder
    # standing on the ground) leaves it when, within a cell of it, there is shape outside every plane rising more
    # than `tol` off this one -- it is the start of that shape, not a ripple of the plane
    inp = np.isin(region, list(fits))
    pl = np.where(inp, region, -1)
    k = inp[i] & ~inp[j]
    ii, jj = i[k], j[k]
    nrm_i = np.array([fits[r][0] for r in pl[ii]]).reshape(-1, 3); c_i = np.array([fits[r][1] for r in pl[ii]]).reshape(-1, 3)
    off = (N[ii] * nrm_i).sum(1) < cos_t
    far = np.abs(np.einsum("fkd,fd->fk", P[F[jj]] - c_i[:, None], nrm_i)).max(1) > tol
    rel = np.unique(ii[off & far])
    region[rel] = nr + np.arange(len(rel)); nr += len(rel)
    keep = np.isin(region, list(fits)); rest = np.flatnonzero(~keep)
    # faces left over (a crease, whose stud-scale normal blends two planes; a strip under a cell wide) join the
    # planes they touch when the whole connected piece of them lies within `tol` of those planes -- each face to the
    # nearest; a piece that rises further off (a boulder on the ground, a beam on a wall) is a shape, and stays
    for fs in pieces(rest):
        touch = np.unique(region[G[fs].indices]); touch = [r for r in touch if r in fits]
        if not touch:
            continue
        dist = np.stack([np.abs((P[F[fs]] - fits[r][1]) @ fits[r][0]).max(1) for r in touch])
        if dist.min(0).max() <= tol:
            region[fs] = np.array(touch)[dist.argmin(0)]
    planes, rows = {}, []
    for r in np.unique(region):
        fs = np.flatnonzero(region == r)
        area = A[fs].sum()
        if area < 0.05 * min_area:                                    # crumbs: not worth a report row
            continue
        n, c, s, turn, width, why = verdict(fs)
        rms = float(np.sqrt(np.mean(((P[np.unique(F[fs])] - c) @ n) ** 2)))
        if not why:
            planes[r] = (s, float(s @ c))
        rows.append(dict(region=int(r), area_share=float(area / total), area_studs=float(area), n_faces=len(fs),
                         width_studs=width, tilt=tilt(n), tilt_snapped=tilt(s), heading=float(np.degrees(
                             np.arctan2(n[0], n[2]))), turn=turn, rms_studs=rms, left=why))
    # deform: one sparse least squares over the displacements (module docstring)
    pair = np.unique(np.c_[F.ravel(), np.repeat(region, 3)], axis=0)
    pair = pair[np.isin(pair[:, 1], list(planes))]
    # a vertex on several planes is sent to their line / corner, unless that is out of its reach: two planes within
    # `angle` of each other (a 50.2 roof next to a 56.3 strip) or three nearly sharing a line meet far off.  So each
    # vertex keeps its planes biggest first while they stay `angle` apart and their meeting point within a cell
    # (1 stud) of it; the rest meet the kept ones at a kink
    big = np.array([-A[region == p].sum() for p in pair[:, 1]]) if len(pair) else np.zeros(0)
    pair = pair[np.lexsort((big, pair[:, 0]))]
    first = np.r_[True, pair[1:, 0] != pair[:-1, 0]] if len(pair) else np.zeros(0, bool)
    ok = np.ones(len(pair), bool)
    for i0 in np.flatnonzero(first & np.r_[~first[1:], False]):          # vertices on two or more planes
        v = pair[i0, 0]; kept = [planes[pair[i0, 1]]]
        for i in range(i0 + 1, len(pair)):
            if pair[i, 0] != v:
                break
            n_i, d_i = planes[pair[i, 1]]
            if all(abs(n_i @ k[0]) < cos_t for k in kept):
                Nm = np.array([k[0] for k in kept] + [n_i]); dm = np.array([k[1] for k in kept] + [d_i])
                x = np.linalg.lstsq(Nm, dm - Nm @ P[v], rcond=None)[0]           # a fourth plane: must pass there too
                ok[i] = np.linalg.norm(x) <= 1.0 and np.abs(Nm @ x - (dm - Nm @ P[v])).max() < 1e-3
            else:
                ok[i] = False
            if ok[i]:
                kept.append((n_i, d_i))
    pair = pair[ok]
    on = np.zeros(nv, bool); on[pair[:, 0]] = True
    nn = np.array([planes[p][0] for p in pair[:, 1]]).reshape(-1, 3)
    dd = np.array([planes[p][1] for p in pair[:, 1]])
    Ap = coo_matrix((w_plane * nn.ravel(), (np.repeat(np.arange(len(pair)), 3),
                     (3 * pair[:, :1] + np.arange(3)).ravel())), shape=(len(pair), 3 * nv))
    bp = w_plane * (dd - (nn * P[pair[:, 0]]).sum(1))
    E = np.unique(np.sort(np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], axis=1), axis=0)
    W = coo_matrix((np.ones(2 * len(E)), (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])), shape=(nv, nv)).tocsr()
    deg = np.asarray(W.sum(1)).ravel()
    L = (identity(nv) - W.multiply(1 / np.maximum(deg, 1)[:, None])).tocsr()[np.flatnonzero(~on)]
    edge = float(np.linalg.norm(P[E[:, 0]] - P[E[:, 1]], axis=1).mean())
    w_a = (edge / fade) ** 2                                         # screened bi-Laplacian: decay ~ edge / sqrt(w_a)
    I3 = identity(3)
    Asys = vstack([Ap, kron(L, I3), w_a * identity(3 * nv)]).tocsr()
    b = np.r_[bp, np.zeros(3 * L.shape[0] + 3 * nv)]
    D = spsolve((Asys.T @ Asys).tocsc(), Asys.T @ b).reshape(nv, 3) if len(pair) else np.zeros((nv, 3))
    Pn = P + D
    resid = np.abs((nn * Pn[pair[:, 0]]).sum(1) - dd) if len(pair) else np.zeros(1)
    move = np.linalg.norm(D, axis=1)
    m.vertices = Pn[weld] * stud
    return m, dict(regions=rows, face_region=region, n_regions=nr, n_snapped=len(planes),
                   snapped_area=float(sum(r["area_share"] for r in rows if not r["left"])),
                   move_mean_studs=float(move[on].mean()) if on.any() else 0.0,
                   move_max_studs=float(move.max()), move_free_mean_studs=float(move[~on].mean()) if (~on).any() else 0.0,
                   plane_resid_max_studs=float(resid.max()), stud=float(stud))

def _gable(pitch: float = 37.0, shingle: float = 0.2, run: float = 0.6) -> trimesh.Trimesh:
    """A closed 8 x 6 box, 4 high, with a gable roof of `pitch` deg falling along +-z (units: studs), shingled: each
    roof vertex pushed out along the roof normal by a sawtooth, 0 -> `shingle` over every `run` up the slope."""
    W, D, H = 8.0, 6.0, 4.0
    R = H + D / 2 * np.tan(np.radians(pitch))
    v = np.array([[0, 0, 0], [W, 0, 0], [W, 0, D], [0, 0, D], [0, H, 0], [W, H, 0], [W, H, D], [0, H, D],
                  [0, R, D / 2], [W, R, D / 2]], float)
    f = [[0, 1, 2], [0, 2, 3], [0, 4, 5], [0, 5, 1], [1, 5, 6], [1, 6, 2], [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0],
         [4, 8, 9], [4, 9, 5], [7, 6, 9], [7, 9, 8], [4, 7, 8], [5, 9, 6]]
    m = trimesh.Trimesh(v, f, process=False); m.fix_normals()
    v, f = trimesh.remesh.subdivide_to_size(m.vertices, m.faces, 0.2)
    t = np.radians(pitch)
    for side in (-1, 1):                                               # z < D/2 falls to -z, z > D/2 to +z
        n = np.array([0, np.cos(t), side * np.sin(t)])
        on = np.abs((v - [0, R, D / 2]) @ n) < 1e-9
        up = np.abs(v[on, 2] - D / 2) / np.cos(t)                      # distance down the slope from the ridge
        v[on] += np.outer(shingle * ((up / run) % 1.0) * (up > 1e-9) * (up < D / 2 / np.cos(t) - 1e-9), n)
    return trimesh.Trimesh(v, f, process=False)


def _check(preset: str, want: float) -> None:
    m0 = _gable(37.0)
    m1, rep = snap(m0, studs=8, pitches=preset)
    roof = [r for r in rep["regions"] if 10 < r["tilt"] < 80 and not r["left"]]
    for r in sorted(rep["regions"], key=lambda r: -r["area_studs"])[:8]:
        print({k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()})
    assert len(roof) == 2 and all(abs(r["tilt_snapped"] - want) < 1e-6 for r in roof), roof
    n = m1.face_normals; V = m1.vertices
    top = (m0.triangles_center[:, 1] > 4.0 + 1e-6) & (np.abs(m0.triangles_center[:, 0] - 4.0) < 4.0 - 1e-6)
    for r in roof:                                                     # each roof side: one plane, flat, at the pitch
        fs = rep["face_region"] == r["region"]
        assert m0.area_faces[fs & top].sum() > 0.9 * m0.area_faces[top].sum() / 2, "roof side split"
        q = V[np.unique(m1.faces[fs])]; q = q - q.mean(0); nrm = np.linalg.svd(q)[2][-1]
        got = np.degrees(np.arccos(abs(nrm[1])))
        assert np.abs(q @ nrm).max() < 1e-3, f"roof side not planar: {np.abs(q @ nrm).max():.4f}"
        assert abs(got - want) < 0.05, f"roof side at {got:.2f}, want {want}"
    walls = np.abs(m0.face_normals[:, 1]) < 1e-6
    assert np.abs(n[walls][:, 1]).max() < 1e-3, "walls tilted"
    assert m1.is_watertight and abs(m1.volume - m0.volume) / m0.volume < 0.1
    print(f"ok [{preset}]: shingled 37 deg roof -> 2 planes at {want}, walls upright, volume {m0.volume:.1f} -> "
          f"{m1.volume:.1f}, plane vertices moved {rep['move_mean_studs']:.3f} mean / {rep['move_max_studs']:.3f} "
          f"max studs, plane residual {rep['plane_resid_max_studs']:.1e}")


if __name__ == "__main__":
    import sys
    if "--derive" in sys.argv:
        for p, ps in derive_pitches():
            print(f"{p:5.1f}  {len(ps):2d}  {' '.join(ps)}")
        sys.exit()
    _check("all", 39.6)
    _check("roof", 31.0)
