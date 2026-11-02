"""Force-based weak-joint analysis of an LDraw build: rigid-block equilibrium (RBE) as a sparse LP.

Model (StableLego, Liu et al. RA-L 2024; artificial links after Kollsker & Malaguti -- own implementation):
  bodies    every part BrickNet loads (tools/validity_check.load), a rigid block; reference point = its box centre,
            mass = 0.1 g per stud x stud x plate cell of its box (2x4 brick 2.4 g), gravity torque-free about it.
  rows      6 per body: force x, y, z and torque x, y, z (yaw included, the "corrected" profile); levers in studs
            (torque rows in N stud: with gravity as the only load the length unit cancels, and the LP scales better).
  variables all >= 0, each one a force magnitude acting +dir on body a and -dir on body b (Newton's third law by
            construction):
    stud      per BrickNet stud edge (stud owner = below, hole owner = above, axis out of the stud, so SNOT works):
              3 (THREE_POINT) or 4 (FOUR_POINT, above part 1 stud wide across the axis) contact points at
              +-1/4 pitch, each a `normal` (pushes above along axis) and a `drag` (pulls it back onto the stud),
              plus 4 in-plane knob `press` forces at the stud centre.
    rest      boxes touching face-to-face vertically with no stud between them: push-only at the overlap corners.
    side      boxes touching on an x or z face: push-only at the 4 face corners (spans the face's force/torque cone).
    ground    a part whose box bottom is the build's lowest: push-only at its footprint corners + 4 friction presses.
  rbe LP    min W sum|residual| + ALPHA sum_i max_drag_i + BETA sum drag   s.t. A F + b = r+ - r-,  drag <= max_drag_i.
            Unstable if a body keeps a residual (no force system holds it) or its max drag >= T = 0.98 N/point.
            score_i = max drag on its own studs / T (1 if out of equilibrium).
  link LP   adds a free vertical shear lambda per side pair (lambda+ - lambda-), caps drag <= T, and minimises
            sum|lambda| + W sum|residual|  (shortcut: L1 instead of StableLego's sum lambda^2 -- no QP solver is
            installed; add osqp/clarabel for the least-squares spread).  lambda > 0 = the build leans on side contact.

    .venv-arm/bin/python tools/weak_joints.py build.ldr [--json out.json] [--per-component] [--validity validity.json]
    .venv-arm/bin/python tools/weak_joints.py --selftest
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import linprog
from scipy.sparse.csgraph import connected_components

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parent / "set_study")]

T, ALPHA, BETA, G = 0.98, 1e-3, 1e-6, 9.8          # StableLego constants (N per contact point, objective weights)
KG_PER_CELL = 1e-4                                   # 0.1 g per stud x stud x plate
TOL_F, TOL_T, TOUCH = 1e-4, 1e-4, 0.5                # residual tolerances (N, N stud); face-contact tolerance in LDU
W_RES = 1e3                                          # residual dominates: a force system is used whenever one exists
FOUR = np.array([(0, -.25), (-.25, 0), (0, .25), (.25, 0)])
THREE = np.array([(.125, -.125), (-.25, 0), (.125, .125)])
UP = np.array([0.0, -1.0, 0.0])                      # LDraw: -y is up
KINDS = ("normal", "drag", "press", "rest", "side", "ground", "link+", "link-")


def boxes_of(g) -> tuple[np.ndarray, np.ndarray, int]:
    """World AABBs (n, 3) lo / hi in LDU without studs, and how many parts fell back to a 1x1 plate box."""
    from bricknet import load_catalog
    from ldraw_read import part_bbox
    stems, lo, hi, missing = load_catalog().id_to_stem, [], [], 0
    for pid, M in zip(g.part_ids, g.transforms):
        bb = part_bbox(stems[pid] + ".dat")
        if bb is None: missing += 1; bb = (np.array([-10.0, 0, -10]), np.array([10.0, 8, 10]))
        a, b = bb[0].copy(), bb[1]
        if b[1] > 0: a[1] = max(a[1], 0.0)                                     # drop the studs above the body
        c = np.array([[x, y, z, 1] for x in (a[0], b[0]) for y in (a[1], b[1]) for z in (a[2], b[2])])
        w = (M @ c.T).T[:, :3]
        lo.append(w.min(0)); hi.append(w.max(0))
    return np.array(lo), np.array(hi), missing


def touching_pairs(lo, hi):
    """Candidate (i, j) with boxes within TOUCH of each other: sweep along x, then exact tests by the caller."""
    order = np.argsort(lo[:, 0]); out = []
    los, his = lo[order], hi[order]
    for k in range(len(order)):
        m = np.searchsorted(los[:, 0], his[k, 0] + TOUCH, side="right")
        for q in range(k + 1, m):
            if (los[q] <= his[k] + TOUCH).all() and (los[k] <= his[q] + TOUCH).all(): out.append((order[k], order[q]))
    return out


def generators(g, lo, hi):
    """Every force generator: dict of arrays a, b (-1 = ground / none), pos (LDU), dir, kind, joint (edge / pair id)."""
    from bricknet.graph import _part_info
    A, B, P, D, K, J = [], [], [], [], [], []

    def add(a, b, pos, d, kind, joint):
        A.append(a); B.append(b); P.append(pos); D.append(d); K.append(KINDS.index(kind)); J.append(joint)

    stud_pairs = set()
    for e_ix, e in enumerate(g.edges):                                     # all-stud in our builds; other families
        a, b = int(e["a"]), int(e["b"])                                    # (hinge, clip, axle) ride the same model
        stud_pairs.add((min(a, b), max(a, b)))
        fs = g.transforms[a] @ _part_info(g.part_ids[a]).frames[int(e["a_conn"])]
        c, u = fs[:3, 3], -fs[:3, 1] / np.linalg.norm(fs[:3, 1])         # stud tip direction, toward the hole owner
        e1 = fs[:3, 0] / np.linalg.norm(fs[:3, 0]); e2 = np.cross(u, e1)
        ext = hi[b] - lo[b]
        across = ext - np.abs(ext @ u) * np.abs(u)                         # hole owner's extent across the axis
        width = min(abs(across @ e1), abs(across @ e2)) / 20
        for ox, oy in (FOUR if width < 1.5 else THREE):
            p = c + 20 * (ox * e1 + oy * e2)
            add(b, a, p, u, "normal", e_ix); add(b, a, p, -u, "drag", e_ix)
        for d in (e1, -e1, e2, -e2): add(b, a, c, d, "press", e_ix)
    n, y_ground = len(lo), hi[:, 1].max()
    for i in range(n):                                                     # ground: the build's lowest bottom face
        if abs(hi[i, 1] - y_ground) < TOUCH:
            for x in (lo[i, 0], hi[i, 0]):
                for z in (lo[i, 2], hi[i, 2]): add(i, -1, np.array([x, y_ground, z]), UP, "ground", -1)
            mid = np.array([(lo[i, 0] + hi[i, 0]) / 2, y_ground, (lo[i, 2] + hi[i, 2]) / 2])
            for d in np.eye(3)[[0, 2]]: add(i, -1, mid, d, "ground", -1); add(i, -1, mid, -d, "ground", -1)
    pairs = []
    for i, j in touching_pairs(lo, hi):
        ov = np.minimum(hi[i], hi[j]) - np.maximum(lo[i], lo[j])          # >0 overlap, ~0 touching
        if ov[0] > TOUCH and ov[2] > TOUCH and abs(ov[1]) < TOUCH:       # vertical face contact
            up, dn = (i, j) if lo[i, 1] < lo[j, 1] else (j, i)
            if (min(i, j), max(i, j)) in stud_pairs: continue
            y = hi[up, 1]; x0, x1 = max(lo[i, 0], lo[j, 0]), min(hi[i, 0], hi[j, 0]); z0, z1 = max(lo[i, 2], lo[j, 2]), min(hi[i, 2], hi[j, 2])
            for x in (x0, x1):
                for z in (z0, z1): add(up, dn, np.array([x, y, z]), UP, "rest", -1)
            continue
        for k, t in ((0, 2), (2, 0)):                                      # side face contact along x or z
            if abs(ov[k]) < TOUCH and ov[1] > TOUCH and ov[t] > TOUCH:
                neg, pos_ = (i, j) if lo[i, k] < lo[j, k] else (j, i)
                s = hi[neg, k]; y0, y1 = max(lo[i, 1], lo[j, 1]), min(hi[i, 1], hi[j, 1])
                t0, t1 = max(lo[i, t], lo[j, t]), min(hi[i, t], hi[j, t])
                ek = np.eye(3)[k]
                for y in (y0, y1):
                    for tt in (t0, t1):
                        p = np.zeros(3); p[k], p[1], p[t] = s, y, tt
                        add(pos_, neg, p, ek, "side", len(pairs))
                p = np.zeros(3); p[k], p[1], p[t] = s, (y0 + y1) / 2, (t0 + t1) / 2
                add(pos_, neg, p, UP, "link+", len(pairs)); add(pos_, neg, p, -UP, "link-", len(pairs))
                pairs.append((neg, pos_, p))
    gen = {k: np.array(v) for k, v in zip("abpdkj", (A, B, P, D, K, J))}
    return gen, pairs


def assemble(gen, lo, hi):
    """A (6n x m) and b so that A F + b is every body's (force, torque) residual."""
    n, m = len(lo), len(gen["a"])
    ctr, vol = (lo + hi) / 2, np.prod((hi - lo) / [20, 8, 20], axis=1)
    rows, cols, vals = [], [], []
    for body, sign in ((gen["a"], 1.0), (gen["b"], -1.0)):
        ok = body >= 0; v = np.flatnonzero(ok); bd = body[ok]
        d = sign * gen["d"][ok]; r = (gen["p"][ok] - ctr[bd]) / 20                # levers in studs: the length unit cancels out
        ent = np.hstack([d, np.cross(r, d)])                               # (k, 6)
        rows.append((6 * bd[:, None] + np.arange(6)).ravel()); cols.append(np.repeat(v, 6)); vals.append(ent.ravel())
    A = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(6 * n, m))
    b = np.zeros(6 * n); b[6 * np.arange(n) + 1] = vol * KG_PER_CELL * G   # gravity along +y (down)
    return A, b, vol * KG_PER_CELL


def solve(A, b, gen, bodies, links: bool):
    """One LP over `bodies` (and the generators among them). Returns F (all m, zeros outside) and residual (6n)."""
    m = A.shape[1]
    inb = np.zeros(A.shape[0] // 6, bool); inb[bodies] = True
    gsel = inb[gen["a"]] & ((gen["b"] < 0) | inb[np.maximum(gen["b"], 0)])
    is_link = (gen["k"] == KINDS.index("link+")) | (gen["k"] == KINDS.index("link-"))
    if not links: gsel &= ~is_link
    gi = np.flatnonzero(gsel); ri = (6 * np.asarray(bodies)[:, None] + np.arange(6)).ravel()
    As, bs, R, nf = A[ri][:, gi], b[ri], len(ri), len(gi)
    drag = gen["k"][gi] == KINDS.index("drag")
    I = sparse.identity(R, format="csr")
    w = np.full(R, W_RES)
    if links:                                                              # min |lambda| + W |r|, drag <= T
        c = np.concatenate([np.where(is_link[gi], 1.0, BETA * drag), w, w])
        Aeq = sparse.hstack([As, I, -I], format="csr")
        ub = np.concatenate([np.where(drag, T, np.inf), np.full(2 * R, np.inf)])
        res = linprog(c, A_eq=Aeq, b_eq=-bs, bounds=np.column_stack([np.zeros_like(ub), ub]), method="highs-ipm")
    else:                                                                  # min W |r| + ALPHA max drag/body + BETA drag
        above = gen["a"][gi][drag]; ub_, inv = np.unique(above, return_inverse=True); nd = len(ub_)
        c = np.concatenate([BETA * drag, w, w, np.full(nd, ALPHA)])
        Aeq = sparse.hstack([As, I, -I, sparse.csr_matrix((R, nd))], format="csr")
        k = np.flatnonzero(drag)
        Aub = sparse.csr_matrix((np.r_[np.ones(len(k)), -np.ones(len(k))],
                                 (np.r_[np.arange(len(k)), np.arange(len(k))], np.r_[k, nf + 2 * R + inv])),
                                shape=(len(k), nf + 2 * R + nd))
        res = linprog(c, A_ub=Aub, b_ub=np.zeros(len(k)), A_eq=Aeq, b_eq=-bs, bounds=(0, None), method="highs-ipm")
    if res.x is None: raise RuntimeError(f"LP failed: {res.message}")
    F = np.zeros(m); F[gi] = res.x[:nf]
    return F


def analyse(path: Path, per_component=False, top=20) -> dict:
    import validity_check
    t0 = time.time()
    g, flat, n_file, dropped = validity_check.load(path)
    lo, hi, missing = boxes_of(g)
    gen, pairs = generators(g, lo, hi)
    A, b, mass = assemble(gen, lo, hi)
    n = len(lo); t_build = time.time() - t0
    if per_component:
        nb = gen["b"] >= 0
        adj = sparse.coo_matrix((np.ones(nb.sum()), (gen["a"][nb], gen["b"][nb])), shape=(n, n))
        _, lab = connected_components(adj, directed=False)
        groups = [np.flatnonzero(lab == c) for c in range(lab.max() + 1)]
    else:
        groups = [np.arange(n)]
    t1 = time.time(); F = sum(solve(A, b, gen, grp, False) for grp in groups); t_rbe = time.time() - t1
    t2 = time.time(); FL = sum(solve(A, b, gen, grp, True) for grp in groups); t_link = time.time() - t2
    tag = lambda i: f"#{i} {flat[i].stem} c{flat[i].color} @ {tuple(round(float(v), 1) for v in flat[i].pos)}"  # noqa: E731

    res = (A @ F + b).reshape(n, 6)
    out_eq = (np.abs(res[:, :3]).max(1) > TOL_F) | (np.abs(res[:, 3:]).max(1) > TOL_T)
    drag = gen["k"] == KINDS.index("drag")
    dmax = np.zeros(n); np.maximum.at(dmax, gen["a"][drag], F[drag])
    score = np.where(out_eq | (dmax >= T), 1.0, dmax / T)
    joints = {}                                                            # stud pair -> drag / normal / position
    for k in np.flatnonzero(drag | (gen["k"] == KINDS.index("normal"))):
        e = g.edges[gen["j"][k]]; key = (int(e["a"]), int(e["b"]))
        J = joints.setdefault(key, {"max_drag": 0.0, "drag": 0.0, "normal": 0.0, "pos": set()})
        if drag[k]: J["max_drag"] = max(J["max_drag"], F[k]); J["drag"] += F[k]
        else: J["normal"] += F[k]
        J["pos"].add(gen["j"][k])
    jl = []
    for (a, bb), J in joints.items():
        pos = np.mean([_stud_pos(g, e) for e in J["pos"]], axis=0)
        jl.append({"below": a, "above": bb, "studs": len(J["pos"]), "load": round(J["max_drag"] / T, 4),
                   "drag_n": round(J["drag"], 5), "normal_n": round(J["normal"], 5),
                   "pos_ldu": [round(float(v), 1) for v in pos], "below_part": tag(a), "above_part": tag(bb)})
    jl.sort(key=lambda j: -j["load"])
    lam = np.zeros(len(pairs))
    for k, s in ((KINDS.index("link+"), 1.0), (KINDS.index("link-"), -1.0)):
        sel = gen["k"] == k; np.add.at(lam, gen["j"][sel], s * FL[sel])
    resL = (A @ FL + b).reshape(n, 6)
    outL = (np.abs(resL[:, :3]).max(1) > TOL_F) | (np.abs(resL[:, 3:]).max(1) > TOL_T)
    links = sorted(({"a": int(i), "b": int(j), "lambda_n": round(float(l), 6), "pos_ldu": [round(float(v), 1) for v in p],
                     "a_part": tag(i), "b_part": tag(j)} for (i, j, p), l in zip(pairs, lam) if abs(l) > 1e-7),
                   key=lambda x: -abs(x["lambda_n"]))
    r = {"file": str(path), "parts": n, "not_loaded": dict(dropped), "box_fallbacks": missing,
         "generators": {k: int((gen["k"] == i).sum()) for i, k in enumerate(KINDS)}, "stud_edges": len(g.edges),
         "side_pairs": len(pairs), "lp_columns": int(A.shape[1]), "lp_rows": int(A.shape[0]), "groups": len(groups),
         "stable": bool(score.max(initial=0) < 1.0), "out_of_equilibrium": int(out_eq.sum()),
         "over_capacity": int(((dmax >= T) & ~out_eq).sum()), "max_score_in_equilibrium": round(float(score[~out_eq].max(initial=0)), 4),
         "score": [round(float(s), 4) for s in score], "unsupported": [f"{tag(i)} residual {np.abs(res[i, :3]).max():.2g} N" for i in np.flatnonzero(out_eq)],
         "weakest_joints": jl[:top], "joints": jl,
         "links_total_n": round(float(np.abs(lam).sum()), 6), "links": links,
         "link_out_of_equilibrium": [tag(i) for i in np.flatnonzero(outL)],
         "seconds": {"build": round(t_build, 2), "rbe_lp": round(t_rbe, 2), "link_lp": round(t_link, 2)}}
    r["_arrays"] = {"out_eq": out_eq, "score": score, "outL": outL, "components": g.components}
    return r


def _stud_pos(g, e):
    from bricknet.graph import _part_info
    ed = g.edges[e]
    return (g.transforms[int(ed["a"])] @ _part_info(g.part_ids[int(ed["a"])]).frames[int(ed["a_conn"])])[:3, 3]


def compare(r: dict, validity: Path) -> dict:
    """PyBullet fallen parts (validity.json 'stability', moved) vs RBE flags; indices match (same loader).
    PyBullet moves whole connector components, so the fair unit is the component: hit = any of its parts flagged."""
    v = json.loads(validity.read_text()); comps = r["_arrays"]["components"]
    fell = {i for c in v["stability"] if c["moved"] for i in comps[c["component"]]}
    rbe = set(np.flatnonzero(r["_arrays"]["out_eq"]).tolist())
    weak = rbe | {i for i, s in enumerate(r["_arrays"]["score"]) if s >= 0.5}
    f = lambda s: {"flagged": len(s), "hit": len(s & fell), "recall": round(len(s & fell) / max(1, len(fell)), 3),  # noqa: E731
                   "precision": round(len(s & fell) / max(1, len(s)), 3)}
    per = [{"component": c["component"], "parts": c["parts"], "max_ldu": c["max_ldu"], "start": c["start"],
            "rbe_flagged": sorted(set(comps[c["component"]]) & rbe)} for c in v["stability"] if c["moved"]]
    flagged_comps = {k for k, cp in enumerate(comps) if set(cp) & rbe}
    return {"pybullet_fallen_parts": len(fell), "rbe_out_of_equilibrium": f(rbe), "rbe_score_ge_0.5": f(weak),
            "fallen_components": len(per), "fallen_components_hit": sum(bool(c["rbe_flagged"]) for c in per),
            "rbe_flagged_components": len(flagged_comps),
            "rbe_flagged_components_fallen": len(flagged_comps & {c["component"] for c in per}),
            "components": per, "flagged_not_fallen": sorted(rbe - fell)}


def selftest() -> None:
    import tempfile

    def run(body: str) -> dict:
        with tempfile.NamedTemporaryFile("w", suffix=".ldr", delete=False) as f: f.write(body)
        try: return analyse(Path(f.name))
        finally: Path(f.name).unlink()
    p = "1 4 {} {} {} 1 0 0 0 1 0 0 0 1 {}.dat\n"                         # LDraw origin = top face; ground y = 0
    r = run(p.format(0, -24, 0, 3001) + p.format(0, -48, 0, 3001))                        # 2x4 on 2x4 on the ground
    assert r["stable"] and r["out_of_equilibrium"] == 0 and r["max_score_in_equilibrium"] < 0.01, r["weakest_joints"][:1]
    base = p.format(50, -8, -10, 3958)                       # 6x6 plate (studs at z = +-10) under it all: a free column would tip
    column = lambda x: "".join(p.format(x, -8 - 24 * k, 0, 3005) for k in (1, 2, 3))     # noqa: E731  1x1 column, top -80
    weights = lambda x: "".join(p.format(x, -112 - 24 * k, 0, 3005) for k in range(4))  # noqa: E731
    plate = p.format(50, -88, 0, 3666)                                                    # 1x6, studs x = 0 .. 100
    r = run(base + column(0) + plate + weights(100))         # cantilever: held by its end stud, 4 bricks on the far end
    w = r["weakest_joints"][0]
    assert r["out_of_equilibrium"] == 0 and (w["below"], w["above"]) == (3, 4), w
    tower = max(j["load"] for j in r["joints"] if j["pos_ldu"][0] == 100)               # the weights' own studs
    assert w["load"] > 0.05 and w["load"] > 10 * tower, r["joints"]                      # the end stud carries the moment
    cant = w["load"]
    r = run(base + column(0) + column(100) + plate + weights(50))   # bridge: same plate on two columns, load mid-span
    assert r["stable"] and r["out_of_equilibrium"] == 0 and r["weakest_joints"][0]["load"] < cant / 3, r["weakest_joints"][:2]
    r = run(p.format(0, -24, 0, 3005) + p.format(20, -40, 0, 3005))                       # hangs beside a brick
    assert r["out_of_equilibrium"] == 1 and r["links"] and abs(r["links"][0]["lambda_n"] - 0.3e-3 * G) < 1e-5, r["links"]  # its weight (torque stays unpatched: a shear link holds no moment)
    print(f"selftest ok (cantilever end-stud load {cant:.2f} T)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ldr", type=Path, nargs="?"); ap.add_argument("--json", type=Path)
    ap.add_argument("--per-component", action="store_true"); ap.add_argument("--validity", type=Path)
    ap.add_argument("--top", type=int, default=10); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest: return selftest()
    r = analyse(a.ldr, a.per_component)
    if a.validity: r["pybullet"] = compare(r, a.validity)
    r.pop("_arrays")
    if a.json: a.json.write_text(json.dumps(r, indent=1))
    print(json.dumps({k: v for k, v in r.items() if k not in ("score", "joints", "links", "weakest_joints", "unsupported",
                                                              "link_out_of_equilibrium")}, indent=1))
    print(f"unsupported ({len(r['unsupported'])}):", *r["unsupported"][:a.top], sep="\n  ")
    print("weakest joints (max drag / T):")
    for j in r["weakest_joints"][:a.top]: print(f"  {j['load']:.3f}  {j['studs']} stud(s) @ {j['pos_ldu']}  {j['below_part']} -> {j['above_part']}")
    print(f"side links (sum {r['links_total_n']} N):")
    for l in r["links"][:a.top]: print(f"  {l['lambda_n']:+.4f} N @ {l['pos_ldu']}  {l['a_part']} | {l['b_part']}")


if __name__ == "__main__":
    main()
