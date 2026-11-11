"""Physical validity of an LDraw build with BrickAgent's checks (github.com/kulits/BrickAgent), reported in full.

brickagent.check() raises one ValueError on the first failing stage (stability is skipped while anything collides,
and "inexact" only covers authored contacts), so this calls the same pieces directly and always runs every stage:

    loaded      parts BrickNet can read (catalogue part / alias, unscaled, has a convex collider) vs parts in the file
    collisions  BrickNet convex-collider pairs (bricknet.collision.colliding_pairs, BrickAgent's tolerance)
    inexact     detected connections whose pose misses the exact mate (BrickNet accepts near misses)
    components  connector-connected pieces (BrickNet graph); each is one rigid body in the simulation
    stability   brickagent.physics._simulate: 2 s PyBullet gravity per component; unstable if a point moves > 3 LDU
                (a fallen component is removed at that moment, so its max_ldu is where it crossed the limit).
                Run even with collisions (check() skips it): interpenetrating components push apart, and a loose part
                can be knocked over by a falling neighbour, so read stability next to the collision list.
    Caveat: inexact also flags mates off by a part's symmetry (e.g. wheel 6014a / tyre 6015 at 180 deg).

    .venv-arm/bin/python tools/validity_check.py build.ldr [--json out.json] [--no-stability]
    .venv-arm/bin/python tools/validity_check.py --selftest
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

os.environ.setdefault("BRICKNET_DATA", "/Volumes/PRO-G40/lego_data/bricknet")   # colliders in <data>/cvx
sys.path[:0] = [str(Path(__file__).resolve().parent / "set_study")]

SCALE_TOL = 0.05        # as brickae_data: column-norm deviation we orthonormalise away
EXACT_LDU, EXACT_DEG = 0.01, 0.1   # LDR files carry ~6 digits; below this a mate counts as exact


def load(path: Path):
    """-> (bricknet Graph over the readable parts, brickagent Placed list in the same order, n in file, dropped names)."""
    from bricknet import load_catalog
    from bricknet.collision import _path
    from bricknet.data import load_aliases
    from bricknet.graph import parse_ldr
    from brickagent.model import Placed
    from ldraw_read import flatten
    cat, aliases = load_catalog(), load_aliases()
    lines, dropped, allp = [], Counter(), flatten(path)
    for p in allp:
        st = p.name.lower().removesuffix(".dat"); canon = aliases[st][0] if st in aliases else st
        try: ok = canon in cat.stem_to_id and _path(cat.stem_to_id[canon]) and np.abs(np.linalg.norm(p.M[:3, :3], axis=0) - 1).max() <= SCALE_TOL
        except FileNotFoundError: ok = False                                    # catalogue part without a collider
        if not ok: dropped[st] += 1; continue
        lines.append(f"1 {p.colour} " + " ".join(f"{v:.6g}" for v in [*p.M[:3, 3], *p.M[:3, :3].ravel()]) + f" {st}.dat")
    g = parse_ldr("\n".join(lines) + "\n")                                     # resolves aliases, orthonormalises
    flat = [Placed(cat.id_to_stem[pid], c, tuple(M[:3, :3].ravel()), tuple(M[:3, 3]), (str(i),))
            for i, (pid, c, M) in enumerate(zip(g.part_ids, g.colors, g.transforms))]
    return g, flat, len(allp), dropped


def inexact(g) -> list[tuple[int, int, float, float]]:
    """Detected edges whose pose misses the exact mate: (i, j, LDU, degrees); same residual as brickagent.check."""
    from bricknet.graph import _edge_matrix
    out, M = [], g.transforms
    for e in g.edges:
        i, j = int(e["a"]), int(e["b"])
        d = M[i] @ _edge_matrix(e, g.part_ids[i], g.part_ids[j]) - M[j]
        ldu, deg = float(np.linalg.norm(d[:3, 3])), float(np.degrees(2 * np.arcsin(min(1, np.linalg.norm(d[:3, :3]) / np.sqrt(8)))))
        if ldu > EXACT_LDU or deg > EXACT_DEG: out.append((i, j, ldu, deg))
    return out


def validity(path: Path, stability: bool = True, limit: float = 3.0, examples: int = 8) -> dict:
    from bricknet.collision import colliding_pairs
    t0 = time.time()
    g, flat, n_file, dropped = load(path)
    tag = lambda i: f"#{i} {flat[i].stem} c{flat[i].color} @ {tuple(round(v, 1) for v in flat[i].pos)}"   # noqa: E731
    r = {"file": str(path), "parts_in_file": n_file, "parts_loaded": len(flat), "not_loaded": dict(dropped.most_common())}
    coll = colliding_pairs(g.part_ids, g.transforms) if flat else []
    r["collisions"] = len(coll)
    r["collision_parts"] = len({i for p in coll for i in p})
    r["collision_kinds"] = dict(Counter(" <> ".join(sorted((flat[a].stem, flat[b].stem))) for a, b in coll).most_common(10))
    r["collision_examples"] = [f"{tag(a)} <> {tag(b)}" for a, b in coll[:examples]]
    bad = inexact(g) if flat else []
    r["connections"], r["inexact"] = len(g.edges), len(bad)
    r["inexact_examples"] = [f"{tag(i)} -> {tag(j)}: {ldu:.3g} LDU, {deg:.3g} deg" for i, j, ldu, deg in bad[:examples]]
    comps = g.components
    r["components"], r["component_sizes"] = len(comps), [len(c) for c in comps[:20]]
    r["seconds_load_collide"] = round(time.time() - t0, 1)
    if stability and flat:
        from brickagent.check import _support
        from brickagent.physics import _mesh, _simulate
        t1 = time.time()
        motion, witness, heights, _, _ = _simulate(flat, comps, max_displacement=limit)
        pts = [_mesh(p.stem)[0] @ np.reshape(p.rot, (3, 3)).T + p.pos for p in flat]
        box = np.array([(q.min(0), q.max(0)) for q in pts])                    # world collider bounds, +y down

        def start(j):                                                           # brickagent.check's wording
            if heights[j] <= 0.01: return "on the ground"
            gap, below = _support(box, comps[j])
            return "nothing beneath" if below < 0 else f"{gap:g} LDU above {tag(below)}" if gap > 0.5 else f"resting on {tag(below)} without a connection"

        fell = np.flatnonzero(motion > limit)
        r["unstable_components"], r["unstable_parts"] = len(fell), int(sum(len(comps[j]) for j in fell))
        r["stability"] = [{"component": int(j), "parts": len(comps[j]), "moved": bool(motion[j] > limit),
                           "max_ldu": round(float(motion[j]), 2), "above_ground_ldu": round(float(heights[j]), 1),
                           "furthest": tag(int(witness[j])), "start": start(j),
                           "part_kinds": dict(Counter(flat[i].stem for i in comps[j]).most_common(5))}
                          for j in sorted(range(len(comps)), key=lambda j: -motion[j])]
        r["seconds_stability"] = round(time.time() - t1, 1)
    return r


def selftest() -> None:
    import tempfile
    def run(body: str) -> dict:
        with tempfile.NamedTemporaryFile("w", suffix=".ldr", delete=False) as f: f.write(body)
        try: return validity(Path(f.name))
        finally: Path(f.name).unlink()
    b = "1 4 {} {} 0 1 0 0 0 1 0 0 0 1 3001.dat\n"
    ok = run(b.format(0, 0) + b.format(0, -24))                                 # 2x4 on a 2x4
    assert ok["collisions"] == 0 and ok["components"] == 1 and ok["unstable_components"] == 0 and ok["inexact"] == 0, ok
    hit = run(b.format(0, 0) + b.format(10, 0))                                 # overlapping half a brick
    assert hit["collisions"] == 1 and hit["components"] == 2, hit
    fall = run(b.format(0, 0) + b.format(0, -60))                               # floating 36 LDU above: falls
    assert fall["unstable_components"] == 1 and fall["collisions"] == 0, fall
    print("selftest ok")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ldr", type=Path, nargs="?"); ap.add_argument("--json", type=Path)
    ap.add_argument("--no-stability", action="store_true"); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest: return selftest()
    r = validity(a.ldr, stability=not a.no_stability)
    if a.json: a.json.write_text(json.dumps(r, indent=1))
    s = {k: v for k, v in r.items() if k != "stability"}
    print(json.dumps(s, indent=1))
    for c in [c for c in r.get("stability", []) if c["moved"]][:10]: print("unstable:", json.dumps(c))


if __name__ == "__main__":
    main()
