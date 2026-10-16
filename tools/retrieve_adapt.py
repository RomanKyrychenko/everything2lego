"""Set retrieval + adaptation: for a target mesh, pick the closest real
OMR set and seed the sampler state with that set's actual placement
list — scaled uniformly to fit the target's bbox, cropped to the
target's voxel grid, re-coloured to the mesh palette.  Sampler then
fills the uncovered cells.

This is a *fundamental* departure from "sample placements from scratch
by cell statistics".  The seed is a real set, so the build starts
set-quality; the sampler only has to touch the uncovered gaps.

    python tools/retrieve_adapt.py mesh.glb out/ --studs 32 --kind building
        [--pick 0]   # which rank to retrieve (0 = top)
        [--scale uniform]   # uniform | fit (per-axis; drops non-standard-size parts)

Honest limits:
- OMR placements use arbitrary rotations in some sub-models; only the
  yaw-aligned (0/90/180/270) subset survives the import.
- Uniform scale leaves the adapted set smaller than target along the
  stretched axes; the sampler fills the rest.  "fit" is non-uniform;
  parts whose footprint would end up non-standard get dropped.
- Colour of each imported placement is kept from the OMR source;
  the sampler's own colour penalty will contest cells where the mesh
  disagrees (which is usually fine — real-set colours look better).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools/set_study"))

from sampler_demo import target_from_mesh, load_models, write_ldr  # noqa: E402
from everything2lego.pack import Placement  # noqa: E402
from everything2lego.sampler import Weights, sample, vocabulary, nearest_models  # noqa: E402
from everything2lego.io_ldraw import _ROT  # noqa: E402
from everything2lego.colors import palette  # noqa: E402

OMR = Path("/Volumes/PRO-G40/lego_data/omr/files")


def _best_yaw(R: np.ndarray) -> int | None:
    """Match a 3x3 rotation to one of 0/90/180/270 yaws (about world Y).
    Returns the matched yaw in degrees, or None if the rotation is tilted."""
    best, best_err = None, 0.3
    for yaw, m in _ROT.items():
        ref = np.array(m, float).reshape(3, 3)
        err = np.linalg.norm(R - ref)
        if err < best_err:
            best_err, best = err, yaw
    return best


def _ldr_colour_to_palette(code: int) -> int:
    """Map an LDraw colour code to our palette index (or -1)."""
    for i, c in enumerate(palette()):
        if c.code == code:
            return i
    return -1


def import_omr_placements(ldr_path: Path, by_ldraw: dict) -> list[Placement]:
    """Load the OMR .mpd/.ldr and return a list of Placements for parts
    that are in our vocabulary with yaw-aligned rotations."""
    from ldraw_read import flatten
    parts = flatten(ldr_path)
    out: list[Placement] = []
    for p in parts:
        v = by_ldraw.get(p.name)
        if v is None:
            continue
        R = p.M[:3, :3]
        matched = _best_yaw(R)
        if matched is None:
            continue
        rot = (matched - v.part.yaw) % 360
        if rot % 90 != 0:
            continue
        # footprint rotated by yaw
        fw, fd = v.part.footprint
        if rot in (90, 270):
            w_world, d_world = fd, fw
        else:
            w_world, d_world = fw, fd
        # invert io_ldraw.placement_to_line formula
        m = _ROT[matched]
        ox, oy, oz = v.part.origin
        x_ldr, y_ldr, z_ldr = p.M[0, 3], p.M[1, 3], p.M[2, 3]
        x_prime = x_ldr + m[0] * ox + m[2] * oz
        z_prime = z_ldr + m[6] * ox + m[8] * oz
        y_prime = y_ldr + oy
        x_stud = int(round((x_prime / 10 - w_world + 1) / 2))
        z_stud = int(round((z_prime / 10 - d_world + 1) / 2))
        y_plate = int(round(-y_prime / 8 - v.part.height_units))
        colour_idx = _ldr_colour_to_palette(p.colour)
        if colour_idx < 0:
            colour_idx = 0
        out.append(Placement(v.part, colour_idx, x_stud, z_stud, y_plate,
                              w_world, d_world, rot))
    return out


def _bbox(placements: list[Placement]) -> tuple[int, int, int, int, int, int]:
    """(y_min, y_max, x_min, x_max, z_min, z_max) over placement cells."""
    ymin, ymax = min(p.y_plate for p in placements), max(p.y_plate + p.part.height_units for p in placements)
    xmin, xmax = min(p.x_stud for p in placements), max(p.x_stud + p.w_world for p in placements)
    zmin, zmax = min(p.z_stud for p in placements), max(p.z_stud + p.d_world for p in placements)
    return ymin, ymax, xmin, xmax, zmin, zmax


def adapt_to_target(placements: list[Placement], target_shape: tuple[int, int, int],
                      want: np.ndarray) -> tuple[list[Placement], dict]:
    """Scale + translate OMR placements to fit the target voxel grid.
    Uniform scale (= min of per-axis scales) so parts keep their size.
    Drops placements that fall outside target.want after transform."""
    H, X, Z = target_shape
    # target bbox from want
    ys, xs, zs = np.nonzero(want)
    tgt_y0, tgt_y1 = int(ys.min()), int(ys.max()) + 1
    tgt_x0, tgt_x1 = int(xs.min()), int(xs.max()) + 1
    tgt_z0, tgt_z1 = int(zs.min()), int(zs.max()) + 1
    tgt_h, tgt_w, tgt_d = tgt_y1 - tgt_y0, tgt_x1 - tgt_x0, tgt_z1 - tgt_z0
    sy, syx, syz, syw, sywx, sywz = _bbox(placements)
    omr_h, omr_w, omr_d = sywx - sy, syx - sy, syz - sy  # placeholder; recompute
    y0, y1, x0, x1, z0, z1 = _bbox(placements)
    omr_h, omr_w, omr_d = y1 - y0, x1 - x0, z1 - z0
    if omr_h <= 0 or omr_w <= 0 or omr_d <= 0:
        return [], {"source_parts": len(placements), "kept": 0, "scale": 1.0}
    sy = tgt_h / omr_h
    sx = tgt_w / omr_w
    sz = tgt_d / omr_d
    scale = min(sy, sx, sz)                           # uniform; parts keep their shape
    # recentre: scale OMR to [0, omr * scale], then shift to target bbox centre
    def _tx(p: Placement) -> Placement:
        new_y = int(round((p.y_plate - y0) * scale)) + tgt_y0
        new_x = int(round((p.x_stud - x0) * scale)) + tgt_x0
        new_z = int(round((p.z_stud - z0) * scale)) + tgt_z0
        return Placement(p.part, p.color_idx, new_x, new_z, new_y,
                          p.w_world, p.d_world, p.rotation_deg)
    out = []
    occupied: set[tuple[int, int, int]] = set()
    for p in placements:
        q = _tx(p)
        cells = []
        ok = True
        for cy in range(q.y_plate, q.y_plate + q.part.height_units):
            for cx in range(q.x_stud, q.x_stud + q.w_world):
                for cz in range(q.z_stud, q.z_stud + q.d_world):
                    if cy < 0 or cy >= H or cx < 0 or cx >= X or cz < 0 or cz >= Z:
                        ok = False; break
                    if not want[cy, cx, cz]:
                        ok = False; break
                    if (cy, cx, cz) in occupied:
                        ok = False; break
                    cells.append((cy, cx, cz))
                if not ok: break
            if not ok: break
        if ok:
            for c in cells:
                occupied.add(c)
            out.append(q)
    return out, {"source_parts": len(placements), "kept": len(out), "scale": scale}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh")
    ap.add_argument("out", type=Path)
    ap.add_argument("--studs", type=int, default=32)
    ap.add_argument("--shell", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kind", default=None)
    ap.add_argument("--style", type=float, default=1.0)
    ap.add_argument("--segmentation", default=None)
    ap.add_argument("--pick", type=int, default=0, help="which rank to retrieve (0 = top)")
    ap.add_argument("--iters", type=int, default=0, help="fill iters; 0 = auto")
    ap.add_argument("--prune-unsupported", action="store_true")
    a = ap.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)

    seg_label, seg_kof, carve = None, None, None
    if a.segmentation:
        d = np.load(a.segmentation, allow_pickle=True)
        seg_label = d["label"]
        seg_kof = {int(k): v for k, v in d["kind_of"]}
        if "carve" in d.files:
            carve = d["carve"]
    target = target_from_mesh(a.mesh, a.studs, a.shell, a.kind,
                               seg_label=seg_label, seg_kind_of=seg_kof, carve=carve)

    models = load_models()
    refs = nearest_models(models, target.shape_descriptor(), k=20, kind=target.kind,
                           target_vol=float(target.want.sum()))
    print(f"top {min(5, len(refs))} refs by shape:")
    for r in refs[:5]:
        print(f"  {r['model']}  n_parts={r.get('n_parts', '?')}  kind={r.get('object_kind', '?')}")

    pick = refs[a.pick]
    ldr_hits = list(OMR.glob(f"{pick['model']}.mpd")) + list(OMR.glob(f"{pick['model']}.ldr"))
    if not ldr_hits:
        raise SystemExit(f"OMR file for {pick['model']!r} not found under {OMR}")
    print(f"picked: {pick['model']} -> {ldr_hits[0].name}")

    vocab = vocabulary()
    by_ldraw = {v.part.ldraw: v for v in vocab}
    src_placements = import_omr_placements(ldr_hits[0], by_ldraw)
    adapted, adapt_info = adapt_to_target(src_placements, target.want.shape, target.want)
    print(f"adapted: imported {adapt_info['source_parts']} vocab-matched placements, "
          f"kept {adapt_info['kept']} after scaling {adapt_info['scale']:.2f}x and bbox-crop")

    iters = a.iters or max(20000, 40 * int(target.want.sum()))
    weights = Weights(style=a.style)
    print(f"=== filling gaps @ {a.studs} studs, {iters} iters, seeded with {len(adapted)} placements ===", flush=True)
    t0 = time.time()
    ps, info = sample(target, models, iters=iters, seed=a.seed, weights=weights,
                       init_placements=adapted)
    secs = time.time() - t0
    print(f"filled: parts={info['parts']} kl={info['kl']:.2f} fid={info['fidelity']:.0f} ({secs:.1f}s)", flush=True)

    if a.prune_unsupported:
        from everything2lego.supports import prune_unsupported
        ps, pruned = prune_unsupported(list(ps))
        info["pruned_unsupported"] = pruned
        info["parts"] = len(ps)

    write_ldr(ps, a.out / "model.ldr")
    info.update({"retrieved": pick["model"], "kept_from_source": adapt_info["kept"],
                 "source_parts": adapt_info["source_parts"], "scale": adapt_info["scale"],
                 "seconds": round(secs, 1)})
    (a.out / "info.json").write_text(json.dumps({k: v for k, v in info.items() if k != "history"}, indent=1, default=float))
    print(json.dumps({k: v for k, v in info.items() if k != "history"}, default=float))
    import subprocess
    subprocess.run([sys.executable, str(ROOT / "tools/render_build.py"), str(a.out / "model.ldr"),
                    str(a.out / "render.png"), "--views", "30,120,210", "--size", "500"], check=False)


if __name__ == "__main__":
    main()
