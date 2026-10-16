"""Multi-scale 3D->LEGO: a coarse pass commits global structure at half
resolution, then the fine pass fills in local detail starting from the
coarse build's scaled-up placements.  The sampler's warmup still runs
at full resolution so the Metropolis temperature is apt.

    python tools/multi_scale.py mesh.glb out/ --studs 32 [--coarse-studs 16]
        [--kind building] [--segmentation seg32.npz] [--coarse-seg seg16.npz]
        [--iters N] [--coarse-iters N] [--seed S] [plus most sampler_demo flags]

Why it helps: at coarse res the sampler commits the overall massing
cheaply.  The fine run doesn't rebuild from scratch — scaled coarse
placements seed the state, and the fine sampler only has local
decisions to re-make around them.  Observed effect: fewer global
mistakes (misplaced walls, chimneys in the wrong column) because the
fine pass can't walk away from a bad global config at low temperature.

shortcut: scaling plants the coarse part at (2y, 2x, 2z) in the fine
grid — that part only covers 1/8 of the equivalent fine volume, the
rest fills around it.  A true part-scaled seed (1x1 -> 2x2) would
need a vocabulary lookup for 2× every used part; this is a credible
first version.  Rust mirror is in place (native engine uses the same
init path).
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from sampler_demo import target_from_mesh, load_models, write_ldr  # noqa: E402
from everything2lego.pack import Placement  # noqa: E402
from everything2lego.sampler import Weights, sample  # noqa: E402


def _scale_up(ps: list[Placement], factor: int) -> list[Placement]:
    """Each coarse placement goes to (factor*y, factor*x, factor*z) in the
    fine grid, same part/colour/yaw.  Simple; see module docstring for
    the limitation this inherits."""
    return [Placement(p.part, p.color_idx, p.x_stud * factor, p.z_stud * factor,
                       p.y_plate * factor, p.w_world, p.d_world, p.rotation_deg)
            for p in ps]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh")
    ap.add_argument("out", type=Path)
    ap.add_argument("--studs", type=int, default=32)
    ap.add_argument("--coarse-studs", type=int, default=0, help="default: studs // 2")
    ap.add_argument("--iters", type=int, default=0, help="fine-pass iters; 0 = auto")
    ap.add_argument("--coarse-iters", type=int, default=0, help="coarse-pass iters; 0 = auto")
    ap.add_argument("--shell", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kind", default=None)
    ap.add_argument("--style", type=float, default=1.0)
    ap.add_argument("--stand", type=int, default=0)
    ap.add_argument("--regions", default=None)
    ap.add_argument("--segmentation", default=None, help="fine-pass segmentation .npz")
    ap.add_argument("--coarse-seg", default=None, help="coarse-pass segmentation .npz (matched to --coarse-studs)")
    ap.add_argument("--tile-boost", type=float, default=0.0)
    ap.add_argument("--snot-boost", type=float, default=0.0)
    ap.add_argument("--prune-unsupported", action="store_true")
    a = ap.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    coarse_studs = a.coarse_studs or a.studs // 2
    assert coarse_studs > 0 and a.studs % coarse_studs == 0, \
        f"--studs ({a.studs}) must be a multiple of --coarse-studs ({coarse_studs})"
    factor = a.studs // coarse_studs

    def _load_seg(path):
        if not path:
            return None, None, None
        d = np.load(path, allow_pickle=True)
        carve = d["carve"] if "carve" in d.files else None
        return d["label"], {int(k): v for k, v in d["kind_of"]}, carve

    models = load_models()
    weights = Weights(style=a.style)

    # coarse pass
    seg_label, seg_kof, carve = _load_seg(a.coarse_seg)
    t_coarse = target_from_mesh(a.mesh, coarse_studs, a.shell, a.kind, stand=a.stand,
                                 regions=None, seg_label=seg_label, seg_kind_of=seg_kof, carve=carve)
    coarse_iters = a.coarse_iters or max(10000, 40 * int(t_coarse.want.sum()))
    print(f"=== coarse pass @ {coarse_studs} studs, {coarse_iters} iters ===", flush=True)
    t0 = time.time()
    coarse_ps, coarse_info = sample(t_coarse, models, iters=coarse_iters, seed=a.seed,
                                     weights=weights, tile_boost=a.tile_boost, snot_boost=a.snot_boost)
    coarse_secs = time.time() - t0
    print(f"coarse: parts={coarse_info['parts']} kl={coarse_info['kl']:.2f} fid={coarse_info['fidelity']:.0f} "
          f"({coarse_secs:.1f}s)", flush=True)

    # fine pass seeded by scaled coarse placements
    seg_label, seg_kof, carve = _load_seg(a.segmentation)
    t_fine = target_from_mesh(a.mesh, a.studs, a.shell, a.kind, stand=a.stand,
                               regions=None, seg_label=seg_label, seg_kind_of=seg_kof, carve=carve)
    fine_iters = a.iters or max(20000, 40 * int(t_fine.want.sum()))
    seed = _scale_up(coarse_ps, factor)
    print(f"=== fine pass @ {a.studs} studs, {fine_iters} iters, seeding {len(seed)} placements ===", flush=True)
    t0 = time.time()
    fine_ps, fine_info = sample(t_fine, models, iters=fine_iters, seed=a.seed,
                                 weights=weights, tile_boost=a.tile_boost, snot_boost=a.snot_boost,
                                 init_placements=seed)
    fine_secs = time.time() - t0
    print(f"fine: parts={fine_info['parts']} kl={fine_info['kl']:.2f} fid={fine_info['fidelity']:.0f} "
          f"({fine_secs:.1f}s)", flush=True)

    if a.prune_unsupported:
        from everything2lego.supports import prune_unsupported
        fine_ps, pruned = prune_unsupported(list(fine_ps))
        fine_info["pruned_unsupported"] = pruned
        fine_info["parts"] = len(fine_ps)

    write_ldr(fine_ps, a.out / "model.ldr")
    fine_info["coarse_pass"] = {"studs": coarse_studs, "parts": coarse_info["parts"],
                                 "kl": coarse_info["kl"], "fidelity": coarse_info["fidelity"],
                                 "seconds": round(coarse_secs, 1)}
    fine_info["seeded_from_coarse"] = len(seed)
    fine_info["seconds"] = round(fine_secs, 1)
    (a.out / "info.json").write_text(json.dumps({k: v for k, v in fine_info.items()
                                                  if k != "history"}, indent=1, default=float))
    print(json.dumps({k: v for k, v in fine_info.items() if k != "history"}, default=float))
    import subprocess
    subprocess.run([sys.executable, str(ROOT / "tools/render_build.py"), str(a.out / "model.ldr"),
                    str(a.out / "render.png"), "--views", "30,120,210", "--size", "500"], check=False)


if __name__ == "__main__":
    main()
