"""Sample a build of any mesh with the statistics sampler and render it.

    python tools/sampler_demo.py mesh.glb out_dir [--studs 24] [--iters 20000] [--kind building] [--style 1]

The target is the mesh's voxel shell (real sets are hollow): the solid's
cells within `--shell` studs of its surface, coloured by the mesh, each
with the mesh's shape in it (4x4x4 samples) — so a sloped or rounded
surface asks for slopes and rounds, where it is in the cell.  `--kind` is what
the object is (vocabulary.json's kinds: building, road_vehicle, creature,
...); the reference models are official models of that kind.
"""
from __future__ import annotations

import os
import argparse
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from everything2lego.io_ldraw import write_ldr  # noqa: E402
from everything2lego.sampler import FULL, SUB, Target, popcount, sample  # noqa: E402
from everything2lego.semantic_v2.massing import solid_volume  # noqa: E402
from everything2lego.semantic_v2.mesh_target import MeshTarget  # noqa: E402

FEATURES = Path(os.environ.get("LEGO_DATA", "/Volumes/PRO-G40/lego_data")) / "analysis" / "features.pkl"


KINDS = Path(os.environ.get("LEGO_DATA", "/Volumes/PRO-G40/lego_data")) / "analysis" / "set_kinds.csv"
KEEP_SHAPE = {"cov", "sil", "sup", "direction", "dir_thr", "sdist"}          # --staged stage 1: the shape terms and support
MOVE_SHARES = {"resolve", "orient", "guide", "motif", "echo", "mirror", "big", "big_end", "p_sideways", "p_stack", "tile_boost",
               "snot_boost", "snot_bonus", "fid_cap", "part"}     # not energy terms: kept as they are


def mesh_bits(t: MeshTarget, shape) -> np.ndarray:
    """Each grid cell's shape: the mesh voxelised SUB times finer in grid
    units (studs, plates, studs), closed and filled solid, each cell's
    SUB^3 samples as a 64-bit mask (sampler.pack's bit order)."""
    from scipy.ndimage import binary_closing, binary_fill_holes
    m = t.mesh.copy()
    m.apply_transform(t.mesh_to_voxel)                   # mesh -> grid units (x stud, y plate, z stud)
    half = 0.5 / SUB                                     # trimesh centres voxels on multiples of the
    m.apply_translation([-half] * 3)                     # pitch; samples sit mid sub-cell
    v = m.voxelized(pitch=1.0 / SUB)
    surf = v.matrix
    solid = binary_fill_holes(binary_closing(surf, iterations=1) | surf)
    idx = np.argwhere(solid)
    c = (v.transform @ np.c_[idx, np.ones(len(idx))].T).T[:, :3] + half   # sample centres, grid units
    cell = np.floor(c).astype(int)
    sub = np.clip(np.floor((c - cell) * SUB).astype(int), 0, SUB - 1)
    H, X, Z = shape
    ok = ((cell >= 0) & (cell < np.array([X, H, Z]))).all(axis=1)
    cell, sub = cell[ok], sub[ok]
    bit = (sub[:, 1] * SUB * SUB + sub[:, 0] * SUB + sub[:, 2]).astype(np.uint64)
    bits = np.zeros(shape, dtype=np.uint64)
    np.bitwise_or.at(bits, (cell[:, 1], cell[:, 0], cell[:, 2]), np.left_shift(np.uint64(1), bit))
    return bits


def target_from_mesh(path: str, studs: int, shell: int, kind: str | None = None,
                     stand: int = 0, regions: dict[int, str] | None = None,
                     seg_label: np.ndarray | None = None,
                     seg_kind_of: dict[int, str] | None = None,
                     carve: np.ndarray | None = None,
                     smooth: int = 0, rim: int | None = None, snap_geometry: bool = False,
                     snap_pitches: str = "all") -> Target:
    if smooth > 0 or snap_geometry:
        import trimesh, tempfile, os
        m = trimesh.load(path, force="mesh", process=False)
        # gentle Taubin: lamb/nu balanced close enough that cumulative volume
        # loss is minimal; purpose is to kill 1-face noise, not reshape
        trimesh.smoothing.filter_taubin(m, iterations=smooth, lamb=0.1, nu=-0.11)
        if snap_geometry:
            from everything2lego.target_snap import snap
            m, rep = snap(m, studs=studs, pitches=snap_pitches)
            print(f"snap geometry: {rep['n_snapped']}/{rep['n_regions']} regions, {rep['snapped_area']:.0%} of the surface, "
                  f"vertices moved {rep['move_mean_studs']:.2f} studs mean, {rep['move_max_studs']:.2f} max")
        tmp = tempfile.NamedTemporaryFile(suffix=".glb", delete=False)
        m.export(tmp.name)
        tmp.close()
        try:
            # snap squares the mesh to the grid itself; aligning again would turn its snapped planes off the axes
            t = MeshTarget.from_path(tmp.name, max_studs=studs, n_samples=40000, min_island_cells=8, align=not snap_geometry)
        finally:
            os.unlink(tmp.name)
    else:
        t = MeshTarget.from_path(path, max_studs=studs, n_samples=40000, min_island_cells=8)
    occ = t.voxel_grid.occupied()
    if carve is not None:
        if carve.shape != occ.shape:
            raise ValueError(f"carve shape {carve.shape} != voxel shape {occ.shape}; "
                             f"use the same --studs for segment_photo and sampler_demo")
        occ = occ & ~carve
    solid = solid_volume(occ)
    bits = mesh_bits(t, occ.shape)
    if carve is not None:
        bits[carve] = 0
    frac = popcount(bits) / SUB ** 3
    inner = solid.copy()
    for _ in range(shell):
        inner = binary_erosion(inner, structure=np.ones((3, 3, 3), bool) & np.array(
            [[[0, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, 1, 0], [1, 1, 1], [0, 1, 0]], [[0, 0, 0], [0, 1, 0], [0, 0, 0]]], bool))
    want = (solid & ~inner) | ((frac > 0.1) & ~inner)
    # the base: layer 0 is the model's full x-z shadow, `rim` studs wider than
    # the model, so the build sits on a plate that reaches past its walls
    # (None: 0 when the mesh brings its own ground -- its bottom layer covers
    # most of the shadow -- since the rim is then shape the mesh doesn't have).
    # Vehicles don't want a solid chassis plate — they sit on wheels at the
    # corners; let the mesh decide what layer 0 contains.
    shadow = want.any(0)
    from scipy.ndimage import binary_dilation as _dil
    if rim is None:
        rim = 0 if (frac[0][shadow] >= 0.5).mean() >= 0.5 else 1    # watermill 82 %, a mesh standing on nothing ~2 %
        print(f"base rim {rim} (own ground covers {(frac[0][shadow] >= 0.5).mean():.0%} of the shadow)")
    if kind != "road_vehicle":
        base = _dil(shadow, iterations=rim) if rim else shadow   # iterations=0 would dilate to a fixed point
        want[0] |= base
        bits[0] = np.where(base, FULL, bits[0])
    if stand > 0:
        # a plate `stand` studs wider on each side than the model's shadow,
        # for a figure or small object that would otherwise sit on no plate
        pad = _dil(shadow, iterations=stand)
        want[0] |= pad
        bits[0] = np.where(pad, FULL, bits[0])
    visible = (solid & ~binary_erosion(solid)) | (want & ~solid)
    pal = t.voxel_grid.palette_idx
    from scipy.ndimage import distance_transform_edt
    known = pal >= 0
    _, (iy, ix, iz) = distance_transform_edt(~known, return_indices=True)
    colour = pal[iy, ix, iz]
    region_kind = None
    if regions:
        from everything2lego.sampler import known_kinds
        valid = known_kinds()
        bad = set(regions.values()) - valid
        if bad:
            raise ValueError(f"regions: unknown kinds {bad}; valid: {sorted(valid)}")
        region_kind = np.full(want.shape, "", dtype=object)
        for code, k in regions.items():
            region_kind[colour == int(code)] = k
    if seg_label is not None and seg_kind_of:
        from everything2lego.sampler import known_kinds
        valid = known_kinds()
        bad = set(seg_kind_of.values()) - valid
        if bad:
            raise ValueError(f"segmentation kinds not known: {bad}; valid: {sorted(valid)}")
        if seg_label.shape != want.shape:
            raise ValueError(f"segmentation shape {seg_label.shape} != target {want.shape}; "
                             f"use the same --studs for segment_mesh and sampler_demo")
        region_kind = np.full(want.shape, "", dtype=object)
        for rid, k in seg_kind_of.items():
            region_kind[seg_label == int(rid)] = k
    return Target(want=want, colour=colour, visible=visible, bits=bits, kind=kind,
                   region_kind=region_kind, region_label=seg_label if seg_label is not None else None)


def load_models() -> list[dict]:
    """The official models' features, each with what it builds (kinds.py)."""
    import pandas as pd
    rows = pickle.load(open(FEATURES, "rb"))
    if KINDS.exists():
        k = pd.read_csv(KINDS).drop_duplicates("id", keep="last")
        kind = dict(zip(k.id, k.kind))
        for r in rows:
            r["object_kind"] = kind.get(r["model"]) or kind.get(r["model"].split("_")[0])
    return [r for r in rows if not r["held_out"]]


def write_error_map(out: Path, ps, err_cell: np.ndarray, part_err) -> None:
    """error.npz (per cell and per part, by layer), and model_error.ldr / render_error.png: each part coloured by its
    local error per cell (white low -> yellow -> orange -> red -> dark red, quantiles of the build's parts)."""
    from everything2lego.colors import palette
    from everything2lego.native import ERR_LAYERS
    from dataclasses import replace
    np.savez_compressed(out / "error.npz", cell=err_cell.astype(np.float32), layers=np.array(ERR_LAYERS),
                        part=np.zeros((0, len(ERR_LAYERS))) if part_err is None else part_err.astype(np.float32))
    tot = err_cell.reshape(-1, len(ERR_LAYERS)).sum(0)
    print("local error by layer: " + "  ".join(f"{k} {v:.0f}" for k, v in zip(ERR_LAYERS, tot)), flush=True)
    if part_err is None or len(part_err) != len(ps):
        return                                                     # parts changed after sampling: no part colours
    idx = {c.code: i for i, c in enumerate(palette())}
    heat = [idx[c] for c in (15, 14, 25, 4, 320)]
    per_cell = part_err.sum(1) / np.array([max(1, p.w_world * p.d_world * p.part.height_units) for p in ps])
    edges = np.quantile(per_cell, [0.5, 0.75, 0.9, 0.97])
    hot = [replace(p, color_idx=heat[int(np.searchsorted(edges, e))]) for p, e in zip(ps, per_cell)]
    write_ldr(hot, out / "model_error.ldr")
    subprocess.run([sys.executable, str(ROOT / "tools/render_build.py"), str(out / "model_error.ldr"),
                    str(out / "render_error.png"), "--views", "30,120,210", "--size", "500"], check=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh"); ap.add_argument("out")
    ap.add_argument("--studs", type=int, default=24)
    ap.add_argument("--iters", type=int, default=0, help="sampler iterations; 0 (default) scales with target volume (~40 per occupied cell, min 20k)")
    ap.add_argument("--shell", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kind", default=None)
    ap.add_argument("--style", type=float, default=1.0, help="style weight: real-set statistics against form")
    ap.add_argument("--size-weight", type=float, default=0.0, help="weight of the part-size mix term: n_parts * KL(our size classes || real sets', sizes.py); native engine")
    ap.add_argument("--slope-courses", action="store_true", help="slopes may also be placed on a plate of their footprint (one option, two parts): rows climb one plate more each, e.g. 33 deg slopes -> ~39 deg roofs")
    ap.add_argument("--snot", action="store_true", help="small plates and tiles may also be placed on their side, held by side studs (snot_options; prior = real-set sideways share)")
    ap.add_argument("--sideways-weight", type=float, default=0.0, help="with --snot: weight of n_parts * KL(sideways share || --sideways-share)")
    ap.add_argument("--sideways-share", type=float, default=0.10, help="target share of sideways parts (real buildings ~0.15 not studs-up)")
    ap.add_argument("--roof-mix", type=float, default=0.0, help="weight of the part mix on the target's pitched roof against real sets' pitched roofs (tools/learn/roof_mix.py); 0 = off")
    ap.add_argument("--surface-mix", type=float, default=0.0, help="weight of the part mix by surface type (roof / top / wall / under; surface_mix.py) against real sets; 0 = off")
    ap.add_argument("--target-colours", type=float, default=0.0, help="parts only in the colours covering at least this share of the target's visible cells (e.g. 0.005); a part made in none of them is left out; 0 = off")
    ap.add_argument("--flat-mix", action=argparse.BooleanOptionalAction, default=False, help="with --roof-mix: the other tops (flat roofs, ledges, the ground) get real sets' part mix and patterns there too. Off: real sets leave more studs bare there (35 %% tiles) than our builds already do (60 %%), so it adds exposed studs")
    ap.add_argument("--guide", type=float, default=0.0, help="share of moves that pick their place in proportion to the local error (shape, colour, support / exposed studs, direction, atypical neighbours); 0 = uniform. The final error map is written either way (error.npz, render_error.png)")
    ap.add_argument("--resolve-share", type=float, default=0.0, help="share of moves that clear a window and cover it by the best of a bounded exact search (a slope run where the surface slants); taken from retile's share")
    ap.add_argument("--orient-share", type=float, default=0.0, help="share of moves that turn a part into the best-scoring option of its box (another turn or part); taken from retile's share")
    ap.add_argument("--layers", type=int, default=0, help="build bottom-up first: the lowest 1/N of the model's height, optimised with 1/N of --iters, then the next 1/N on top of it, and so on (20 = 5%% steps), before the whole is optimised together; exposed studs count only then; 0 = off")
    ap.add_argument("--layer-bare", type=float, default=0.0, help="with --layers: exposed-stud weight during the layer stages, as a share of --bare-top-weight (a layer's top, which the next layer covers, never counts)")
    ap.add_argument("--rebuild-regions", type=int, default=0, help="after the build, rebuild its N worst regions (error map) on their own, the build around each locked, each kept when the whole energy falls; 0 = off")
    ap.add_argument("--region-steps", type=int, default=250, help="with --rebuild-regions: steps per wanted cell of a region (at least 50 000)")
    ap.add_argument("--region-weights", default="", help='with --rebuild-regions: the regions are rebuilt and judged with these Weights fields multiplied, e.g. "direction=2,bare_top=2" (the whole build outside is locked, so this changes what counts only there); the reported energy stays the plain one')
    ap.add_argument("--region-snot", action="store_true", help="with --rebuild-regions: the regions may also use sideways parts (snot_options)")
    ap.add_argument("--whole-share", type=float, default=0.5, help="with --layers: steps for optimising the whole after the layers, as a share of --iters (the layers take --iters between them)")
    ap.add_argument("--pattern-share", type=float, default=0.0, help="share of moves that place a building pattern mined from real sets (tools/learn/motifs.py); 0 = off")
    ap.add_argument("--prefer-big", type=str, default="0", help="the filler tries bigger parts first: + W * ln(footprint, studs^2) on each candidate -- longer / wider, not taller (the energy still decides); W0:W1 falls linearly from W0 to W1 over the run")
    ap.add_argument("--direction-weight", type=float, default=0.0, help="surface direction: energy per stud-top of outward face area * (1 - cos) vs the mesh's direction (slopes on slanted roofs, tiles on flat tops)")
    ap.add_argument("--bare-top-weight", type=float, default=0.0, help="energy per top stud left exposed where nothing will sit (cell above outside the target): tiles / slopes instead of studded parts there")
    ap.add_argument("--colour-regions", type=float, default=0.0, help="read the target's colours by regions (colour_regions.py): showing cells merged by Ward's rule to this many regions per showing cell, each region in one colour (real sets' median 0.038); 0 = off")
    ap.add_argument("--collision-weight", type=float, default=1000.0, help="energy per collision of a part's geometry reaching past its cells (side stud, clip, handle) or a stud under no hole, unless they join there (watermill, 2 seeds with --connect-weight: BrickNet collisions 108 -> 3-7); 0 = off")
    ap.add_argument("--connect-weight", type=float, default=1000.0, help="energy per part with no chain of connections (stud in hole, side connectors) to the ground layer: islands resting on the build without a connection (watermill, 2 seeds: pieces 206 -> 62-69, parts falling in PyBullet 259 -> 12-37); 0 = off")
    ap.add_argument("--grip-weight", type=float, default=0.0, help="weight of n_parts * KL(share of parts joined through top and bottom to just one part || real 0.113): interlocked courses instead of columns and lone plates; 0 = off")
    ap.add_argument("--surface-distance", type=float, default=0.0, help="energy per stud-top of a part's face area per tolerance (half a stud across, a plate up) it stands past one tolerance off the target: LEGO surface sticking out (an antenna) costs by its surface, not its few sub-samples; 0 = off")
    ap.add_argument("--staged", action="store_true", help="three stages: shape alone (shape, outline, direction, support); then fewest parts with the shape held (iters / 2 more steps); then colour with the parts held")
    ap.add_argument("--mirror-share", type=float, default=0.0, help="share of the mirror move (from retile's): a part's mirror image (same part or its chiral counterpart) placed across a plane the target is mirror-symmetric about around it; 0 = off")
    ap.add_argument("--echo-share", type=float, default=0.1, help="share of the echo move (from retile's): a part copied along runs where the target repeats itself, so a regular stretch changes at once (watermill, 2 seeds: energy -1.7 %%, shape 0.851 -> 0.842 %%, base exposed 22 -> 19 %%); 0 = off")
    ap.add_argument("--phase-share", type=float, default=0.0, help="share of the phase move (from retile's): a window's plates cleared and rebuilt as bricks on the course phase (y mod 3) most brick-height parts there start at, the rest by the filler; 0 = off")
    ap.add_argument("--stack-weight", type=float, default=0.0, help="weight of n_bricks * KL(lined-up joints share || real 0.093): bricks overlap like real brickwork")
    ap.add_argument("--n-special", type=int, default=160, help="special (non plain-box) parts the build may use: the N most widely used by sets of its kind (vocabulary.json has ~1100)")
    ap.add_argument("--colour-max", type=float, default=12.0, help="Lab distance past which a part's colour error on a showing cell costs 10x more: a part not made near the area's colour is all but kept out (watermill, 2 seeds: colour error 4.9 -> 1.5 at 12, 2.8 at 20, shape +1 %% / 0; 0 = off)")
    ap.add_argument("--shape-weight", type=float, default=16.0, help="energy per cell wholly wrong in shape (Weights.cov is 4: the shape term is then ~3 %% of the energy and statistics outweigh any sub-cell fit; 16 fits closer and still looks clean, 32+ adds busy detail)")
    ap.add_argument("--surface-fit", type=float, default=0.0, help="weight the shape term by distance to the mesh's surface (surface_fit.py; value = alpha, cost per stud of overhang; 0 = plain sample count)")
    ap.add_argument("--whole-cells", action="store_true", help="match the mesh at whole-cell detail (every target cell a full cube) instead of 4x4x4 sub-cells: round 1x1 plates no longer win by tracing tiny curves")
    ap.add_argument("--sup-weight", type=float, default=0.0, help="energy per unsupported part (0 = keep the weights' own: 40 default, 100 learned); raise it when --size-weight would trade support for bigger parts")
    ap.add_argument("--learned-weights", action="store_true", help="energy weights fitted to real sets (Weights.learned(kind)) instead of the hand-set defaults")
    ap.add_argument("--colour-smooth", type=float, default=0.0, help="Potts-MRF colour smoothing of the target (colour_mrf.smooth beta, Lab units; ~40-80): fewer colour specks, so same-colour parts can merge")
    ap.add_argument("--base-rim", type=int, default=None, help="studs the base plate (layer 0) reaches past the model's shadow; default: 0 when the mesh's bottom layer is its own ground, else 1")
    ap.add_argument("--stand", type=int, default=0, help="pad base plate by N studs (for figures or small objects)")
    ap.add_argument("--regions", default=None, help='JSON file mapping palette codes to kinds, e.g. {"0":"road_vehicle","4":"object"} — gives different sampler weights per colour region')
    ap.add_argument("--segmentation", default=None, help="npz from tools/segment_mesh.py: per-voxel region_kind (overrides --regions)")
    ap.add_argument("--prune-unsupported", action="store_true", help="drop parts with no vertical supporter below from the final build (post-process)")
    ap.add_argument("--drop-small-clusters", type=int, default=0, help="post-process: drop connected clusters smaller than N parts (1-2 safe; larger values will drop SNOT-held structure since side connectors aren't considered as adjacency)")
    ap.add_argument("--apply-templates", default=None, help="path to a regions.npz (from classify_regions.py) — post-process: swap each part for a same-footprint one in a region-preferred shape family (roof->slope, wall->brick, ...)")
    ap.add_argument("--depth-carve", default=None, help="npz from tools/depth_carve.py: carve mask OR-ed into --segmentation's carve")
    ap.add_argument("--tile-boost", type=float, default=0.0, help="log-bonus on every tile option at proposal time; real sets use 15-23%% tiles vs our 11%%, +1 to +2 is a reasonable range")
    ap.add_argument("--snot-boost", type=float, default=0.0, help="log-bonus on every SNOT option (brickS/plateS/tileS) at proposal time; real sets use 17-32%% SNOT vs our ~0%%")
    ap.add_argument("--engine", default="auto", choices=["auto", "python", "native"], help="sampler engine (python = SNOT templates + hinges via snot_insert / hinge_swap; native = Rust fast path, no SNOT yet)")
    ap.add_argument("--snot-decorate", action=argparse.BooleanOptionalAction, default=False, help="after sampling, add SNOT children to anchors (snot.decorate); engine-agnostic.  OFF by default: current decorate() picks anchors blindly and the output looks like a brick explosion for the watermill mesh.  Needs an outward-facing / exterior-only filter before re-enabling.")
    ap.add_argument("--snot-rate", type=float, default=0.8, help="probability per eligible BAND to decorate; bands cap at 5 per key so this gates how many distinct bands fire, not raw count")
    ap.add_argument("--realism-best-of", type=int, default=1, help="run N seeds (seed, seed+1, ...), score each with tools/learn/realism and keep the highest-scoring build; 1 disables.  Needs $LEGO_DATA/analysis/realism/model.pt.")
    ap.add_argument("--cnn-prior", default=None, help="path to a per-option bias .npy from tools/cnn_prior.py (whole-target CNN score, added to each option's logprior)")
    ap.add_argument("--init-from-cnn", default=None, help="path to a cnn_decoder.pt checkpoint: run its greedy decode, then feed those placements as init to the sampler (composition: fast CNN seeds the Metropolis walk)")
    ap.add_argument("--init-prefer-length", type=float, default=1.0, help="--prefer-length passed to the CNN greedy seed; 1.0 is the sched-sampling champion")
    ap.add_argument("--init-support-penalty", type=float, default=0.0, help="--support-penalty passed to the CNN greedy seed; penalises unsupported bottom cells at inference")
    ap.add_argument("--init-guidance-model", default=None, help="path to an OMR-only CNN checkpoint for CFG when the --init-from-cnn is a hybrid/distilled model")
    ap.add_argument("--init-guidance", type=float, default=1.0, help="CFG scale for the CNN seed (1.0 = pure conditional, >1 amplifies, <1 blends toward unconditional)")
    ap.add_argument("--subassemblies", type=int, default=0, help="group placements into N-part submodels in an MPD .ldr (real sets have 11-26 sub-assemblies; 25 is a reasonable target)")
    ap.add_argument("--snap-geometry", action="store_true", help="turn the mesh's stud-scale planes to pitches LEGO slopes build, headings to the grid axes, the rest deformed smoothly, before voxelising (target_snap.py)")
    ap.add_argument("--snap-pitches", default="all", choices=["all", "roof"], help="pitch set for --snap-geometry: all slope row pitches, or roof = without 39.6 (studless cheese caps only; target_snap.PRESETS)")
    ap.add_argument("--semantic", default=None, help="npz from tools/semantic_labels.py (same --studs): per-cell part category (window, door, wheel, fence, plant)")
    ap.add_argument("--semantic-weight", type=float, default=0.0, help="with --semantic: energy bonus per cell a part fills whose label is its Rebrickable category (and the filler proposes such parts there); 0 = off")
    ap.add_argument("--smooth", type=int, default=0, help="apply N iterations of Taubin smoothing to the mesh before voxelising (reduces 1-stud surface noise; 5-10 is a reasonable range)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    regions = None
    if a.regions:
        regions = {int(k): v for k, v in json.loads(Path(a.regions).read_text()).items()}
    seg_label, seg_kind_of, carve = None, None, None
    if a.segmentation:
        d = np.load(a.segmentation, allow_pickle=True)
        seg_label = d["label"]
        seg_kind_of = {int(k): v for k, v in d["kind_of"]}
        if "carve" in d.files:
            carve = d["carve"]
    if a.depth_carve:
        d = np.load(a.depth_carve, allow_pickle=True)
        extra = d["carve"]
        carve = extra if carve is None else (carve | extra)
    target = target_from_mesh(a.mesh, a.studs, a.shell, a.kind, stand=a.stand, rim=a.base_rim, regions=regions,
                              seg_label=seg_label, seg_kind_of=seg_kind_of, carve=carve,
                              smooth=a.smooth, snap_geometry=a.snap_geometry, snap_pitches=a.snap_pitches)
    if a.semantic:
        d = np.load(a.semantic, allow_pickle=True)
        if d["cell_cat"].shape != target.want.shape:
            raise SystemExit(f"--semantic shape {d['cell_cat'].shape} != target {target.want.shape}: use the same --studs")
        target.cell_cat, target.categories = d["cell_cat"].astype(np.int8), tuple(str(c) for c in d["categories"])
    if a.whole_cells:
        target.bits = np.where(target.want, FULL, np.uint64(0)).astype(np.uint64); target.fill = target.want.astype(float)
    if a.colour_smooth > 0:
        from everything2lego.colour_mrf import smooth
        target.colour = smooth(target.colour, target.want, beta=a.colour_smooth)
    if a.colour_regions > 0:
        from everything2lego.colour_regions import regions
        n0 = len(np.unique(target.colour[target.want & target.visible]))
        target.colour = regions(target.colour, target.want, target.visible, density=a.colour_regions)
        print(f"colour regions: {a.colour_regions * 100:.1f} per 100 showing cells; colours shown {n0} -> "
              f"{len(np.unique(target.colour[target.want & target.visible]))}")
    models = load_models()
    t0 = time.time()
    from everything2lego.sampler import Weights
    iters = a.iters if a.iters > 0 else max(20000, 40 * int(target.want.sum()))
    opt_bias = np.load(a.cnn_prior) if a.cnn_prior else None
    init_placements = None
    if a.init_from_cnn:
        sys.path.insert(0, str(ROOT / "tools"))
        from cnn_build import cnn_greedy
        print(f"init-from-cnn: running greedy over {Path(a.init_from_cnn).name}...", flush=True)
        init_placements = cnn_greedy(target, Path(a.init_from_cnn), prefer_length=a.init_prefer_length,
                                       support_penalty=a.init_support_penalty,
                                       guidance_model=Path(a.init_guidance_model) if a.init_guidance_model else None,
                                       guidance=a.init_guidance)
        print(f"init: {len(init_placements)} placements from CNN", flush=True)
    def _once(seed: int):
        _t = time.time()
        w = Weights.learned(target.kind) if a.learned_weights else Weights()
        w.style = a.style; w.size = a.size_weight; w.sideways = a.sideways_weight; w.p_sideways = a.sideways_share; w.surface = a.surface_fit; w.big, w.big_end = (float(a.prefer_big.split(':')[0]), float(a.prefer_big.split(':')[1]) if ':' in a.prefer_big else -1.0); w.motif = a.pattern_share; w.surface_mix = a.surface_mix; w.roof_mix = a.roof_mix; w.guide = a.guide; w.resolve = a.resolve_share; w.orient = a.orient_share; w.flat_mix = a.flat_mix; w.stack = a.stack_weight; w.bare_top = a.bare_top_weight; w.direction = a.direction_weight
        if a.sup_weight > 0: w.sup = a.sup_weight
        if a.shape_weight is not None: w.cov = a.shape_weight
        w.col_max = a.colour_max
        w.echo = a.echo_share
        w.mirror = a.mirror_share
        w.phase = a.phase_share
        w.sdist = a.surface_distance
        w.conn = a.connect_weight
        w.coll = a.collision_weight
        w.grip = a.grip_weight
        w.sem = a.semantic_weight
        def build(w):
            init, left = init_placements, iters
            if a.layers > 1:
                # bottom-up: each stage builds the target up to its cut, from the stages below; the mesh's shape stays
                # whole (only which cells are built is cut), so the cut is no surface and needs no direction; its studs
                # are covered by the next stage, so exposed studs are not counted until the whole is optimised
                import copy
                from everything2lego.region_rebuild import _cells
                ys = np.nonzero(target.want.any((1, 2)))[0]
                lo, hi = int(ys.min()), int(ys.max()) + 1
                per = iters // a.layers                      # a layer of 1/N of the height gets 1/N of the steps
                wl = copy.copy(w); wl.bare_top = w.bare_top * a.layer_bare
                for k in range(1, a.layers + 1):
                    tk = copy.copy(target)
                    tk.want = target.want & (np.arange(target.want.shape[0])[:, None, None] < lo + -(-(hi - lo) * k // a.layers))
                    if init:                                 # a part reaching past the cut starts the next stage over
                        init = [p for p in init if tk.want[tuple(_cells(p).T)].all()]
                    init, li = sample(tk, models, iters=per, seed=seed + 1000 * k, weights=wl, tile_boost=a.tile_boost,
                                      snot_boost=a.snot_boost, opt_bias=opt_bias, init_placements=init, engine=a.engine,
                                      snot=a.snot, n_special=a.n_special, courses=a.slope_courses, colour_share=a.target_colours)
                    print(f"layer {k}/{a.layers}: {int(tk.want.sum())} cells, {li['parts']} parts, energy {li['energy']:.0f}", flush=True)
                init = [p for p in init if target.want[tuple(_cells(p).T)].all()]
                left = int(iters * a.whole_share)
            _ps, _info = sample(target, models, iters=left, seed=seed, weights=w,
                                 tile_boost=a.tile_boost, snot_boost=a.snot_boost, opt_bias=opt_bias,
                                 init_placements=init, engine=a.engine, snot=a.snot, n_special=a.n_special, courses=a.slope_courses,
                                 colour_share=a.target_colours)
            return _ps, _info
        if a.staged:
            # three stages, each optimised with the earlier ones held: (1) shape alone -- shape, outline, surface
            # direction and support, nothing else; (2) fewest parts, the shape term not let past stage 1's and the shape
            # terms kept above a part's worth so they only break ties; (3) colour, the parts as they are
            import copy
            from dataclasses import fields
            zero = [f.name for f in fields(Weights) if f.type in ("float", float) and f.name not in KEEP_SHAPE]
            w1 = copy.copy(w)
            for name in zero:
                if name not in MOVE_SHARES: setattr(w1, name, 0.0)
            w1.flat_mix = False
            _ps, _info = build(w1)
            print(f"stage 1 (shape): shape term {_info['fidelity']:.1f}, {_info['parts']} parts", flush=True)
            w2 = copy.copy(w1); w2.fid_cap = _info["fidelity"] * (1 + 1e-9); w2.part = 1.0
            _ps, _info = sample(target, models, iters=iters // 2, seed=seed + 7, weights=w2, tile_boost=a.tile_boost,
                                snot_boost=a.snot_boost, opt_bias=opt_bias, init_placements=list(_ps), engine=a.engine,
                                snot=a.snot, n_special=a.n_special, courses=a.slope_courses, colour_share=a.target_colours)
            print(f"stage 2 (parts): shape term {_info['fidelity']:.1f}, {_info['parts']} parts", flush=True)
            w3 = copy.copy(w2); w3.col, w3.col_max, w3.recolour = w.col, w.col_max, True
            _ps, _info = sample(target, models, iters=0, seed=seed + 8, weights=w3, tile_boost=a.tile_boost,
                                snot_boost=a.snot_boost, opt_bias=opt_bias, init_placements=list(_ps), engine=a.engine,
                                snot=a.snot, n_special=a.n_special, courses=a.slope_courses, colour_share=a.target_colours)
            print(f"stage 3 (colour): shape term {_info['fidelity']:.1f}, {_info['parts']} parts", flush=True)
        else:
            _ps, _info = build(w)
        if a.rebuild_regions > 0:
            from everything2lego.region_rebuild import rebuild_worst, evaluate
            kw = dict(weights=w, tile_boost=a.tile_boost, snot_boost=a.snot_boost, engine=a.engine, snot=a.snot or a.region_snot,
                      n_special=a.n_special, courses=a.slope_courses, colour_share=a.target_colours)
            import copy
            wr = copy.copy(w)
            for f in filter(None, a.region_weights.split(",")):
                name, m = f.split("=")
                setattr(wr, name, getattr(wr, name) * float(m))
            _ps, rep = rebuild_worst(target, models, list(_ps), _info["_err_cell"], sample, k=a.rebuild_regions,
                                     per_cell=a.region_steps, seed=seed, **{**kw, "weights": wr})
            _info.update({k: v for k, v in evaluate(target, models, _ps, sample, **kw).items()
                          if k in ("energy", "unsupported", "bare_side_connectors", "fidelity", "_err_cell")})
            _info.pop("_part_error", None)                 # the evaluation lists the parts in its own order
            _info["regions"] = rep
        _info["iters"] = iters
        _info["seconds"] = round(time.time() - _t, 1)
        _info["seed"] = seed
        if a.prune_unsupported:
            from everything2lego.supports import prune_unsupported
            _ps, pruned = prune_unsupported(list(_ps))
            _info["pruned_unsupported"] = pruned
        if a.drop_small_clusters > 0:
            from everything2lego.supports import drop_isolated_clusters
            _ps, dropped = drop_isolated_clusters(list(_ps), min_cluster=a.drop_small_clusters)
            _info["dropped_clusters"] = dropped
        if a.apply_templates:
            d = np.load(a.apply_templates, allow_pickle=True)
            from everything2lego.region_templates import apply as apply_region_templates
            _ps, swaps = apply_region_templates(list(_ps), d["label"])
            _info["template_swaps"] = swaps
        _info["parts"] = len(_ps)
        _deco = _info.pop("_snot_decorations", None) or {}
        if a.snot_decorate:
            from everything2lego import snot as _snot
            extra = _snot.decorate(_ps, seed=seed, rate=a.snot_rate)
            for p, kids in extra.items():
                if p not in _deco:
                    _deco[p] = list(kids)
            _info["snot_decorations"] = sum(len(v) for v in _deco.values())
        return _ps, _info, _deco

    if a.realism_best_of > 1:
        from tempfile import TemporaryDirectory
        from everything2lego.io_ldraw import write_ldr as _wld
        sys.path.insert(0, str(ROOT / "tools"))
        from learn import realism as _real
        best_score = -1.0
        best = None
        for k in range(a.realism_best_of):
            seed = a.seed + k
            print(f"realism-best-of {k+1}/{a.realism_best_of}: seed={seed}", flush=True)
            cps, cinfo, cdeco = _once(seed)
            with TemporaryDirectory() as td:
                tmp = Path(td) / "model.ldr"
                _wld(cps, tmp, decorations=cdeco or None)
                try:
                    feat = _real.extract(tmp)
                except Exception:
                    feat = None
            score = 0.0
            if feat is not None:
                import torch, torch.nn as nn, torch.nn.functional as F
                ckpt = torch.load(_real.MODEL_PT, map_location="cpu", weights_only=True)
                net = nn.Sequential(nn.Linear(ckpt["feature_len"], 64), nn.ReLU(),
                                     nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))
                net.load_state_dict(ckpt["state_dict"]); net.eval()
                with torch.no_grad():
                    logits = net(torch.tensor(feat).unsqueeze(0))
                    score = float(F.softmax(logits, dim=-1)[0, 1].item())
            cinfo["realism"] = score
            print(f"  realism={score:.3f}  parts={cinfo['parts']}", flush=True)
            if score > best_score:
                best_score = score
                best = (cps, cinfo, cdeco)
        ps, info, deco = best
        info["realism_best_of"] = a.realism_best_of
    else:
        ps, info, deco = _once(a.seed)
    subs = None
    if a.subassemblies > 0:
        from everything2lego.subassemblies import split_by_y_bands
        subs = split_by_y_bands(list(ps), target_size=a.subassemblies)
        info["subassemblies"] = [len(g) for g in subs]
    write_ldr(ps, out / "model.ldr", subassemblies=subs, decorations=deco or None)
    err_cell, part_err = info.pop("_err_cell", None), info.pop("_part_error", None)
    if err_cell is not None:
        write_error_map(out, ps, err_cell, part_err)
    (out / "info.json").write_text(json.dumps({k: v for k, v in info.items()}, indent=1, default=float))
    print(json.dumps({k: v for k, v in info.items() if k != "history"}, default=float))
    subprocess.run([sys.executable, str(ROOT / "tools/render_build.py"), str(out / "model.ldr"),
                    str(out / "render.png"), "--views", "30,120,210", "--size", "500"], check=False)


if __name__ == "__main__":
    main()
