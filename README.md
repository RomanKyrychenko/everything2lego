# everything2lego

Convert a 3D mesh, a set of turntable images, or a single photo into a
buildable LEGO model — a `.ldr` file, a parts list (`wanted.xml`), an offline
3D viewer, illustrated PDF instructions, and a preview render.

The pipeline goes **photo → 3D → LEGO**, so any of the three stages can be
skipped:

```
photo  ──►  Hunyuan3D  ──►  colored mesh  ──►  semantic pipeline  ──►  LEGO
       (or TripoSR / DUSt3R / silhouette carving)                       ├─ .ldr
                                                                        ├─ wanted.xml
                                                                        ├─ viewer/
                                                                        ├─ instructions.pdf
                                                                        └─ preview.png
```

<p align="center">
<img src="examples/gallery/watermill_38k_parts.png" width="45%"/>
<img src="examples/gallery/watermill_12k_parts.png" width="45%"/>
</p>

<p align="center"><em>Left: watermill at 80 studs (38K parts, very detailed).
Right: same watermill at 56 studs with hollow interior (12K parts, still
recognizable).  Both were reconstructed from one photo via Hunyuan3D-2mini,
then passed through the semantic + soft-train pipeline.</em></p>

---

## Install

Requires Python 3.9+.  On Apple Silicon the neural backends need an
`x86_64` venv with Rosetta 2 installed (`softwareupdate --install-rosetta`)
because PyTorch's Hunyuan/TripoSR builds are still Intel-only for some ops.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
```

Optional heavy deps for the photo → 3D backends:

```bash
.venv/bin/python -m pip install torch torchvision diffusers transformers \
    einops omegaconf timm scikit-image opencv-python-headless onnxruntime \
    scikit-learn pymeshlab pyyaml
```

External model checkpoints (~1.5 GB each) live on a directory you point at
with environment variables — put them on an external disk if internal SSD
is tight:

```bash
export HF_HOME=/Volumes/PRO-G40/huggingface
export TORCH_HOME=/Volumes/PRO-G40/torch
export HY3DGEN_MODELS=/Volumes/PRO-G40/hy3dgen
```

---

## Quick start

### Pure LEGO pipeline (mesh you already have)

```bash
.venv/bin/everything2lego semantic examples/sample.glb \
    --out out/ --max-studs 56 --soft-train
```

### Photo → 3D → LEGO

```bash
# 1) Photo to mesh (Hunyuan3D-2mini, best on 16 GB Apple Silicon).
HUNYUAN3D_REPO=tencent/Hunyuan3D-2mini \
HUNYUAN3D_SUBFOLDER=hunyuan3d-dit-v2-mini \
PYTORCH_ENABLE_MPS_FALLBACK=1 \
.venv/bin/everything2lego hunyuan photo.jpg \
    --out out_mesh/ --resolution 384 --steps 50 --guidance 7.5

# 2) Preprocess the mesh so the LEGO pipeline packs cleanly:
.venv/bin/python tools/bake_texture_to_vertex.py     out_mesh/model.glb out_mesh/baked.glb
.venv/bin/python tools/decimate_keep_color.py        out_mesh/baked.glb out_mesh/dec.glb 60000
.venv/bin/python tools/quantise_colors.py            out_mesh/dec.glb   out_mesh/q.glb 8

# 3) LEGO pipeline with all quality improvements enabled.
.venv/bin/everything2lego semantic out_mesh/q.glb \
    --out out_lego/ --max-studs 56 \
    --subvoxels 2 --hollow \
    --soft-train --soft-iters 500 --w-color 3.0
```

### Silhouette hull from turntable images (no neural net)

```bash
.venv/bin/everything2lego carve examples/views --max-studs 24 --out out/
```

---

## The semantic pipeline

`semantic` runs a five-stage pipeline (see `ARCHITECTURE_V2.md`):

1. **Voxelise** the mesh at `max_studs` resolution.  `--subvoxels 2` super-
   samples occupancy for cleaner thin walls and window frames.
2. **Region detector** classifies each mesh face into a region — planar
   walls (from `trimesh.facets`), planar roofs (via normal clustering of
   leftover marching-cubes triangles), foundations, curved primitives, or
   generic detail.
3. **Per-region MILP packing** — each region calls a specialised builder
   (`PlanarWallBuilder`, `PlanarRoofBuilder`, …) which uses HiGHS to solve
   an exact-cover problem over the region's cells, picking the biggest LEGO
   plate/tile that fits.
4. **Optional soft training (`--soft-train`)** — 300 steps of Adam (set
   `--soft-iters`) over a Gumbel-Softmax-relaxed cell grid.  Needs PyTorch.  The trainer *initialises* from the
   MILP placements (idea #1), keeps per-region palette locks (idea #2),
   and includes a support term that penalises floating cells (idea #6).
   Loss = `2·shape + 1·color + 0.05·type_prior + 0.02·coherence
   + 0.05·support`.
5. **Discretise** the soft grid via a brick pre-pass (5-plate `brick_tall`
   → 3-plate `brick` → 1-plate) with vertical + horizontal colour majority
   smoothing so bigger footprints actually get consolidated.  Slopes emit
   as 2×1 slopes (part 3040) with 1×1 brick fillers for orphans; cheese
   slopes (54200) at pitched-roof tips.

### CLI flags that matter

| Flag | What it does |
|---|---|
| `--max-studs 40..80` | LEGO grid resolution. Halving pitch drops parts ~5× and detail ~2×. |
| `--subvoxels 2` | Super-sample occupancy for sharper thin features. |
| `--soft-train` | Enable the differentiable trainer; refines colour + fills gaps. |
| `--hollow` | Carve out the interior, leaving shell + floors + columns + foundation. |
| `--shell-thickness 2` | Cells kept just below the surface skin. |
| `--floor-spacing 12` | One structural floor slab every N plates. |
| `--column-spacing 8` | Column node every N × N (x, z) grid positions. |
| `--w-color 3.0` | Colour-fidelity weight. Raise when colours drift. |
| `--photos path.jpg` | Compare renders of the soft model with the source photo(s): first photo is the front view, second the side, third the top. Weighted by `--w-photo-pixel` (default 1) and `--w-photo-percep` (default 0, VGG). |
| `--refine` / `--grad-refine` | Simulated-annealing and gradient refinement of the MILP placements (`--refine-iters`, `--grad-iters`). Soft training starts from their result. |
| `--milp-time 1.0` | Seconds HiGHS may spend per region. |
| `--w-shape-surface`, `--w-shape-volume`, `--w-color`, `--w-diversity`, `--w-coherence`, `--w-count` | Soft-training loss weights (defaults 1, 2, 1, 0, 0.02, 0). |
| `--n-samples` | Surface samples taken from the mesh for region detection. |

### `build` styling flags

`build` (and `carve`, `triposr`, `hunyuan`...) packs voxels with the classical
packer, then runs optional post-pack passes.  `report.json` describes the
model after all of them.

| Flag | What it does |
|---|---|
| `--brace-mode auto\|floors\|columns\|none` | How floating parts get support: internal floors for hollow shells, pegs and columns otherwise. |
| `--hollow --wall-thickness 2` | Carve the interior before packing. |
| `--texture log\|masonry\|curved\|smooth` | Swap bricks for textured equivalents. |
| `--windows-color 71 --doors-color 70 --panels-color 71` | Swap stacks of bricks in those LDraw colours for window frames, doors and panels. |
| `--detect-features` | Find window- and door-shaped colour clusters and swap in real parts. |
| `--variety`, `--architectural-style` | Decorative part variation; foundation, cornice, ridge and quoin details. |
| `--color-k`, `--color-mode`, `--subvoxels` | Colour quantisation and occupancy super-sampling. |

Every command exits with status 1 when the model it wrote is not one
supported assembly (see `report.json`), `--no-supports` included.

### Other commands

* `optimize mesh.glb` runs the simulated-annealing optimiser on its own
  (`--iters`, `--t-start`, `--t-end`, the same `--w-*` weights).
* `demo` builds the bundled parametric templates (gable cabin, modernist
  box) for smoke tests.
* All commands write to `out/` unless `--out` says otherwise.

### Reducing part count without breaking the model

The watermill demo (`examples/gallery/`) went from 38K parts down to 5.7K
by stacking:

| Step | Parts | Loss |
|---|---|---|
| 80 studs, solid, 10-colour palette | 38,196 | 0.005 |
| `--hollow` (floors 12, columns 8) | 26,724 | 0.008 |
| 56 studs + tighter hollow | 12,598 | 0.009 |
| 40 studs + 5-colour quantisation + aggressive hollow | 5,690 | 0.009 |

Stud resolution is the biggest lever.  Then colour cluster count — fewer
clusters mean longer same-colour runs, so MILP finds bigger bricks.
Aggressive hollowing (fewer floors / columns) is the last knob — watch the
`connections` report for structural warnings if you push it hard.

---

## Backends

| Command | Input | Backend | Notes |
|---|---|---|---|
| `build` | one textured mesh | none | classical voxel packer |
| `carve` | 2+ turntable images | silhouette hull | CPU-only, deterministic |
| `triposr` | 1 photo | Stability AI TripoSR | MPS/CUDA |
| `hybrid` | 2+ photos | TripoSR ∩ silhouettes | best classical / neural mix |
| `hunyuan` | 1 photo | Hunyuan3D-2 base / mini / mini-turbo | MPS/CUDA |
| `hunyuan-mv` | 2-4 canonical views | Hunyuan3D-2mv | MPS/CUDA |
| `hunyuan21` | 1 photo | Hunyuan3D-2.1 (bigger MoE) | needs 24+ GB VRAM |
| `photos` | 4+ unstructured photos | Naver DUSt3R | real-world subjects |
| `semantic` | any mesh | region + MILP + soft-train | **the recommended path** |

---

## Layout

```
src/everything2lego/
    bricks.py                Brick catalog (Rebrickable-derived).
    colors.py                LDraw palette + LAB nearest-neighbour matching.
    voxelize.py              Mesh → colored voxel grid.
    pack.py                  Classical greedy packer.
    pipeline.py              build/carve/… glue.
    cli.py                   Typer CLI.
    stability.py             Stud-connectivity + support checker.
    io_ldraw.py              .ldr writer.
    io_bricklink.py          wanted.xml writer.
    io_stl.py                Solid STL writer for the LEGO output.
    instructions.py          Step-by-step PDF renderer.
    viewer_export.py         Offline WebGL viewer.
    backends/                triposr, hybrid, dust3r, hunyuan3d, hunyuan3d_mv,
                             hunyuan3d_v21.
    semantic_v2/             Region detector, per-region MILP builders,
                             SoftAssembly + SoftLoss + SoftTrainer, and the
                             interior filler (with hollow support).
    templates/               Parametric template models (gable-cabin,
                             modernist-box) for tests.

tools/
    bake_texture_to_vertex.py   Bake a textured GLB into per-vertex RGB.
    decimate_keep_color.py      Quadric-decimate with KDTree colour transfer.
    quantise_colors.py          K-means colour clustering of vertex_colors.
    remove_bg.py                U^2-Net (ONNX) foreground extraction.
    delight.py                  Hunyuan3D-delight-v2-0 lighting removal.

examples/
    sample.glb                 A trivial mesh for `build` smoke tests.
    views/                     Turntable images for `carve`.
    gallery/                   Preview renders from the semantic pipeline.
    house/                     A cabin photo set used in the writeup.

tests/                          Pytest suite; regression tests for the packer,
                                connectivity checker, and colour matching.
```

---

## Testing

```bash
.venv/bin/python -m pytest tests/ -x
```

---

## Licences

MIT for our own code.  Third-party components (Hunyuan3D-2 weights, TripoSR,
DUSt3R, U^2-Net, LDraw catalogue, Rebrickable database) keep their upstream
licenses — see `LICENSE` for the list.  Hunyuan3D-2 is CC-BY-NC; commercial
use of its weights needs a separate agreement with Tencent.
