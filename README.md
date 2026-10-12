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

That installs the core mesh → LEGO pipeline plus `pytest`.  Nothing in it
needs a GPU or PyTorch, so it works on a plain CI runner; the tests that
touch the neural backends skip themselves when torch is absent.

Extras (combine as `'.[dev,neural]'`):

| extra    | what it adds                                                       |
|----------|--------------------------------------------------------------------|
| `dev`    | `pytest` — enough to run the whole suite                            |
| `neural` | torch, torchvision, diffusers, transformers, einops, omegaconf, timm, scikit-image, opencv-python-headless, onnxruntime, scikit-learn, pymeshlab, pyyaml — the photo → 3D backends (Hunyuan3D, TripoSR, DUSt3R, U^2-Net) |

```bash
.venv/bin/python -m pip install -e '.[dev,neural]'
```

The interactive viewer's JavaScript bundle (`src/everything2lego/viewer/bundle.js`)
is committed, so Node is only needed when you change `main.js`:

```bash
npm ci                    # three.js, lucide, esbuild, Playwright
npm run build:viewer      # regenerate bundle.js (commit the result)
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

### Part labels (optional)

The region detector finds flat facets and guesses what they are from their
normals.  A part segmenter knows that a patch is a window and another a
chimney, and so does a person painting a mesh in Blender.  Either hands the
pipeline one colour per part:

```bash
# A colour-coded mesh next to the input (any segmenter, or hand-painted).
.venv/bin/everything2lego semantic house.glb --parts house.parts.glb --out out/

# No GPU: a built-in CPU segmentation by colour, normal and connectivity.
.venv/bin/everything2lego semantic house.glb --segmenter builtin --out out/
```

A JSON sidecar next to the labelled mesh (`house.parts.json`) may name the
colours: `{"#ff0000": "roof", "#0000ff": "window", "#ffff00": "door"}`.
Named windows and doors become frames and doors the walls are packed around;
unnamed parts are classified by normal and height.  Labels made on a
decimated copy of the mesh are carried back by nearest face.

[SegviGen](https://github.com/Nelipot-Lee/SegviGen) (Li et al., SIGGRAPH
2026, MIT) is the segmenter this was built for.  It needs Linux, an NVIDIA
GPU with 24 GB of memory and the TRELLIS.2 environment; `tools/segvigen.py`
drives it, caches results by mesh hash, decimates the mesh first (the model
works on a fixed voxel grid, so extra triangles only cost time) and passes a
painted 2D map through for its guided mode:

```bash
python tools/segvigen.py house.glb --segvigen ~/SegviGen --faces 50000
.venv/bin/everything2lego semantic house.glb --parts house.parts.glb
```

Without that hardware, `--segmenter builtin` and hand-painted labels give the
same pipeline its parts.  None of this is required: without `--parts` or
`--segmenter` the facet detector runs as before.

### Naming the parts with a vision model (optional)

A segmenter says where the parts are, not what they are, and the builders
only treat a part as a window, door, roof, wall or foundation when it is
named one.  Geometry can guess at a house; it cannot at a car or a chair.
`--label-parts` draws the object with each part highlighted (a PIL
rasteriser, no GPU), shows the pictures to Claude and asks what every part
is and how a LEGO builder should treat it:

```bash
pip install 'everything2lego[vlm]'        # the anthropic SDK
export ANTHROPIC_API_KEY=...

# Name the parts while building (uses --parts, or the builtin segmenter).
.venv/bin/everything2lego semantic truck.glb --label-parts --hint "a 1960s pickup" --out out/

# Or name them once, keep the result, and build from it any number of times.
.venv/bin/everything2lego label-parts truck.glb -o truck.parts.glb
.venv/bin/everything2lego semantic truck.glb --parts truck.parts.glb --out out/
```

The command prints what the model saw ("object: a pickup truck", "part 4:
windscreen -> window", "part 7: left front wheel") and keeps the renders
under `out/parts/`.  Names already in a sidecar win over the model's;
answers are cached per mesh and segmentation, so a second run is free.
One mesh costs a few cents (two overview pictures and one per part, at
most sixteen parts per request).  Requests opt into the API's default
refusal fallback, so a declined picture is retried on another model
inside the same call.  Without the package, a key or a network the
pipeline warns and carries on with unnamed parts.

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
| `--color-k`, `--color-mode`, `--subvoxels` | Colour sampling and occupancy super-sampling. |
| `--max-parts N` | Target part count; picks resolution, hollowing and colour count for you (see below). |
| `--palette house\|auto\|full` | Colour handling before packing: `house` snaps to building colours, `auto` (default) clusters to `--color-clusters` (10) LDraw colours, `full` keeps every colour. |
| `--min-color-region 4` | Same-colour voxel regions smaller than this take their neighbours' majority colour, so speckles vanish. 0 disables. |

Every command exits with status 1 when the model it wrote is not one
supported assembly (see `report.json`), `--no-supports` included.

### Photo colour handling

The photo commands (`photos`, `hunyuan`, `hunyuan21`, `hunyuan-mv`,
`triposr`, `hybrid`) run `tools/delight.py` on their input photos first
when `torch` and `diffusers` are importable, so baked-in shadows and
highlights do not become colour changes in the model; without those
packages they print one line and continue.  `--no-delight` turns it off.

`--palette house` is the right choice for buildings: every colour snaps
(nearest CIE Lab) to white, tan, light / dark bluish grey, reddish brown,
red, dark red or black; glass snaps to trans-light-blue, or to light bluish
grey (71) when the transparent colour is not in the shipped palette.
`--palette auto` keeps the k-means-style clustering (`--color-clusters`,
default 10); `--palette full` disables quantisation.  Either way, the
per-region speckle filter (`--min-color-region`) runs afterwards.

### Part budget: `--max-parts`

```bash
.venv/bin/everything2lego carve examples/views --max-parts 1500 --out out/
```

`--max-parts N` replaces guessing at `--max-studs`.  The command packs the
model once at 16 studs (a few seconds), then predicts the part count at
every other setting with a power law calibrated on the table below and on
the carve example (parts grow like resolution^2.3 for hollow shells and
^2.5 for solids; hollowing saves ~30% at 80 studs; halving the colour count
saves ~10%).  It then picks, in order, the largest resolution, then
hollowing, then the smallest colour-count reduction that fits, and prints
the choice.  When `--max-studs` is also given, `--max-parts` wins and says
so.  Photo backends cannot be re-run cheaply, so there the budget only
lowers the resolution below the one the backend produced; `semantic`
estimates with the classical packer and leaves colours alone.

On the carve example the prediction from the 16-stud measurement (632
parts) is 1,742 at 24 studs (actual 1,601) and 3,575 at 32 studs (actual
3,550); `--max-parts 1500` chooses 20 studs and builds 1,255 parts.

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
`--max-parts N` (see above) walks these levers for you.

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

Python suite (runs without torch; the three DUSt3R tests report as
skipped unless the `neural` extra is installed):

```bash
.venv/bin/python -m pytest tests/ -q
```

Viewer smoke test — rebuilds `bundle.js` in a temp dir and checks it matches
the committed one, exports a small fixture scene with the Python package, then
drives the exported page in headless Chromium at desktop, phone and 320px
widths (orbit, step slider, auto-rotate, download links):

```bash
npm ci
npx playwright install chromium   # once; or set CHROME_PATH to a Chrome/Chromium binary
npm run test:viewer
```

The test finds the Python interpreter via `$PYTHON`, then `.venv/bin/python`,
then `python3`.  Set `VIEWER_OUT=some/dir` to keep the fixture and the
screenshots it takes.

CI (`.github/workflows/ci.yml`) runs the Python suite on 3.10 and 3.12 and the
viewer job (bundle up to date, Playwright Chromium) on every push and pull
request.

---

## Licences

MIT for our own code.  Third-party components (Hunyuan3D-2 weights, TripoSR,
DUSt3R, U^2-Net, LDraw catalogue, Rebrickable database) keep their upstream
licenses — see `LICENSE` for the list.  Hunyuan3D-2 is CC-BY-NC; commercial
use of its weights needs a separate agreement with Tencent.
