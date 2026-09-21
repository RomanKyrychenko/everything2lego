# everything2lego: Next-Generation Architecture

This document describes two fundamentally different pipelines being built
alongside the current voxelize-and-pack pipeline. They target the ceiling the
current pipeline hit: uniform voxelization loses geometry, greedy packing
doesn't think in architectural elements, and post-hoc styling passes fight
each other.

## Why not just keep patching the current pipeline

- **Uniform voxelization is destructive.** Curved rooflines quantize to
  staircases; round windows quantize to square patches. Subvoxels only
  mitigate the loss factor-by-factor, never remove it.
- **Bottom-up greedy packing is not how LEGO designers work.** They think in
  terms of walls, roofs, openings, details — not "biggest brick that fits
  at (x, y, z)."
- **Every post-hoc pass fights every other post-hoc pass.** `--texture`,
  `--variety`, `--architectural-style`, `--detect-features` all rewrite each
  other's decisions. Order matters. Adding a 6th pass makes it worse.
- **Colors carry no semantics.** A blue-gray patch could be window, wall
  accent, or reflection. We do color→palette snap, then reason about
  "clusters" post-hoc — reversed causality.
- **No structural planning.** Stability is checked after packing; nothing
  during packing steers *away* from unbuildable states.

## Two pipelines, complementary

Both consume the same input (mesh + optional photos) and produce the same
output (LDraw placements + report). They share the palette, part catalog,
Placement type, and IO writers. They differ in how they *decide what to
build*.

```
                  ┌───────────────────────┐
   Mesh + Photos →│  Existing voxel+pack  │→ placements  (baseline)
                  └───────────────────────┘

                  ┌───────────────────────┐
   Mesh + Photos →│    Template pipeline  │→ placements  ← ships fastest,
                  └───────────────────────┘                 known types
                                                            LEGO-Ideas quality

                  ┌───────────────────────┐
   Mesh + Photos →│    Semantic pipeline  │→ placements  ← ships slowest,
                  └───────────────────────┘                 general purpose
                                                            highest ceiling
```

Route selection at CLI level:
```bash
everything2lego build            mesh.glb                # existing greedy pack
everything2lego template         gable-cabin  --width 32 # parametric template
everything2lego semantic         mesh.glb                # semantic decomposition
```

---

## Pipeline (c): Templates + parametric fitting

**Idea.** Curate a small library of hand-designed LEGO architectures at
LEGO-Ideas quality. Classify the input into one of them. Fit the template's
parameters (width, depth, height, roof pitch, window count) to the mesh.
Optionally project photo colors onto the wall/roof surfaces. Output the
fully-designed model.

**Strengths.**
- Each template is *guaranteed* LEGO-Ideas-quality — a human designed it.
- No voxel discretization loss inside the template — parametric geometry.
- Fast path: fitting is minutes, not the multi-tens-of-minutes voxel pack.

**Weaknesses.**
- Limited to known architectural types.
- Each new type = days of hand design.
- If input doesn't match any template, output is unfaithful.

### Template protocol

```python
class Template(Protocol):
    name: str
    param_schema: type   # dataclass declaring the parameters
    def fit(self, mesh: trimesh.Trimesh) -> Params: ...  # optional
    def build(self, params: Params) -> list[Placement]: ...
```

- `param_schema` is a dataclass. Every parameter has a default + range.
- `fit` is optional: if a template can guess parameters from a mesh, it
  should; otherwise the CLI passes them explicitly.
- `build` is deterministic: same params → same placements.

### Template library layout

```
src/everything2lego/templates/
├── __init__.py            registry: {name: Template class}
├── base.py                Template protocol + shared helpers
├── gable_cabin.py         ✓ shipped: rectangular cabin, gable roof
├── castle_tower.py        planned
├── modernist_box.py       planned
├── a_frame.py             planned
└── vintage_house.py       planned  (multi-story, ornate — Home Alone style)
```

### Fitting

Two modes:
1. **Manual**: user passes parameters via CLI. Best for prototyping.
2. **Auto**: `fit(mesh)` estimates from mesh bounds + heuristics
   (bounding-box dims → width/depth/height, PCA on top half → roof pitch,
   photo color quantization → wall/roof colors).

Auto fitting is a *per-template* concern: `gable_cabin` guesses roof pitch
from the mesh's top-half y-z slice; `castle_tower` looks for circular
cross-sections; etc.

### Photo-derived colors

Reuse `voxelize.project_photo_colors()` on template-generated placements:

```python
placements = template.build(params)
if photo:
    placements = project_colors_onto_placements(placements, photo, params)
```

Simpler and more accurate than voxelizing the source mesh.

---

## Pipeline (a): Semantic decomposition + specialized builders

**Idea.** Decompose the mesh into semantic regions (foundation, walls, roof,
openings, details). Give each region to a specialist builder that knows the
right LEGO techniques for it. Reconcile the builders' outputs into a
coherent whole.

**Strengths.**
- General purpose: works on shapes no template matches.
- Correct-by-construction: each builder makes locally-optimal choices.
- Extensible: adding a new detail type = one new builder.

**Weaknesses.**
- Big: needs a segmenter, ≥5 builders, a reconciler.
- Each builder has its own subtle placement logic.
- Segmentation errors cascade.

### Semantic pipeline layout

```
src/everything2lego/semantic/
├── __init__.py            top-level: semantic_build(mesh) → placements
├── segment.py             mesh/voxel → region labels
├── regions.py             Region dataclass (label, voxels, metadata)
├── builders/
│   ├── __init__.py        registry: {label: Builder class}
│   ├── foundation.py      ✓ scaffolded: bottom slab detection + build
│   ├── wall.py            planned: shell with SNOT / quoin awareness
│   ├── roof.py            planned: pitched with cheese-slope shingles
│   ├── opening.py         planned: windows, doors, arches
│   └── detail.py          planned: chimney, dormers, protrusions
└── reconcile.py           planned: stud-grid alignment, color coherence
```

### Stage 1: Segmentation

```python
def segment(voxels: np.ndarray) -> dict[str, Region]:
    """Voxel grid → semantic regions.

    Uses geometric heuristics:
      - foundation:  bottom ≤3 plates
      - walls:       vertical shell above foundation, below roof line
      - roof:        pitched region above the walls
      - openings:    holes in walls (mesh-level: face normals that flip)
      - details:     protrusions (chimneys, dormers, porches)
    """
```

### Stage 2: Per-region builders

Each builder has the same signature:
```python
class Builder(Protocol):
    def build(self, region: Region, context: BuildContext) -> list[Placement]:
        ...
```

`BuildContext` gives access to:
- The global palette
- The catalog (filtered to what this builder is likely to use)
- Neighboring regions (so a wall builder can ask "where's the roof line?")
- Photo (for color projection on that region)

### Stage 3: Reconciliation

Given per-region placements:
- Align stud grids (make sure a wall ends where a roof starts)
- Coalesce colors (same palette index across the seam)
- Resolve overlaps (roof and wall may both claim the corner brick)
- Verify stability holistically

---

## Shared foundations (both pipelines use these)

- **Placement dataclass** and IO writers from `pack.py` and `io_ldraw.py`.
- **Catalog and palette** from `bricks.py` and `colors.py`.
- **Photo color projection** from `voxelize.project_photo_colors`.
- **Report format** from `pipeline.py`.
- **Viewer** from `viewer_export.py`.

Neither new pipeline touches the existing `pack.py` greedy packer. That
stays as the baseline `everything2lego build` command.

---

## What ships in this round

| | Status |
|---|---|
| `ARCHITECTURE.md` | This document |
| `templates/` — Template protocol, registry | ✓ |
| `templates/gable_cabin.py` — working template | ✓ |
| CLI: `everything2lego template gable-cabin` | ✓ |
| `semantic/segment.py` — Foundation segmenter | ✓ |
| `semantic/builders/foundation.py` — working builder | ✓ |
| CLI: `everything2lego semantic` | scaffold only |
| Wall / Roof / Opening / Detail builders | planned |

---

## Roadmap

**Short term (days each):**
- Add `castle_tower` template
- Add `modernist_box` template
- Photo-derived color parameter fitting

**Medium term (1-2 weeks):**
- Semantic Wall builder with SNOT-aware wall runs and corner quoins
- Semantic Roof builder with cheese-slope shingle patterns
- Segmenter: opening detection via mesh face-normal flips

**Long term (research):**
- Vintage-house / Home-Alone-style template with dormers and ornate trim
- Learned template classifier (image → template type)
- BrickGPT-style search backend as a third pipeline
