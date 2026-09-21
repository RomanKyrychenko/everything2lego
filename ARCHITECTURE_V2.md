# Architecture v2 — Semantic Decomposition + MILP

## Motivation

The v1 pipeline (SA over per-cell types) has hit a ceiling.  Concrete symptoms:

1. **Loss ≠ final output.** The SA optimises cell-level loss, but the export
   consolidator (deterministic greedy) decides the actual parts.  SA has no
   way to know its cell layout produces N vs 2N parts.
2. **Voxel bottleneck.** A uniform grid loses sub-plate detail forever; slopes
   are inferred from voxel edges, not from the mesh.
3. **No LEGO priors.** Real MOCs use technique (SNOT, half-stud offset, arch,
   hinges).  The SA only picks from a per-cell type dictionary.
4. **Greedy packing is suboptimal.** Even with size-first sweeps, a
   size-first-across-positions scan can't backtrack when a big rectangle
   fits at a non-scanned origin.

## The v2 pipeline

```
mesh (STL / GLB)
   │
   ▼
MeshTarget            — samples surface, normals, colours; caches voxel grid
   │
   ▼
RegionDetector        — RANSAC-fits primitives (planes, cylinders, spheres);
   │                    emits typed Region subclasses
   │
   ▼
[Region, Region, …]   — PlanarRegion, CylindricalRegion, SphericalRegion,
   │                    CornerRegion, FoundationRegion, DetailRegion
   │
   ▼ (dispatch)
RegionBuilder         — abstract; concrete subclasses per Region type
   │
   ▼
Assembly              — list[Placement] union across regions
   │
   ▼
Export (existing)     — LDR / STL / viewer / PDF
```

Each RegionBuilder converts one Region into an *optimal* placement set, using
**MILPPacker** internally for its 2D exact-cover subproblem.

## Class hierarchy

```
MeshTarget            (mesh + sampled point cloud + palette-quantised colours)

Region                (abstract)
├── PlanarRegion      (wall, roof plane, floor — flat surface with a normal)
├── CylindricalRegion (round tower, pillar, drum roof)
├── SphericalRegion   (dome, orb, blob)
├── CornerRegion      (junction between two planes)
└── FoundationRegion  (ground-level slab, always PlainRegion horizontal)

RegionDetector        (RANSAC + normal-clustering on mesh faces)

RegionBuilder         (abstract; can_build(region) → bool; build(region) → placements)
├── PlanarWallBuilder      (vertical planar → brick course + optional SNOT)
├── PlanarRoofBuilder      (sloped planar → slope-ring)
├── FoundationBuilder      (horizontal planar at y=0 → plate tiling)
├── CylindricalBuilder     (curved-slope stack around axis)
├── SphericalBuilder       (dome pattern: cheese slope crown + curved slope belt)
└── GenericVoxelBuilder    (fallback: voxelise + greedy pack)

MILPPacker            (rectangle exact-cover on a 2D mask via scipy.optimize.milp)

SemanticLegoPipeline  (orchestrator: mesh path → Assembly)
```

## MILP formulation (per-region 2D pack)

For a same-colour same-type mask M ⊂ [0..W) × [0..D) and a catalog of usable
rectangle parts (footprint × orientation × colour choice):

**Variables**

  * `x[i, ox, oz, rot] ∈ {0, 1}` — place part `i` at origin `(ox, oz)` with
    rotation `rot`.  Only enumerated for tuples whose footprint fits inside
    M without extending outside it.

**Constraints**

  * ∀ cell `(x, z) ∈ M`:  Σ over placements that cover this cell = 1
    *(exact cover)*
  * ∀ cell `(x, z) ∉ M`:  Σ = 0
    *(implicit — variables that would cover such cells are pruned before
    the MILP is instantiated)*

**Objective**

    minimise  Σ x[i, ox, oz, rot]  +  λ · |distinct part types used|

  where the second term is linearised via an auxiliary `y[i] ∈ {0, 1}` with
  `y[i] ≥ (Σ_positions x[i, ·, ·, ·]) / M_i` and objective term `+λ · Σ y[i]`.

Solved by `scipy.optimize.milp` with a per-region time limit (default 30 s).
Falls back to the existing size-first greedy if MILP times out or returns
infeasible.  Regions bigger than ~40 × 40 studs are chunked before ILP.

## Concrete Builder behaviours

| Builder | Region | Technique |
|---|---|---|
| `FoundationBuilder` | horizontal slab at ground | `MILPPacker` on plate catalog |
| `PlanarWallBuilder` | vertical plane | brick courses via `MILPPacker`; optional 1-plate SNOT layer at height changes |
| `PlanarRoofBuilder` | tilted plane | slope-brick ring at bottom edge, plates behind |
| `FoundationBuilder` | horizontal floor | plate tile |
| `CylindricalBuilder` | cylinder | curved-slope 4×1 (61678) stacks in rings; cheese slopes fill remainders |
| `SphericalBuilder` | sphere | curved 2×2 (15068) at equator; cheese slopes toward poles; tile at top |
| `CornerRegion` | 2-plane junction | corner slope 2×2 (3045) or SNOT header brick |
| `GenericVoxelBuilder` | unclassified | current voxel+pack fallback (safety net) |

## What's kept from v1

* `everything2lego.bricks.catalog()` — 419-part catalog
* `everything2lego.colors.palette()` — 55-colour palette
* `everything2lego.io_ldraw`, `io_stl`, `io_bricklink`, `viewer_export`,
  `instructions` — all export surfaces
* `everything2lego.pack.Placement` — the atomic output object

## What's replaced

* `everything2lego.optimize.*` — the SA loop, cell types, greedy consolidator.
  Kept in tree as `GenericVoxelBuilder` fallback but no longer the primary
  pipeline.

## Roadmap

1. **Phase 1** — scaffolding + `MILPPacker` proven on a synthetic 2D mask.
2. **Phase 2** — `RegionDetector` with plane RANSAC.  `FoundationBuilder`
   + `PlanarWallBuilder` end-to-end on a synthetic box.
3. **Phase 3** — `PlanarRoofBuilder` + `CornerRegion` + smoke on gable house.
4. **Phase 4** — `CylindricalBuilder`, `SphericalBuilder` — smoke on sphere.
5. **Phase 5** — `GenericVoxelBuilder` wraps v1 pipeline; enable it as
   fallback for any region the detector can't classify.
6. **Phase 6** — deprecate `everything2lego optimize` CLI in favour of
   `everything2lego semantic`.
