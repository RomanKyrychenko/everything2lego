---
name: lego-building
description: How real LEGO sets are built. Covers which parts sets use and what for, in which colours, the building techniques for buildings, sculptures, vehicles, micro-scale models and plants, measured over 1,818 official models, and how to look up any real set. Use it when designing or changing a recipe, the part catalogue, the palette or the structure pass, and whenever you decide "how would LEGO build this".
---

# How real LEGO sets are built

The target is a set a buyer could mistake for a store set. Decide
building questions from real sets. Look them up with the tools below
rather than inventing rules.

## Sources (all on the external disk)

| What | Where | Regenerate / use |
|---|---|---|
| Every set's part list (Rebrickable dumps: sets, themes, parts, colours, inventories) | `/Volumes/PRO-G40/lego_data/*.csv.gz` | `tools/set_study/rebrickable.py` loads them |
| 1,818 official models as LDraw (OMR), with steps and sub-models | `/Volumes/PRO-G40/lego_data/omr/files/`, index `omr/sets.txt` | `tools/set_study/ldraw_read.py` flattens them |
| Pattern survey, one row per model | `/Volumes/PRO-G40/lego_data/analysis/patterns.csv` | `tools/set_study/patterns.py <dir> out.csv`, then `by_group.py out.csv` |
| Instruction booklets (PDF) | `/Volumes/PRO-G40/lego_instructions/` | see "Look up a set" |
| Part shapes and connection points | `/Applications/Studio 2.0/ldraw` (`parts/`, `connectivity/*.conn`) | the LDraw library BrickLink Studio ships |
| Which colours each catalogue part is made in, and how real sets use it (per role, joints) | `src/everything2lego/data/part_colours.json`, `part_usage.json` | `tools/partsdb/export.py` |
| The 120 most used parts, by kind of set | `top_parts.md` (next to this file) | |

## The parts database: ask it first

`/Volumes/PRO-G40/lego_data/parts.sqlite` (built and documented in
`tools/partsdb/`) holds everything known about every part:

- colours with set counts and years
- size
- every connector and the side it faces
- usage by kind of set
- which parts share sets (lift)
- which parts are physically joined in the official models, how
  (stacked, SNOT, inverted) and with which offset
- what real-world role a part plays (roof, window, wheel…)

```bash
python tools/partsdb/query.py 3037        # part card
python tools/partsdb/query.py 3039 3037   # how two parts join, seen vs expected
```

From Python (`tools/partsdb/query.py`):

- `PartsDB().for_role("roof")`
- `.connected_to(pn)`
- `.together(pn)`
- `.never_connected(a, b)`
- `.colours(pn)`

Decide part choices from these numbers, not from intuition.

## Look up a set

- **Render an OMR model:** `python tools/render_build.py "/Volumes/PRO-G40/lego_data/omr/files/5766-1.mpd" out.png`.
  Then Read the PNG. Alternate builds are `<set>_<Name>.mpd`.
- **List its parts by orientation:** use `flatten`, `to_model_frame`,
  `describe` and `orientation` from `tools/set_study/`. See how
  `patterns.study` uses them.
- **Find sets by theme or name:** grep `omr/sets.txt` (id|set|name|theme|year|models).
  Rebrickable's `sets.csv.gz` covers every set, including ones not in the OMR.
- **Read a booklet** (techniques the numbers can't show): `curl -A "Mozilla/5.0"
  https://www.lego.com/en-us/service/building-instructions/<set>`, then grep
  for `product.bi.core.pdf/<id>.pdf`. Download to
  `/Volumes/PRO-G40/lego_instructions/<set>_<id>.pdf` and read pages with
  `pages: "40-42"`, at most about 3 pages a call.
  Downloaded already:
  - 31139 Cozy House
  - 31129 Majestic Tiger
  - 76914 Ferrari 812
  - 10311 Orchid

Reference models worth knowing:

| Set | Kind | Why it's useful |
|---|---|---|
| 5766 Log Cabin | small house | 45° slope-brick roof, gable ends in stepped bricks, chimney, baseplate landscaping, 281 parts |
| 31009 Small Cottage | small house | |
| 10246 Detective's Office, 10264 Corner Garage, 10218 Pet Shop, 10270 Bookshop | modular buildings | walls, floors, facades at large scale |
| 21009 Farnsworth House, 21010 Robie House | Architecture | micro scale |
| 10267 Gingerbread House | seasonal building | |

## What the numbers say

### Real sets against our builds

Median piece shares for modern sets (2015 onward, 150+ pieces; Rebrickable):

| category | buildings | sculpture | vehicles | architecture | **our cabin / house** |
|---|--:|--:|--:|--:|--:|
| Plates and special plates | 31% | 35% | 32% | 36% | 31% |
| Bricks (plain) | 9% | 7% | 6% | 7% | **65%** |
| Tiles, all kinds | 15% | 15% | 15% | 23% | **0–0.4%** |
| Slopes and curved | 11% | 13% | 12% | 9% | 3% |
| Round plates and bricks | 9% | 10% | 6% | 8% | 0% |

### How official models sit (1,818 OMR models)

| | 1980s | 1990s | 2000s | 2010s | 2020s | **ours** |
|---|--:|--:|--:|--:|--:|--:|
| Parts sideways (SNOT) | 3% | 4% | 10% | 17% | **32%** | 0% |
| Tiles | 2% | 6% | 10% | 11% | 17% | 0% |
| Plates | 24% | 28% | 31% | 35% | 38% | 31% |
| Bricks | 17% | 12% | 13% | 14% | 17% | 65% |
| Sub-models (sub-assemblies) | 8 | 10 | 10 | 11 | 26 | 1 |
| Parts per step | 9.3 | 7.1 | 4.1 | 3.5 | 3.0 | 94–161 |

Typical sizes:

| Kind of set | Parts, median |
|---|--:|
| Town or Creator house | 250 |
| Modular building | about 2,000 |
| Architecture model | 600 |
| Speed Champions car | 300 |
| BrickHeadz or seasonal sculpture | 160 |
| **Our photo builds** | **5,000–6,400** |

Scale is a design choice: pick the size a real set of that subject would
be.

### Notes on the measures

- Parts within 2 studs of the model's outline are 38% of a modular and 80%
  of a sculpture. Our cabin is 17% and our house 20%, because the rest of
  our parts are interior columns and fill.
- Running bond: in real buildings only about 15% of the vertical joints
  between plain 1×N bricks are staggered over a brick below. Real walls are
  mostly not long runs of bonded 1×N bricks. They are columns, panels,
  window and door units, and SNOT facades. Ours score 9–15% on that
  measure too.

## Parts by role

The most used parts in modern sets of every kind:

- **Plate 1×2 (3023):** the most used part in every kind of set.
- **Plate 1×1 (3024).**
- **Round plate 1×1 (6141).**
- **Tile 1×2 (3069b).**
- **Plate 1×4 (3710).**
- **Cheese slope (54200).**
- **Jumper (15573).**

Full table: `top_parts.md`.

- **Structure (hidden):** plates 2×N and 1×N, bricks 1×2, 1×4, 2×2 and
  2×4. Real sets put bright, mismatched colours inside a model where they
  aren't seen. The tiger's core is red, pink and blue. Colour only what
  shows.
- **Locking layers:** plates laid across brick joints tie a structure
  together. A plate layer every 1–2 bricks is normal, where we use long
  runs of bricks.
- **Surface finish:** tiles 1×1 to 2×4 (3070b, 3069b, 63864, 2431, 6636,
  87079, 3068b) cover studs on floors, walls with SNOT, roof caps and
  tables. Exposed studs are deliberate texture, not the default.
- **Shaping:**
  - cheese slopes 54200, slopes 30° 85984, curved 11477, 15068 and 61678
  - quarter round tiles 25269 and round tiles 98138, 24246
  - corner plates 2420 and wedge plates
  - Sculptures lean on curved slopes, building a smooth skin rather than
    voxel steps.
- **Half-stud offsets:** jumpers 15573 and 34103 centre a part on a
  half-stud grid. Architecture uses them heavily (2.9%).
- **SNOT (studs not on top):**
  - parts: 22885 (1×2×1⅔ with 4 side studs; 2.9% of sculpture parts), 4070
    headlight brick, 87087/11211 bricks with side studs, brackets 99780,
    99781 and 99207, 32952
  - Sideways surfaces in modern sets are built as SNOT panels attached to
    a core.
- **Openings:** window frames 60592 (1×2×2), 60593 (1×2×3) and 60594
  (1×4×3) with glass, door frames, arches.
- **Joints:** clips 15712, 60897 and 61252 and bars 3484 hinge parts. Ball
  joints, and hinge bricks and plates, set angled roofs and limbs.

## Building patterns by kind of model

**Buildings (31139, 5766, modulars):**
- Start from a baseplate or plate base, then a one-stud-thick shell. The
  inside is open floor with furniture.
- Walls mix plain bricks, 1×1 columns, panels, and window and door units,
  locked with plates. Corners use corner bricks 2357 or overlap.
- Roofs are one of two kinds:
  - **Small sets:** rows of 45° slope bricks (2×4, 2×8), with the gable
    ends stepped bricks in the wall or a darker colour.
  - **Newer and larger sets:** plates on hinges set to the pitch, covered
    with tiles or slopes as shingles, with a one-stud eave overhang.
- Chimneys, porches, window boxes and lamps are built as small separate
  sub-assemblies, then attached.
- Landscaping on the base (plants, round plates, flowers 24866) is part
  of the set.

**Sculptures and animals (31129, BrickHeadz):**
- A hidden core of bricks with side studs (22885, 87087) in bright
  internal colours.
- The skin is SNOT panels of curved slopes on that core.
- Limbs and the head are separate sub-assemblies on clips, bars and ball
  joints, so they can pose.

**Vehicles (Speed Champions 76914):**
- A chassis plate, with the body built up in plates.
- Sides, nose and headlight blocks are SNOT sub-assemblies (brackets
  99780 and 99207).
- Grille tiles 2412b are used for detail.

**Architecture (micro):**
- 1×1 and 1×2 plates and tiles with jumpers dominate (half-stud detail).
- Grille tiles and headlight bricks give facade texture.
- Studs are almost always tiled over.

**Botanicals (10311):**
- Stems are Technic axles and connectors.
- Petals and leaves are plant parts and plates on clips and bars, set at
  angles. There are almost no bricks.

## Colours

**Only use part and colour pairs that exist.** The pipeline enforces this.

- **The data:** `data/part_colours.json` lists, for each catalogue part, the
  LDraw colour codes it has been made in since 2010, with how many sets
  used each. Rebrickable numbers some colours differently from LDraw, so
  colours are matched by name.
- **The palette** (`colors.palette()`) keeps only colours that the basic
  bricks and plates (`bricks.CORE_PARTS`) are made in. Retired colours
  drop out: old Brown, old Light Gray, Light Brown, Rust.
- **`bricks.made_in(part, code)`** is the one lookup. Every packer filters
  its parts with it:
  - the wall, roof and slab packers
  - `_tile_mask`, used by piers, floors, decks, the stand and sculpted
    courses
  - bonding merges
  - openings, through `WallFrame.placement`
- **Roof colour:** the roof needs its slope families and ridge tiles at
  run sizes 1, 2 and 4. When the photo's material colour isn't made in
  those, `dominant_colour(..., ok=)` picks the substitute from the photo
  samples at full Lab weight, so a substitute keeps the tone. For example,
  slate that snaps to Sand Blue becomes Dark Bluish Gray, not Dark Blue.
- **Single parts** (sculpt slopes, inverted slopes, window frames) take the
  nearest colour they are made in (`colors.nearest_where`).
- **Still to replace in the catalogue:**

  | Part | Problem |
  |---|---|
  | 30413 | no modern production, so `made_in` already rules it out |
  | 30144 | no modern production, so `made_in` already rules it out |
  | 2549 | no modern production, so `made_in` already rules it out |
  | 3853 | no modern production, so `made_in` already rules it out; also wrongly listed as 6 wide, it is 1×4×3 |
  | 2875 | no modern production, so `made_in` already rules it out |
  | 3939 | slope 33 3×6, made only in transparent colours since 2010 |
