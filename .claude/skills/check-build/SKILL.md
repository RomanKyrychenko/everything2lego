---
name: check-build
description: Build an everything2lego model in set mode, render it with real LDraw parts from five sides, and review it against the photo and the store-set checklist. Use after any change to the set-design, massing, facade or structure code, and whenever judging how good a build is — never judge a build by its counts alone.
---

# Check a build

The goal is a set built the way a store set is built, faithful to the
photo. Part and component counts hide most defects. Always look at the
render.

## 1. Build and render

```bash
.claude/skills/check-build/build_and_render.sh <mesh.glb> <out_dir> [facade.npz]
```

This writes `model.ldr`, `report.json`, `build.log` and `render.png` (views
at longitudes 0, 45, 90, 180 and 270) and prints the structure numbers.
Put `out_dir` in the session scratchpad, not the repo.

Reference inputs:

| Model | Mesh | Facade | Photo |
|---|---|---|---|
| House | `out_photo/mesh/q.glb` | `out_photo/facade_rgb.npz` | `out_photo/photo_clean.png` |
| Cabin (¾ view) | `out_photo_cabin3/mesh/q.glb` | `out_photo_cabin2/facade_rgb.npz` | `out_photo_cabin3/photo_clean.png` |

Rebuild both after a pipeline change, and run them in parallel. Each takes
about a minute. If a facade is missing, `tools/facade_depth.py` recreates it.
That run needs MapAnything at about 10 GB, so run it alone.

To render any `.ldr`, run `python tools/render_build.py model.ldr out.png
[--views 0,45,90 --top --size 600]`. It uses LDView from
`/Volumes/PRO-G40/ldview` and the parts library that ships with
BrickLink Studio. It reports parts missing from the library, and any
missing part is a catalogue bug to fix in `data/brick_catalog.json`.

## 2. Look

Read `render.png` and the photo side by side, then go through:

1. **Walls are walls.** They should be bonded bricks in the wall's own
   colour. Grey (71) 1×1 columns inside a wall line mean the plan didn't
   build that wall, and `brace()` propped up what was above it.
2. **The roof sits on the walls.** Eaves overhang by at most 1–2 studs. The
   slopes are continuous with a ridge, not loose slope pieces scattered
   over a terrace.
3. **Silhouette and proportions match the photo.** Check the number of
   storeys, the roof pitch, the towers and chimneys, and where the porch is.
4. **Colours match the photo.** Check the wall material, the roof colour,
   and that the windows are visible and in the right places.
5. **It is one assembly.** Nothing floats, `components` is 1, and there is
   no debris.
6. **It looks like a store set.** Openings use real window and door parts.
   The building has a base plate. Hidden support stays hidden: grey inside
   is fine, grey on the outside is not.

## Reference: how real sets do it

When a recipe has to decide how something is built (walls, roof, base,
porch, openings), check a real set before you invent a rule. Official
instructions are linked from
`https://www.lego.com/en-us/service/building-instructions/<set number>`.
Fetch that page with `curl -A "Mozilla/5.0"`, then grep it for
`product.bi.core.pdf/<id>.pdf`. Each link is one booklet. Save booklets to
`/Volumes/PRO-G40/lego_instructions/` and read pages with the Read tool
(`pages: "112-115"`).

Already downloaded: 31139 Cozy House, 3-in-1, booklets `31139_6456165`
(144 pp), `…167` (96 pp) and `…169` (116 pp).

What these sets show:

- The walls are a hollow shell one stud thick on a base plate. The inside
  is open floor.
- The roof is plates, usually hinged to the pitch, with tiles or shingles
  and a one-stud eave overhang.
- Chimneys, porches and window boxes are small sub-assemblies attached to
  the shell.

## 3. Report and fix

- Say what is wrong in the picture, which item above it breaks, and the
  plan-level cause once you have found it. Fix the cause in the plan
  (setdesign, massing, facade), not by post-processing parts. The project
  rule is fundamental fixes only.
- After a fix, rebuild and re-render both reference models, compare them
  with the previous renders, then run `pytest -q`.
- A pipeline step can be verified with a synthetic test, such as
  `tests/test_setdesign.py` and `tests/test_facade.py`. Add one for the
  defect you fixed.
