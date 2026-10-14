# LEGO parts database

This database holds everything known about every LEGO part, so the
pipeline can decide by statistics rather than hand-written rules: which
part to use, in which colour, and what it connects to.

It is one SQLite file, `$PARTSDB` (default
`/Volumes/PRO-G40/lego_data/parts.sqlite`). It is built from open data and
is not stored in the repo:

| Source | What it gives |
|---|---|
| Rebrickable CSV dumps in `$LEGO_DATA` | every set's part list since 1949 |
| LDraw parts library (`$LDRAWDIR`, BrickLink Studio's copy) | part geometry |
| LDCad shadow library (`$LDCAD_SHADOW`, CC BY-SA 4.0) | the connection points (studs, sockets, clips, bars, pins, axles, hinges, balls) of the LDraw primitives and parts |
| 1,818 official models (OMR) in `$LEGO_DATA/omr/files` | how parts are really put together |

## Build

```bash
python tools/partsdb/build.py                    # all stages, about 40 min
python tools/partsdb/build.py --stage contacts   # one stage
```

| Stage | Writes |
|---|---|
| `catalog` | `part`, `colour`, `part_colour`, `part_theme`, `relationship` |
| `geometry` | `ldraw_name`, `geometry`, `connector`, `connector_summary` |
| `inherit` | Prints and mould variants without their own LDraw file get their family's rows (`inherited_from`) |
| `cooccur` | `cooccur` |
| `contacts` | `contact`, `part_omr` |
| `roles` | `role` |

Two more steps run on their own:

```bash
python tools/partsdb/shapes.py            # part_shape: each grid part's shape and connections, about 1 min
python tools/partsdb/kinds.py             # set_kinds.csv: what each set builds (local LLM via ollama, about 2 h, resumable)
python tools/partsdb/kinds.py affinity    # part_kind: what each part is for
python tools/partsdb/vocabulary.py        # src/everything2lego/data/vocabulary.json for the sampler
```

## Query

```bash
python tools/partsdb/query.py 3023          # part card
python tools/partsdb/query.py 3001 3023     # every recorded way two parts connect, and seen vs expected
```

From Python: `from query import PartsDB` (see `query.py`).

## Tables

Part numbers are Rebrickable's (`part.part_num`). `ldraw_name` maps LDraw
file names to them.

### `part`

`part_num, name, category, system, year_from, year_to, n_sets, n_pieces, n_sets_recent, n_pieces_recent, family`

- **`system`:** false for minifigure parts, stickers, Duplo and the like.
- **`family`:** the mould the part is a print or variant of.
- **`*_recent`:** counts from sets since 1990.

### `colour`

`rb_id, name, rgb, is_trans, ldraw_code, bl_id, year_from, year_to`

### `part_colour`

`part_num, rb_colour, ldraw_code, colour, n_sets, n_pieces, year_from, year_to`

Every colour a part has been made in.

### `part_theme`

`part_num, group, n_pieces, share`

`share` is the part's share of all pieces in sets of that kind: buildings,
sculpture, vehicles, architecture, botanical, other.

### `relationship`

`kind, child, parent`

Rebrickable's print, mould, alternate, pair and sub-part links.

### `geometry`

`part_num, ldraw, x/y/z_min/max (LDU), w_studs, d_studs, h_plates`

LDraw coordinates, with y pointing down.

### `connector`

`part_num, kind, gender, x, y, z, fx, fy, fz, shape`

- **Kinds:** stud, antistud, bar, barhole, clip, pin, pinhole, axle,
  axlehole, ball, socket, finger, generic.
- **`f*`:** the side a connector is met from, in the part's own frame.
  `fy = -1` is up: a stud on top, or a socket underneath met from below.
  The other values mean sideways: studs not on top, or SNOT.
- **Implicit sockets:** a System part takes a stud under each of its own
  studs even where the shadow library leaves that out (1×N plates). These
  rows have shape `implicit`.

### `connector_summary`

One row per part, with counts per kind and facing: `stud_up`,
`stud_side`, `antistud_down`, `clip` and so on.

### `cooccur`

`part_a, part_b, n_sets, p_b_given_a, lift`

- **Scope:** parts in at least 20 sets since 1990, each with its 60
  strongest partners.
- **`lift`:** how much more often the two share a set than chance would
  give.

### `contact`

`male, female, joint, relation, n, n_models, top_dx, top_dy, top_dz, top_rot, top_share`

- **What counts:** a male connector of one part meeting a matching female
  of another in an official model.
- **`joint`:** for example `stud-antistud`, `bar-clip`, `pin-pinhole`.
- **`relation`:** how the two parts sit, judged in each part's own frame:
  - `stacked`: a top stud into a bottom socket
  - `snot`: a connector on a side
  - `inverted`: one part turned over
- **`top_*`:** the commonest placement of the female part in the male's
  frame (LDU, rounded) and the share of contacts that use it.
- **Never connected:** two common parts that are absent here, or rarer
  than chance (`PartsDB.never_connected`), are a pairing set designers
  avoid.

### `part_omr`

`part_num, n_models`

The number of official models a part appears in, as the base for
"expected" contact counts.

### `role`

`part_num, word, n, share, tfidf`

Words from the names of the official models' sub-models that a part sits
in ("roof", "floor", "wheel", "tree"): what real-world thing it builds.
The set's own title words are left out.

### `part_shape`

`part_num, ldraw, w, d, h, origin_x, origin_y, origin_z, x0, y0, z0, occ, bits, fill, studs_top, holes_bottom, sides, offgrid`

The shape of every part whose body sits on the build grid (654 parts:
bricks, plates, tiles, slopes, curves, rounds, wedges and specials). The
footprint is a whole number of studs and the height a whole number of
plates; studs on top are not counted.

- **Box and origin:** `w` x `d` studs by `h` plates. `origin_*` is the
  box's footprint centre and body top in the part's own frame (LDU).
- **Shape:** `bits` gives each cell of the box (`[y from the bottom][x][z]`)
  as 4x4x4 samples packed into a 64-bit mask. `occ` is each cell's filled
  share.
- **How the shape is measured:** it is the visual hull of the part's three
  axis views. That is exact for slopes, curves, rounds, wedges and corners,
  and ignores the hollow underside every brick has.
- **Connections:**
  - `studs_top` and `holes_bottom` list the cells with a stud on top or a
    hole underneath.
  - `sides` lists studs and holes on the part's sides, with the face they
    point out of (`+x`, `-x`, `+z`, `-z`).
  - `offgrid` counts connectors that sit off the cell grid, such as a
    jumper's centred stud.

### `part_kind`

`part_num, kind, n_pieces, n_sets, share, set_share, lift`

What each part is for. Every set with 30 or more parts, and every official
model, is labelled by a local language model from its name and theme
(`kinds.py`) with one of 13 kinds:

- **Buildings and vehicles:** building, road_vehicle, train, aircraft,
  watercraft, spacecraft.
- **Figures and things:** robot, creature, character, plant, object.
- **Other:** scene, and parts (loose-brick packs, left out of the table).

The columns, per part and kind:

- **`set_share`:** the share of that kind's sets that use the part.
- **`share`:** the part's share of that kind's pieces.
- **`lift`:** that share divided by the part's share of all pieces,
  smoothed toward 1. A wheel arch has lift well above 1 for road vehicles
  and well below 1 for buildings.

## Coverage

- **Mapping:** LDraw files reach Rebrickable parts by the same number, the
  LEGO design ID (through Studio's part table), the BrickLink number, the
  mould letter and Rebrickable's mould links (`ldraw_name.how`).
- **Geometry:** the official ldraw.org library comes first
  (`$LDRAW_OFFICIAL`). Geometry covers 99.3% of the pieces in sets since
  1990. The rest is non-building parts: watch links, rubber bands.
- **Roles:** come from the innermost named sub-model around a part. A
  role needs at least 3 official models showing it.
- **Contacts:** only cover parts that appear in the OMR.

## Used by the pipeline

`tools/partsdb/vocabulary.py` writes `src/everything2lego/data/vocabulary.json`,
read by the statistics sampler (`everything2lego.sampler`). It holds every
grid part in production with its shape masks, connections, statistics
token, colours and per-kind use. The sampler draws a build's parts from the
plain bricks, plates and tiles, plus the parts most widely used by sets of
the object's kind.

`tools/partsdb/export.py` writes `src/everything2lego/data/part_colours.json`
and `part_usage.json`, read by `everything2lego.usage`:

- **Colours:** parts are only used in the colours LEGO makes them in now
  (sets since 2010).
- **Run splitting:** a wall course, a roof row or a ridge is split the way
  real sets most often split it for that role (`part_costs`). Wall courses
  still bond: joints off the joints below, and off the sills and lintels
  of openings.
- **Report:** `joints:unlike_sets_pct` is the share of a build's stud
  joints between part pairs that real sets join no more than 0.2 times as
  often as chance.
