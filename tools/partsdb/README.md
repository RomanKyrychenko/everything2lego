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

## Known gaps

- **Unmapped parts:** about 1.3% of pieces since 1990 (mostly Technic
  pins and hinge assemblies) have LDraw numbers that don't map yet. They
  need an alias table.
- **Contacts:** these only cover parts that appear in the OMR. Common
  parts are there; new moulds may not be.
- **Roles:** these are raw words. A curated mapping to roles (roof, wall,
  window, door, floor, wheel, plant…) is the next step.
