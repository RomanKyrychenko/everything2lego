"""Import Rebrickable parts + colors into the everything2lego catalog.

Run:
    arch -arm64 .venv-arm64/bin/python scripts/import_rebrickable.py

Reads:
    data/rebrickable/colors.csv
    data/rebrickable/parts.csv
    data/rebrickable/part_categories.csv
    data/rebrickable/inventory_parts.csv

Writes / updates:
    src/everything2lego/data/ldraw_colors.json     (adds currently-produced solid colors)
    src/everything2lego/data/imported_parts.json   (new file — geometric parts by frequency)

The catalog loader in `bricks.py` reads both the hand-curated list and the
imported file, dedupes by LDraw filename, and hands them to `catalog()` in a
single stream.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from everything2lego.bricks import height_from_name, is_plain_geometry  # noqa: E402

ROOT = Path(__file__).parent.parent
RB = ROOT / "data" / "rebrickable"
DATA = ROOT / "src" / "everything2lego" / "data"

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------

# Rebrickable color IDs match LDraw codes for standard solid colors.
def import_colors(min_num_parts: int = 5000, allow_trans: bool = False) -> None:
    colors_path = RB / "colors.csv"
    if not colors_path.exists():
        print(f"[skip colors] {colors_path} missing", file=sys.stderr)
        return

    existing_path = DATA / "ldraw_colors.json"
    existing = json.loads(existing_path.read_text())
    existing_codes = {c["code"] for c in existing["colors"]}

    added = 0
    skipped_trans = 0
    with open(colors_path) as f:
        for row in csv.DictReader(f):
            try:
                code = int(row["id"])
            except ValueError:
                continue
            if code < 0:
                continue  # -1 "Unknown"
            if code in existing_codes:
                continue
            is_trans = row["is_trans"].strip().lower() == "true"
            if is_trans and not allow_trans:
                skipped_trans += 1
                continue
            num_parts = int(row["num_parts"] or "0")
            if num_parts < min_num_parts:
                continue
            try:
                y2 = int(row["y2"] or "0")
            except ValueError:
                y2 = 0
            if y2 and y2 < 2015:  # long-discontinued
                continue
            rgb_hex = row["rgb"]
            try:
                r, g, b = (int(rgb_hex[0:2], 16),
                            int(rgb_hex[2:4], 16),
                            int(rgb_hex[4:6], 16))
            except ValueError:
                continue
            existing["colors"].append({
                "code": code,
                "name": row["name"],
                "bl": code,  # BrickLink IDs mostly match; a curator can fix later
                "rgb": [r, g, b],
            })
            added += 1

    # Sort by code for readability.
    existing["colors"].sort(key=lambda c: c["code"])
    existing_path.write_text(json.dumps(existing, indent=2) + "\n")
    print(f"colors: added {added}, skipped {skipped_trans} trans, total {len(existing['colors'])}")


# ---------------------------------------------------------------------------
# Parts
# ---------------------------------------------------------------------------

# Rebrickable part_cat_id → our internal shape family. Skip Technic-only cats.
CATEGORY_TO_SHAPE = {
    1:  "plate",       # Baseplates — treat as thin plates (large)
    3:  "slope",       # Bricks Sloped
    5:  None,          # Bricks Special — mixed; parsed by name
    6:  "slope",       # Bricks Wedged
    9:  "plate",       # Plates Special — mixed; try plate first
    11: "brick",       # Bricks
    14: "plate",       # Plates
    15: "tile",        # Tiles Special
    16: None,          # Windows and Doors — split by name
    19: "tile",        # Tiles
    20: "brick",       # Round Bricks & Cones (round bricks land here; tile-wise)
    21: "plate",       # Round/curved plates
    23: "panel",       # Panels
    37: "slope",       # Bricks Curved (curved slopes belong to slope family)
    49: "plate",       # Plates Wedged
    67: "tile",        # Round/curved tiles
}

# Regex to extract WxDxH from a part name like "Brick 1 x 2" or
# "Panel 1 x 4 x 3" or "Brick Sloped 30° 1 x 2 x 2/3".
# Height dim can be integer ("3") or fractional ("2/3") — the fractional
# form denotes a partial-brick height common on low-angle slopes.
_WXD = re.compile(
    r"\b(\d+)\s*[xX]\s*(\d+)(?:\s*[xX]\s*(\d+)(?:\s*/\s*(\d+))?)?"
)

# For classifying by name when category is ambiguous.
_NAME_SHAPE_HINTS = (
    ("Slope Inverted", "slope_inv"),
    ("Sloped Inverted", "slope_inv"),
    ("Slope Curved",   "slope"),
    ("Slope",          "slope"),
    ("Wedge",          "slope"),
    ("Panel",          "panel"),
    ("Window",         "window"),
    ("Door",           "door"),
    ("Tile",           "tile"),
    ("Plate",          "plate"),
    ("Brick",          "brick"),
)


def _parse_geometry(name: str, cat_shape: str | None) -> tuple[str, tuple[int, int], int] | None:
    """Return (shape, footprint (w, d), height_in_plates) or None if not usable.

    Heuristics based on part naming convention:
       "Brick 1 x 2"        → (2, 1),  3 plates
       "Plate 2 x 4"        → (4, 2),  1 plate
       "Tile 1 x 6"         → (6, 1),  1 plate
       "Slope 45 2 x 3"     → (3, 2),  3 plates
       "Panel 1 x 4 x 3"    → (4, 1),  9 plates (third dim × 3)
       "Window 1 x 4 x 3 Frame" → (4, 1), 9 plates
    """
    # Determine shape name from hints (name takes precedence over cat_shape).
    shape = None
    for hint, s in _NAME_SHAPE_HINTS:
        if hint in name:
            shape = s
            break
    if shape is None:
        shape = cat_shape
    if shape is None:
        return None

    if not is_plain_geometry(shape, name):
        return None
    m = _WXD.search(name)
    if not m:
        return None
    w = int(m.group(1))
    d = int(m.group(2))
    # Normalize footprint: (max, min) → LDraw parts are canonically wider on x.
    fw, fd = max(w, d), min(w, d)

    if shape not in ("plate", "tile", "brick", "slope", "slope_inv", "panel", "window", "door"):
        return None
    h = height_from_name(shape, name)

    # Reject anything unreasonably large — noisy names or oversized parts.
    if fw > 16 or fd > 16 or h > 30:
        return None
    if fw == 0 or fd == 0:
        return None
    return shape, (fw, fd), h


def _rank_parts_by_frequency(sample_lines: int = 400_000) -> Counter:
    """Sample the inventory_parts.csv to rank parts by inclusion count.
    We sample rather than full-scan the 133 MB file — the ranking is stable
    across the first ~400k rows."""
    freq: Counter = Counter()
    with open(RB / "inventory_parts.csv") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= sample_lines:
                break
            freq[row["part_num"]] += int(row["quantity"] or "1")
    return freq


def import_parts(top_n: int = 300) -> None:
    parts_csv = RB / "parts.csv"
    if not parts_csv.exists():
        print(f"[skip parts] {parts_csv} missing", file=sys.stderr)
        return

    # 1. Load category map.
    cats: dict[int, str | None] = {}
    with open(RB / "part_categories.csv") as f:
        for row in csv.DictReader(f):
            cats[int(row["id"])] = CATEGORY_TO_SHAPE.get(int(row["id"]))
    useful_cats = {cid for cid, s in cats.items() if s is not None or cid == 16 or cid == 5}

    # 2. Load parts filtered by useful categories.
    print("Loading parts.csv...")
    parts: dict[str, dict] = {}
    with open(parts_csv) as f:
        for row in csv.DictReader(f):
            cat_id = int(row["part_cat_id"] or "0")
            if cat_id not in useful_cats:
                continue
            parts[row["part_num"]] = {
                "part_num": row["part_num"],
                "name": row["name"],
                "cat_id": cat_id,
            }
    print(f"  {len(parts)} parts across useful categories")

    # 3. Rank by inventory frequency (sampled).
    print("Ranking parts by inventory frequency (sampled)...")
    freq = _rank_parts_by_frequency()
    _clean_part_num = re.compile(r"^\d+[a-z]?$")
    scored = []
    for p in parts.values():
        # Skip printed/pattern variants (e.g. "3040bpr0003") — accept only
        # bare "\d+" or "\d+[a-z]" (one letter suffix for mold variants).
        if not _clean_part_num.match(p["part_num"]):
            continue
        # Skip decorated parts by name.
        low = p["name"].lower()
        if "pattern" in low or "print" in low or "sticker" in low:
            continue
        f_ = freq.get(p["part_num"], 0)
        if f_ == 0:
            continue
        cat_shape = cats.get(p["cat_id"])
        geom = _parse_geometry(p["name"], cat_shape)
        if geom is None:
            continue
        shape, footprint, height = geom
        scored.append((f_, p["part_num"], p["name"], shape, footprint, height))
    scored.sort(reverse=True)
    print(f"  {len(scored)} parts with parseable geometry + non-zero frequency")

    # 4. Dedupe against the existing catalog by BASE part number (letter
    # suffixes like 3040a / 3040b are mold variants — if any variant is in
    # the curated catalog, skip all others).
    def _base_id(ldraw: str) -> str:
        """Base part number: leading digits only (drops mold-variant letters
        and any printed/pattern suffix)."""
        stem = ldraw.replace(".dat", "")
        m = re.match(r"^(\d+)", stem)
        return m.group(1) if m else stem

    from everything2lego.bricks import catalog  # noqa: E402  (existing catalog)
    existing_bases = {_base_id(p.ldraw) for p in catalog()}

    imported = []
    seen_bases: set[str] = set()
    for f_, part_num, name, shape, (fw, fd), h in scored:
        ldraw = f"{part_num}.dat"
        base = _base_id(ldraw)
        if base in existing_bases or base in seen_bases:
            continue
        seen_bases.add(base)
        imported.append({
            "ldraw": ldraw,
            "bl": part_num,
            "shape": shape,
            "footprint": [fw, fd],
            "height_units": h,
            "name": name,
            "slope_dir": "+x" if shape in ("slope", "slope_inv") else None,
        })
        if len(imported) >= top_n:
            break

    # 5. Write out.
    out_path = DATA / "imported_parts.json"
    out_path.write_text(json.dumps({
        "_comment": "Auto-imported from Rebrickable via scripts/import_rebrickable.py",
        "parts": imported,
    }, indent=2) + "\n")
    print(f"parts: wrote {len(imported)} to {out_path}")


if __name__ == "__main__":
    import_colors()
    import_parts()
