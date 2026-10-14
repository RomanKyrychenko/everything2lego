"""LEGO part catalog loader + geometry constants."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files


# LDraw unit conversions.
LDU_PER_STUD = 20      # 1 stud footprint = 20 LDU
LDU_PER_PLATE = 8      # 1 plate height = 8 LDU (brick = 3 plates = 24 LDU)

# Real millimeter equivalents (for mesh scaling).
MM_PER_STUD = 8.0
MM_PER_PLATE = 3.2


@dataclass(frozen=True)
class Part:
    ldraw: str             # e.g. "3001.dat"
    bl: str                # Bricklink part id
    shape: str             # brick | plate | tile | slope | slope_inv
    footprint: tuple[int, int]   # (x_studs, z_studs) with x >= z conventionally
    height_units: int      # in plate-heights (brick=3, plate=1)
    name: str
    slope_dir: str | None = None  # for slopes: default low direction ("+x", "-x", "+z", "-z")
    plain: bool = True     # False: a decorative part (round, arch, side studs) that
                           # a styling pass may swap in by id but no packer may pick
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
                           # LDU, the part's own frame: where its footprint's centre
                           # (x, z) and its body's top (y) are from its LDraw origin
    yaw: int = 0           # the quarter turn from this catalogue's convention (footprint w
                           # along x, low side at slope_dir) to how LDraw draws the part

    @property
    def w(self) -> int:
        return self.footprint[0]

    @property
    def d(self) -> int:
        return self.footprint[1]

    @property
    def is_slope(self) -> bool:
        return self.shape in ("slope", "slope_inv")

    def orientations(self) -> list[tuple[int, int]]:
        """Distinct (w,d) footprints under 90-degree rotation.

        Rectangular bricks look identical after a 180-degree rotation, so we
        only enumerate two orientations. Slopes are asymmetric — the direction
        the low side points depends on rotation — but their *footprint* still
        only takes two values, and the refinement pass chooses rotation
        directly, so this method still returns two entries for them.
        """
        w, d = self.footprint
        if w == d:
            return [(w, d)]
        return [(w, d), (d, w)]


@lru_cache(maxsize=1)
def _frames() -> dict[str, dict]:
    """Each catalogue part's measured LDraw frame (data/part_frames.json;
    tools/partsdb/export.py frames, which reads the catalogue itself, so a
    missing file means no corrections)."""
    f = files("everything2lego.data").joinpath("part_frames.json")
    return json.loads(f.read_text())["parts"] if f.is_file() else {}


def _make_part(p: dict) -> Part:
    return Part(
        ldraw=p["ldraw"],
        bl=p["bl"],
        shape=p["shape"],
        footprint=tuple(p["footprint"]),
        height_units=p["height_units"],
        name=p["name"],
        slope_dir=p.get("slope_dir"),
        plain=p.get("plain", True),
        origin=tuple(_frames().get(p["ldraw"], {}).get("origin", (0.0, 0.0, 0.0))),
        yaw=_frames().get(p["ldraw"], {}).get("yaw", 0),
    )


# Rebrickable names of parts whose geometry really is the plain box, wedge,
# frame or door the packer, the stability check and the viewer assume.
# Anything else (arches, round and curved bricks, brackets, clips, wedge
# plates, double slopes, window glass, shutters...) shares a shape label
# with these but not a shape, so it is kept out of the catalog.
_DIMS = r"\d+ x \d+"
_DIMS_H = r"\d+ x \d+(?: x \d+(?:/\d+)?)?"
_PLAIN_NAMES: dict[str, re.Pattern[str]] = {
    "brick": re.compile(rf"^Brick {_DIMS_H}(?: without (?:Bottom Tubes?|Inside Ridges|Side Supports))?$"),
    "plate": re.compile(rf"^Plate {_DIMS}$"),
    "tile": re.compile(rf"^Tile {_DIMS}(?: with Groove)?$"),
    "panel": re.compile(rf"^Panel {_DIMS_H}(?: - Solid Studs| \[(?:Hollow|Solid) Studs\])?$"),
    "slope": re.compile(rf"^Brick Sloped \d+° {_DIMS_H}(?: \(Cheese Slope\))?$"),
    "slope_inv": re.compile(rf"^Brick Sloped Inverted \d+° {_DIMS_H}$"),
    "window": re.compile(rf"^Window {_DIMS_H}(?: Frame(?: with Solid Studs)?)?$"),
    "door": re.compile(rf"^Door(?: Frame)? {_DIMS_H}(?: Left| Right)?$"),
}


_NAME_HEIGHT = re.compile(r"\d+ x \d+ x (\d+)(?:/(\d+))?")
_BASE_HEIGHT = {"plate": 1, "tile": 1, "brick": 3, "slope": 3, "slope_inv": 3,
                "panel": 3, "window": 6, "door": 15}


def height_from_name(shape: str, name: str) -> int | None:
    """Height in plates that a Rebrickable name states ("1 x 2 x 5" is five
    bricks, "1 x 2 x 2/3" two thirds of one), or the shape's default."""
    m = _NAME_HEIGHT.search(name)
    if not m:
        return _BASE_HEIGHT.get(shape)
    num, den = int(m.group(1)), int(m.group(2) or 1)
    return max(1, round(3 * num / den))


def is_plain_geometry(shape: str, name: str) -> bool:
    """Whether a Rebrickable part name describes exactly the geometry that
    `shape` stands for, so the part can be used as a drop-in of that shape."""
    pattern = _PLAIN_NAMES.get(shape)
    return bool(pattern and pattern.match(name))


@lru_cache(maxsize=1)
def catalog() -> tuple[Part, ...]:
    """Every part with plain box, wedge, frame or door geometry: the pool the
    packers, builders and checks draw from.  `full_catalog` adds the
    decorative parts that styling passes substitute by id."""
    return tuple(p for p in full_catalog() if p.plain)


@lru_cache(maxsize=1)
def full_catalog() -> tuple[Part, ...]:
    """The union of the curated catalog and any Rebrickable-imported parts.

    Dedupe key is the LDraw filename so a curated entry always wins over
    an imported one (curated entries have hand-tuned `slope_dir` etc.).
    Imported parts are kept only when their name describes plain geometry
    (`is_plain_geometry`): the import is by category and category alone
    lets arches, brackets, wedge plates and window glass through.
    """
    data_pkg = files("everything2lego.data")
    raw = json.loads(data_pkg.joinpath("brick_catalog.json").read_text())
    parts: list[Part] = [_make_part(p) for p in raw["parts"]]
    seen = {p.ldraw for p in parts}
    imported_path = data_pkg.joinpath("imported_parts.json")
    try:
        imported = json.loads(imported_path.read_text())
    except FileNotFoundError:
        imported = None
    if imported:
        for p in imported["parts"]:
            if p["ldraw"] in seen or not is_plain_geometry(p["shape"], p["name"]):
                continue
            # Older imports stored every brick as 3 plates whatever its name
            # said.  The packer places a "brick" as a 3-plate box, so a taller
            # one belongs with the tall parts, which are placed by true height.
            height = height_from_name(p["shape"], p["name"])
            shape = "brick_tall" if p["shape"] == "brick" and height > 3 else p["shape"]
            parts.append(_make_part({**p, "shape": shape, "height_units": height}))
            seen.add(p["ldraw"])
    return tuple(parts)


@lru_cache(maxsize=None)
def parts_by_shape(shape: str) -> tuple[Part, ...]:
    return tuple(p for p in catalog() if p.shape == shape)


@lru_cache(maxsize=None)
def parts_by_height(height_units: int, *, exclude_tiles: bool = True) -> tuple[Part, ...]:
    return tuple(
        p for p in catalog()
        if p.height_units == height_units and (not exclude_tiles or p.shape != "tile")
    )


@lru_cache(maxsize=1)
def _production() -> dict[str, frozenset[int]]:
    raw = json.loads(files("everything2lego.data").joinpath("part_colours.json").read_text())
    return {k: frozenset(int(c) for c in v) for k, v in raw["parts"].items()}


def made_in(part: "Part | str", code: int) -> bool:
    """Whether LEGO makes `part` in LDraw colour `code`: it came in at least
    two sets since 2010 (data/part_colours.json, from Rebrickable's
    inventories, by mould; tools/partsdb/export.py).  A catalogue part with no
    production record is made in nothing; a part outside the table (a
    decoration a styling pass names by id) is not judged."""
    cols = _production().get(part if isinstance(part, str) else part.ldraw)
    return cols is None or code in cols


# The glass each window frame takes, placed with the frame's own transform
# (LDraw draws frame and glass about the same origin).
GLASS: dict[str, str] = {"60592.dat": "60601.dat", "60593.dat": "60602.dat",
                         "60594.dat": "60603.dat"}
# ...except where the official models offset it, in the frame's own LDU
# (60603 is an opening pane: 8 down and 4 in, as the OMR's sets place it).
GLASS_OFFSET: dict[str, tuple[int, int, int]] = {"60594.dat": (0, 8, 4)}


# The parts every colour of a set's palette must come in: a wall, a floor,
# a sculpted course can always be finished in 1x1 and 1x2 bricks and plates.
CORE_PARTS = ("3005.dat", "3004.dat", "3024.dat", "3023.dat")


def largest_first(parts: tuple[Part, ...]) -> tuple[Part, ...]:
    """Sort parts by descending stud area, so a greedy fill picks the biggest first."""
    return tuple(sorted(parts, key=lambda p: (-p.w * p.d, -max(p.w, p.d))))
