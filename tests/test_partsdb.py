"""The parts database's connector extraction (tools/partsdb), on parts whose
connections are known.  Needs the LDraw and LDCad shadow libraries; skipped
where they are not installed."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "partsdb"))
sys.path.insert(0, str(ROOT / "tools" / "set_study"))

connectors = pytest.importorskip("connectors")
if not (connectors.SHADOW / "p" / "stud.dat").exists() or not (connectors.LDRAW / "parts" / "3001.dat").exists():
    pytest.skip("LDraw / LDCad shadow libraries not installed", allow_module_level=True)

from ldraw_read import part_bbox  # noqa: E402


def counts(name):
    b = part_bbox(name)
    return connectors.summary(connectors.part_connectors(name, height=float(b[1][1])))


@pytest.mark.parametrize("part, expected", [
    ("3001.dat", {"stud_up": 8, "antistud_down": 8}),          # brick 2x4
    ("3023.dat", {"stud_up": 2, "antistud_down": 2}),          # plate 1x2 (sockets implicit)
    ("3069b.dat", {"antistud_down": 2}),                       # tile: no studs
    ("99780.dat", {"stud_up": 2, "stud_side": 2}),             # bracket: SNOT
    ("22885.dat", {"stud_up": 2, "stud_side": 4}),             # brick with 4 side studs
])
def test_known_parts_have_their_connectors(part, expected):
    got = counts(part)
    for k, v in expected.items():
        assert got.get(k, 0) == v, (part, got)
