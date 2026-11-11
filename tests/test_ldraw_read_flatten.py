"""flatten returns real parts only: no primitives, subparts or ~segments; embedded parts stay whole."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "set_study"))
from ldraw_read import LDRAW_OFFICIAL, _is_fragment, flatten  # noqa: E402

pytestmark = pytest.mark.skipif(not (LDRAW_OFFICIAL / "parts" / "3001.dat").exists(), reason="needs the LDraw library")

MPD = """0 FILE main.ldr
1 4 0 0 0 1 0 0 0 1 0 0 0 1 3001.dat
1 16 0 -24 0 1 0 0 0 1 0 0 0 1 string.ldr
1 1 40 0 0 1 0 0 0 1 0 0 0 1 mypart.dat
1 2 80 0 0 1 0 0 0 1 0 0 0 1 8273 - ls60.dat
1 3 0 0 40 1 0 0 0 1 0 0 0 1 42131 - 61332.dat
0 NOFILE
0 FILE string.ldr
0 !LDRAW_ORG Model
1 16 0 0 0 1 0 0 0 1 0 0 0 1 4-4cyli.dat
1 16 0 4 0 1 0 0 0 1 0 0 0 1 77.dat
1 16 0 8 0 1 0 0 0 1 0 0 0 1 3023.dat
0 NOFILE
0 FILE 8273 - LS60.dat
0 LS60
0 !LDRAW_ORG Unofficial_Part
1 16 0 0 0 1 0 0 0 1 0 0 0 1 4-4cyli.dat
0 NOFILE
0 FILE 42131 - 61332.dat
0 61332
0 !LDRAW_ORG Unofficial_Part
1 16 0 0 0 1 0 0 0 1 0 0 0 1 4-4cyli.dat
0 NOFILE
0 FILE mypart.dat
0 !LDRAW_ORG Unofficial_Part
1 16 0 0 0 1 0 0 0 1 0 0 0 1 s/3001s01.dat
1 16 0 0 0 1 0 0 0 1 4-4cyli.dat
0 NOFILE
"""


def test_flatten_keeps_parts_only(tmp_path):
    f = tmp_path / "m.mpd"; f.write_text(MPD)
    assert [(p.name, p.colour) for p in flatten(f)] == [("3001.dat", 4), ("3023.dat", 16), ("mypart.dat", 1), ("61332.dat", 3)]


def test_is_fragment():
    assert not _is_fragment("3001.dat") and not _is_fragment("3820.dat")      # brick, minifig hand
    assert _is_fragment("4-4cyli.dat") and _is_fragment("s/3001s01.dat")      # primitive, subpart
    assert _is_fragment("77.dat") and _is_fragment("166.dat")                 # hose segment, moved-to a tube segment
