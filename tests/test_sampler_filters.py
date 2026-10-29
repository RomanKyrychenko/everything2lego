"""Target-colour part filter, flat-top cells, and pattern mining (Apriori growth)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/learn"))

from everything2lego.colors import palette
from everything2lego.sampler import only_colours, select_vocabulary, target_colours
from everything2lego.surface_mix import flat_cells


def test_target_colours_and_filter():
    col = np.full((2, 4, 4), -1)
    col[0] = 3; col[1, 0, 0] = 5                       # colour 5 is 1 of 17 cells
    show = np.ones_like(col, bool)
    assert target_colours(col, show, 0.1) == {palette()[3].code}
    assert target_colours(col, show, 0.01) == {palette()[3].code, palette()[5].code}
    v = select_vocabulary("building", 50)
    codes = frozenset({palette()[3].code})
    kept = only_colours(v, codes)
    assert kept and all(p.colours == codes for p in kept)
    assert len(kept) == sum(1 for p in v if codes & p.colours)


def test_flat_cells_box():
    occ = np.zeros((6, 12, 12), bool); occ[:4, 2:10, 2:10] = True
    f = flat_cells(occ)
    assert f[3, 5, 5] and f[2, 5, 5] and not f[1, 5, 5] and not f[:, 0, 0].any()


def test_mine_grows_mixed_groups():
    from motifs import mine, touching
    # a 1x2 plate with a 1x1 tile on each stud and a brick beside: in every set
    items = [("a", 0, (0, 0, 0), (1, 1, 2)), ("t", 0, (1, 0, 0), (1, 1, 1)), ("t", 0, (1, 0, 1), (1, 1, 1)),
             ("b", 0, (0, 1, 0), (3, 1, 1))]
    sets = [(items, touching(items), [True] * 4)] * 3
    found = mine(sets, 3, 4)
    assert found[2] and found[4] and all(c == 3 for c in found[4].values())
