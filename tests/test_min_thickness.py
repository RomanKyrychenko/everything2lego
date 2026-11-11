import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from sampler_demo import min_thickness  # noqa: E402


def test_sheet_thickened_block_kept():
    s = np.zeros((40, 40, 40), bool)
    s[5:15, 5:15, 5:15] = True                  # thick block: unchanged
    s[20:35, 20, 5:30] = True                   # one-sample sheet: 3 samples thick, ends grow one sample
    out = min_thickness(s)
    assert (out[0:18] == s[0:18]).all()
    assert out[20:35, 19:22, 5:30].all() and out[19:36, 19:22, 4:31].sum() == out[18:].sum()
