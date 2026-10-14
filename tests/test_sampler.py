"""The statistics sampler fills a target with real parts in colours LEGO
makes them in, from any reference statistics."""
from collections import Counter

import numpy as np

from everything2lego import sampler


def _models():
    pats = Counter({("brick4x2h3", "brick4x2h3", "up"): 10, ("brick4x2h3", "brick4x2h3", "down"): 10,
                    ("brick2x1h3", "brick4x2h3", "level"): 4, ("plate4x2h1", "brick4x2h3", "up"): 3})
    return [{"model": "m", "patterns": pats, "shape": np.zeros(4), "held_out": False}]


def test_vocabulary_is_real_box_parts_with_colours():
    parts = sampler.vocabulary()
    assert {"3001.dat", "3023.dat", "3069b.dat"} <= {p.ldraw for p in parts}
    assert all(p.shape in ("brick", "plate", "tile") for p in parts)
    assert sampler.available(next(p for p in parts if p.ldraw == "3001.dat"), 4)   # red 2x4


def test_sample_fills_a_box():
    want = np.ones((6, 6, 4), dtype=bool)
    target = sampler.Target(want, np.full(want.shape, 4), want.copy())
    placements, info = sampler.sample(target, _models(), iters=300, seed=1, k_models=1, warmup=50)
    assert info["bad_cells"] == 0
    cells = sum(p.w_world * p.d_world * p.part.height_units for p in placements)
    assert cells == want.sum()
    vocab = {p.ldraw for p in sampler.vocabulary()}
    assert all(p.part.ldraw in vocab for p in placements)
