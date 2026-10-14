"""The statistics sampler fills a target with real parts, in yaws that fit
it, in colours LEGO makes them in."""
from collections import Counter

import numpy as np

from everything2lego import sampler
from everything2lego.io_ldraw import placement_to_line


def _models():
    pats = Counter({("brick4x2h3", "brick4x2h3", "up"): 10, ("brick4x2h3", "brick4x2h3", "down"): 10,
                    ("brick2x1h3", "brick4x2h3", "level"): 4, ("slope45-2x2h3", "brick4x2h3", "down"): 4,
                    ("brick4x2h3", "slope45-2x2h3", "up"): 4})
    return [{"model": "m", "patterns": pats, "shape": np.zeros(4), "held_out": False}]


def _vp(ldraw):
    return next(v for v in sampler.vocabulary() if v.part.ldraw == ldraw)


def test_vocabulary_has_measured_shapes_and_connections():
    brick, slope = _vp("3001.dat"), _vp("3039.dat")
    assert brick.box == "brick" and sampler.select_vocabulary(None)
    o = brick.orients[0]
    assert o.size == (3, 4, 2) and np.allclose(o.o, 1) and len(o.top) == len(o.bottom) == 8
    assert len(slope.orients) == 4                       # a slope faces four ways
    assert {len(o.top) for o in slope.orients} == {2} and {len(o.bottom) for o in slope.orients} == {4}
    assert 0.7 < float(slope.orients[0].o.mean()) < 0.85
    # yaws are the same shape turned: equal sample counts, different masks
    assert len({int(sampler.popcount(o.bits).sum()) for o in slope.orients}) == 1
    assert len({o.bits.tobytes() for o in slope.orients}) == 4


def test_masks_round_trip():
    rnd = np.random.default_rng(0)
    fine = rnd.random((8, 12, 4)) > 0.5
    assert (sampler.unpack(sampler.pack(fine, (2, 3, 1))) == fine).all()


def test_a_slope_is_written_where_its_box_is():
    """The LDraw origin of 3039 is off its footprint's centre: placed at a
    box, it must land in that box in every yaw."""
    from everything2lego.pack import Placement
    slope = _vp("3039.dat")
    for o in slope.orients:
        p = Placement(slope.part, 0, 0, 0, 0, o.size[1], o.size[2], o.rot)
        x, y, z = map(float, placement_to_line(p).split()[2:5])
        # the stud row (top) is in the box: origin + R * (0, 0, -10 .. 10) within x 0..40, z 0..40 LDU (- 10 shift)
        assert -10 <= x <= 30 and -10 <= z <= 30 and y == -24


def test_sample_fits_a_ramp_with_slopes():
    """A ramp shaped as a 45° slope's own cells, twice side by side, is
    built of slopes facing the way it falls — not of stepped boxes."""
    o = next(o for o in _vp("3039.dat").orients if o.rot == 90)
    one = np.zeros(o.size, dtype=np.uint64)
    one[tuple(o.cells.T)] = o.bits
    bits = np.concatenate([one, one], axis=2)
    want = bits != 0
    target = sampler.Target(want, np.full(want.shape, 4), want.copy(), bits=bits)
    placements, info = sampler.sample(target, _models(), iters=300, seed=1, k_models=1, warmup=50,
                                      n_special=20)
    assert info["unsupported"] == 0 and info["fidelity"] < 1.0
    assert any(p.part.shape.startswith("slope") and p.rotation_deg == 90 for p in placements)


def test_sample_fills_a_box():
    want = np.ones((6, 6, 4), dtype=bool)
    target = sampler.Target(want, np.full(want.shape, 4), want.copy())
    placements, info = sampler.sample(target, _models(), iters=300, seed=1, k_models=1, warmup=50,
                                      n_special=0)
    assert info["bad_cells"] == 0 and info["unsupported"] == 0
