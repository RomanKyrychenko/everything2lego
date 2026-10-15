"""The instruction generator and the parts list run on sampler builds."""
import numpy as np

from everything2lego import instructions, io_bricklink, sampler


def _models():
    from collections import Counter
    pats = Counter({("brick4x2h3", "brick4x2h3", "up"): 10, ("brick4x2h3", "brick4x2h3", "down"): 10})
    shape = np.array([np.log(4), np.log(4), np.log(2), 0.95])
    return [{"model": "m", "patterns": pats, "paths": pats, "shape": shape, "held_out": False,
             "tokens": Counter({"brick4x2h3": 10}), "n_parts": 20}]


def test_sampler_build_gets_instructions_and_parts_list(tmp_path):
    """A 4x4 target built with the sampler yields a parts list and the
    instruction planner produces at least one step."""
    want = np.ones((3, 4, 4), bool)
    target = sampler.Target(want, np.full(want.shape, 4), want.copy())
    ps, _ = sampler.sample(target, _models(), iters=100, seed=0, k_models=1, warmup=20, n_special=0, net=False)
    assert ps
    xml = io_bricklink.write_wanted_list(ps, tmp_path / "w.xml").read_text()
    assert "<ITEMID>" in xml
    steps = instructions.plan_steps(ps)
    assert steps and len(steps) >= 1
    assert sum(len(s) for s in steps) == len(ps)
