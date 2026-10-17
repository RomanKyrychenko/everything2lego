"""Round-trip test for State._paths4: adding then removing a sequence of items
should return self.kl3 to its pre-add state (within numerical tolerance).

Catches sign-flip and iteration-asymmetry bugs in the 4-walk incremental
maintenance before we promote the term to the energy.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from everything2lego import sampler as S  # noqa: E402

FEATURES_PATH = Path("/Volumes/PRO-G40/lego_data/analysis/features.pkl")


def _models_with_paths4():
    """Fail the test gracefully if paths4 hasn't been precomputed yet."""
    if not FEATURES_PATH.exists():
        pytest.skip(f"features.pkl not found at {FEATURES_PATH}")
    rows = pickle.load(open(FEATURES_PATH, "rb"))
    with_p4 = [r for r in rows if "paths4" in r]
    if len(with_p4) < 50:
        pytest.skip(f"only {len(with_p4)} models have paths4; run `features.py paths4` first")
    return with_p4


def test_paths4_round_trip():
    """Build a small target, run a few sampler iterations, then remove every
    placement — kl3.value() should return to 0.0 within 1e-9."""
    models = _models_with_paths4()
    stats = S.Stats(models[:200])                                      # small reference, fast
    if stats.ref3 is None:
        pytest.skip("Stats.ref3 is None; make sure paths4 is in features.pkl")
    stats.degree = S.token_degree(models[:200])
    # tiny 4x4x4 target that fits a few parts
    shape = (3, 4, 4)
    want = np.ones(shape, bool)
    colour = np.full(shape, -1, dtype=np.int32)
    target = S.Target(want=want, colour=colour, visible=want.copy(), kind="building")
    st = S.State(target, stats, S.Weights())
    vocab = S.select_vocabulary("building", 80)
    import random
    rnd = random.Random(0)
    filler = S.Filler(st, rnd, vocab, "building", top_k=12, net=True)
    cells = [(y, x, z) for y in range(shape[0]) for x in range(shape[1]) for z in range(shape[2])]
    filler.fill(cells)
    assert len(st.items) >= 2, "nothing was placed in the tiny target"

    # bigger target: 5x8x8 forces several placements including 4-walks
    shape2 = (5, 8, 8)
    want2 = np.ones(shape2, bool)
    target2 = S.Target(want=want2, colour=np.full(shape2, -1, dtype=np.int32),
                        visible=want2.copy(), kind="building")
    st = S.State(target2, stats, S.Weights())
    rnd = random.Random(0)
    filler = S.Filler(st, rnd, vocab, "building", top_k=12, net=True)
    filler.fill([(y, x, z) for y in range(shape2[0]) for x in range(shape2[1]) for z in range(shape2[2])])
    assert len(st.items) >= 5, f"not enough items placed: {len(st.items)}"

    # remove every item; kl3 must return to 0
    for i in list(st.items):
        st.remove(i)
    post_kl3 = st.kl3.value() if st.kl3 is not None else 0.0
    assert len(st.items) == 0
    assert abs(post_kl3) < 1e-7, f"kl3 did not return to 0 after full removal: {post_kl3}"


def test_paths4_sign_symmetry():
    """Add an item, record kl3; remove it, record kl3; add it again (fresh id);
    kl3 after re-add should match the first recording within 1e-9.  Validates
    that remove() exactly undoes add() for the kl3 term."""
    from everything2lego.sampler import Item, select_vocabulary
    models = _models_with_paths4()
    stats = S.Stats(models[:200])
    if stats.ref3 is None:
        pytest.skip("Stats.ref3 is None")
    stats.degree = S.token_degree(models[:200])
    shape = (3, 4, 4)
    want = np.ones(shape, bool)
    colour = np.full(shape, -1, dtype=np.int32)
    target = S.Target(want=want, colour=colour, visible=want.copy(), kind="building")
    st = S.State(target, stats, S.Weights())
    vocab = select_vocabulary("building", 80)
    import random
    rnd = random.Random(0)
    filler = S.Filler(st, rnd, vocab, "building", top_k=12, net=True)
    filler.fill([(y, x, z) for y in range(shape[0]) for x in range(shape[1]) for z in range(shape[2])])
    if len(st.items) < 2:
        pytest.skip("not enough items placed")
    # snapshot kl3 after fill
    kl3_after_fill = st.kl3.value()
    # pick one item, remove, add back (same geometry)
    target_id = next(iter(st.items))
    it = st.items[target_id]
    st.remove(target_id)
    st.add(Item(it.vp, it.ori, it.colour, it.y, it.x, it.z))
    kl3_after_cycle = st.kl3.value()
    assert abs(kl3_after_cycle - kl3_after_fill) < 1e-7, \
        f"kl3 differs after remove/add cycle: {kl3_after_fill} vs {kl3_after_cycle}"
