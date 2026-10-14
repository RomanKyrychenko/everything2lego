"""The proposal network's context and output (everything2lego.proposals)."""
import numpy as np

from everything2lego import proposals


def test_context_sees_only_cells_built_before():
    target = np.ones((6, 8, 8))
    placed = np.full(target.shape, 2)                  # everything built, kind 2
    static = proposals.static_context(target)
    f = proposals.features(target, placed, static, 2, 3, 3, object_index=1, n_kinds=4, n_objects=3)
    n = proposals.NWIN
    assert f.shape == (proposals.n_inputs(4, 3),)
    assert f[:n].sum() == n                            # the whole window is object
    assert f[n:2 * n].sum() == proposals.BEFORE.sum()  # only cells before (2, 3, 3) count as built
    assert np.isclose(f[2 * n + 2], proposals.BEFORE.sum() / n) and f[2 * n:2 * n + 5].sum() == f[2 * n + 2]
    assert f[2 * n + 5:2 * n + 8].tolist() == [0, 1, 0]   # object kind one-hot
    assert np.allclose(f[-proposals.NSTATIC:], static[2, 3, 3])


def test_window_off_the_grid_is_empty():
    target = np.ones((2, 2, 2))
    f = proposals.features(target, np.full(target.shape, -1), proposals.static_context(target), 0, 0, 0, -1, 2, 2)
    assert f[:proposals.NWIN].sum() == 8 and f[proposals.NWIN:-proposals.NSTATIC].sum() == 0


def test_static_context_blocks_and_depth():
    fill = np.zeros((9, 12, 12))
    fill[:, 4:8, 4:8] = 1.0                            # a 4 x 4 column
    st = proposals.static_context(fill)
    centre = proposals.BLOCKS.index((0, -2, -2))
    assert np.isclose(st[3, 5, 5, centre], 27 / 48)    # block x, z 3..6 meets the column in 3 x 3 cells, 3 plates
    assert st[0, 5, 5, -2] == 0 and st[8, 5, 5, -2] == 1
    assert st[4, 0, 0, -1] == 0 and st[4, 5, 5, -1] > 0


def test_trained_network_gives_probabilities():
    net = proposals.load()
    if net is None:
        return
    x = np.zeros((3, net.W1.shape[0]), np.float32)
    lp = net.logprobs(x)
    assert lp.shape == (3, len(net.tokens) + 1) and np.allclose(np.exp(lp).sum(1), 1, atol=1e-4)
