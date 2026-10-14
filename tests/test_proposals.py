"""The proposal network's context and output (everything2lego.proposals)."""
import numpy as np

from everything2lego import proposals


def test_context_sees_only_cells_built_before():
    target = np.ones((6, 8, 8), bool)
    placed = np.full(target.shape, 2)                  # everything built, kind 2
    f = proposals.features(target, placed, 2, 3, 3, object_index=1, n_kinds=4, n_objects=3)
    n = proposals.NWIN
    assert f.shape == (proposals.n_inputs(4, 3),)
    assert f[:n].sum() == n                            # the whole window is object
    assert f[n:2 * n].sum() == proposals.BEFORE.sum()  # only cells before (2, 3, 3) count as built
    assert np.isclose(f[2 * n + 2], proposals.BEFORE.sum() / n) and f[2 * n:2 * n + 5].sum() == f[2 * n + 2]
    assert f[-3:].tolist() == [0, 1, 0]                # object kind one-hot


def test_window_off_the_grid_is_empty():
    target = np.ones((2, 2, 2), bool)
    f = proposals.features(target, np.full(target.shape, -1), 0, 0, 0, -1, 2, 2)
    assert f[:proposals.NWIN].sum() == 8 and f[proposals.NWIN:].sum() == 0


def test_trained_network_gives_probabilities():
    net = proposals.load()
    if net is None:
        return
    x = np.zeros((3, net.W1.shape[0]), np.float32)
    lp = net.logprobs(x)
    assert lp.shape == (3, len(net.tokens) + 1) and np.allclose(np.exp(lp).sum(1), 1, atol=1e-4)
