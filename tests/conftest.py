import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _seed_numpy():
    """Surface sampling (trimesh) draws from numpy's global RNG; seed it so
    tests that look for features on sampled colours are repeatable."""
    np.random.seed(1234)
