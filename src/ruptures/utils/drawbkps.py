r"""Draw a random partition."""

import numpy as np


def draw_bkps(n_samples=100, n_bkps=3, seed=None):
    """Draw a random partition with specified number of samples and specified
    number of changes."""
    rng = np.random.default_rng(seed=seed)
    alpha = np.ones(n_bkps + 1) / (n_bkps + 1) * 2000
    bkps = np.cumsum(rng.dirichlet(alpha) * n_samples).astype(int).tolist()
    bkps[-1] = n_samples
    return bkps
