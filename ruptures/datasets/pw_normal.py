"""2D piecewise Gaussian process (pw_normal)"""
from itertools import cycle

import numpy as np
from numpy import random as rd

from ruptures.utils import draw_bkps


def pw_normal(n_samples=200, n_bkps=3):
    """Return a 2D piecewise Gaussian signal and the associated changepoints.

    Args:
        n_samples (int, optional): signal length
        n_bkps (int, optional): number of change points

    Returns:
        tuple: signal of shape (n_samples, 2), list of breakpoints
    """
    # breakpoints
    bkps = draw_bkps(n_samples, n_bkps)
    # we create the signal
    signal = np.zeros((n_samples, 2), dtype=float)
    cov1 = np.array([[1, 0.9], [0.9, 1]])
    cov2 = np.array([[1, -0.9], [-0.9, 1]])
    for sub, cov in zip(np.split(signal, bkps), cycle((cov1, cov2))):
        n_sub, _ = sub.shape
        sub += rd.multivariate_normal([0, 0], cov, size=n_sub)

    return signal, bkps
