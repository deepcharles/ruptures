"""Piecewise sinusoidal (pw_wavy)"""
from itertools import cycle

import numpy as np

from ruptures.utils import draw_bkps


def pw_wavy(n_samples=200, n_bkps=3, noise_std=None, seed=None):
    """Return a 1D piecewise wavy signal and the associated changepoints.

    Args:
        n_samples (int, optional): signal length
        n_bkps (int, optional): number of changepoints
        noise_std (float, optional): noise std. If None, no noise is added
        seed (int): random seed

    Returns:
        tuple: signal of shape (n_samples, 1), list of breakpoints
    """
    # breakpoints
    bkps = draw_bkps(n_samples, n_bkps, seed=seed)
    # we create the signal
    f1 = np.array([0.075, 0.1])
    f2 = np.array([0.1, 0.125])
    freqs = np.zeros((n_samples, 2))
    for sub, val in zip(np.split(freqs, bkps[:-1]), cycle([f1, f2])):
        sub += val
    tt = np.arange(n_samples)

    # DeprecationWarning: Calling np.sum(generator) is deprecated
    # Use np.sum(np.from_iter(generator)) or the python sum builtin instead.
    signal = np.sum([np.sin(2 * np.pi * tt * f) for f in freqs.T], axis=0)

    if noise_std is not None:
        rng = np.random.default_rng(seed=seed)
        noise = rng.normal(scale=noise_std, size=signal.shape)
        signal += noise

    return signal, bkps
