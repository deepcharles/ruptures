"""Simulates piecewise constant function of given length and dimension."""

import numpy as np
from numpy import random as rd

from ruptures.utils import draw_bkps


def pw_constant(n_samples, n_features=1, n_bkps=3, noisy=False, sigma=1.,
                delta=(1, 10)):
    """Return a piecewise constant signal.

    Each regime length is drawn at random. Each mean shift amplitude is drawn uniformly from the
    interval delta.

    Args:
        n_samples (int): signal length
        n_features (int, optional): number of dimensions
        n_bkps (int, optional): number of changepoints
        noisy (bool, optional): If True, noise is added
        sigma (float, optional): noise std
        delta (tuple, optional): (delta_min, delta_max) max and min jump values

    Returns:
        tuple: signal1d of shape (n_samples, n_features), list of breakpoints

    """
    # breakpoints
    bkps = draw_bkps(n_samples, n_bkps)
    # we create the signal
    signal = np.empty((n_samples, n_features), dtype=float)
    tt_ = np.arange(n_samples, dtype=np.int8)
    delta_min, delta_max = delta
    # mean value
    center = np.zeros(n_features)
    for ind in np.split(tt_, bkps):
        # jump value
        jump = rd.uniform(delta_min, delta_max, size=n_features)
        spin = rd.choice([-1, 1], n_features)
        center += jump * spin
        signal[ind] = center

    if noisy:
        noise = rd.normal(size=signal.shape) * sigma
        signal = signal + noise

    return signal, bkps
