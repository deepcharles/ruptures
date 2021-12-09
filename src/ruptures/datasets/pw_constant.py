"""Piecewise constant signal (with noise)"""

import numpy as np

from ruptures.utils import draw_bkps


def pw_constant(
    n_samples=200, n_features=1, n_bkps=3, noise_std=None, delta=(1, 10), seed=None
):
    """Return a piecewise constant signal and the associated changepoints.

    Args:
        n_samples (int): signal length
        n_features (int, optional): number of dimensions
        n_bkps (int, optional): number of changepoints
        noise_std (float, optional): noise std. If None, no noise is added
        delta (tuple, optional): (delta_min, delta_max) max and min jump values
        seed (int): random seed

    Returns:
        tuple: signal of shape (n_samples, n_features), list of breakpoints
    """
    # breakpoints
    bkps = draw_bkps(n_samples, n_bkps, seed=seed)
    # we create the signal
    signal = np.empty((n_samples, n_features), dtype=float)
    tt_ = np.arange(n_samples)
    delta_min, delta_max = delta
    # mean value
    center = np.zeros(n_features)
    rng = np.random.default_rng(seed=seed)
    for ind in np.split(tt_, bkps):
        if ind.size > 0:
            # jump value
            jump = rng.uniform(delta_min, delta_max, size=n_features)
            spin = rng.choice([-1, 1], n_features)
            center += jump * spin
            signal[ind] = center

    if noise_std is not None:
        noise = rng.normal(size=signal.shape) * noise_std
        signal = signal + noise

    return signal, bkps
