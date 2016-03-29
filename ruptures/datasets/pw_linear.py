import numpy as np
from ruptures.datasets import uniform_with_constant_sum
from math import ceil


def pw_linear(n=100, clusters=3, dim=1, min_size=None, noisy=False, snr=0.1):
    """
    Piecewise constant signal.
    Returns the signal and the change point indexes (end of each regime).
    """
# taille minimale de segment
    if min_size is None:
        min_size = ceil(n / clusters / 2)
    assert isinstance(noisy, bool)
    assert min_size * clusters <= n, "The minimum size is too great."
    # segment sizes
    segments = uniform_with_constant_sum(clusters, n - min_size * clusters)
    segments += min_size
    assert all(k >= min_size for k in segments)
    assert clusters > 1, "There must be at least two regimes."

    signals = list()
    for _ in range(dim):
        # slopes
        slopes = np.arange(clusters) - clusters / 2
        # we square the slopes to enhance the differences
        slopes *= np.abs(slopes)
        np.random.shuffle(slopes)

        # we create the signal
        signal1d = np.array([], dtype=float)
        intercept = 0
        for s, length in zip(slopes, segments):
            xx = intercept + np.arange(length) * s
            signal1d = np.append(signal1d, xx)
            intercept = xx[-1]

        # additive noise
        if noisy:
            std = snr * np.std(signal1d, dtype=float)
            signal1d = signal1d + \
                np.random.standard_normal(signal1d.size) * std

        signals.append(signal1d.reshape(-1, 1))

    res = np.hstack(signals)
    chg_pts = np.cumsum(segments)
    return res, chg_pts
