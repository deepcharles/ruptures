import numpy as np
from ruptures.datasets import uniform_with_constant_sum


def pw_linear(n=100, clusters=3,  min_size=None, noisy=False, snr=0.1):
    """
    Piecewise constant signal.
    Returns the signal and the change point indexes (start of each regime).
    """
    # taille minimale de segment
    if min_size is None:
        min_size = int(n / clusters / 2)
    assert min_size > 1
    assert min_size * clusters <= n

    # segment sizes
    segments = uniform_with_constant_sum(clusters, n - min_size * clusters)
    segments += min_size

    # slopes
    assert clusters > 1
    slopes = np.arange(clusters) - clusters / 2
    np.random.shuffle(slopes)

    # we create the signal
    res = np.array([], dtype=float)
    intercept = 0
    for s, length in zip(slopes, segments):
        xx = intercept + np.arange(length) * s
        res = np.append(res, xx)
        intercept = xx[-1]

    chg_pts = np.append([0], np.cumsum(segments)[:-1])

    # additive noise
    if noisy:
        std = snr * np.std(res, dtype=float)
        res = res + np.random.standard_normal(res.size) * std

    return res, chg_pts
