import numpy as np
from ruptures.datasets import uniform_with_constant_sum


def pw_constant(n=100, clusters=3, min_size=None, noisy=False, snr=0.1):
    """
    Piecewise constant signal.
    Returns the signal and the change point indexes (end of each regime).
    """
    # taille minimale de segment
    if min_size is None:
        min_size = int(n / clusters / 2)
    assert min_size > 1
    assert min_size * clusters <= n
    # tailles de segments
    segments = uniform_with_constant_sum(clusters, n - min_size * clusters)
    segments += min_size

    # constantes
    assert clusters > 1
    constantes = list(range((int(clusters))))
    np.random.shuffle(constantes)

    # we create the signal
    res = np.hstack([c] * length for c, length in zip(constantes, segments))
    chg_pts = np.cumsum(segments)
    if noisy:
        std = snr * np.std(res, dtype=float)
        res = res + np.random.standard_normal(res.size) * std

    return res, chg_pts
