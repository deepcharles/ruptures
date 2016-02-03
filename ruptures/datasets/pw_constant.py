import numpy as np
from ruptures.datasets import uniform_with_constant_sum


def pw_constant(n=100, clusters=3, dim=1, min_size=None, noisy=False, snr=0.1):
    """
    Piecewise constant signal.
    Returns the signal and the change point indexes (end of each regime).
    """
    # taille minimale de segment
    if min_size is None:
        min_size = int(n / clusters)
    assert min_size >= 2, "There must be at least two points per segment"
    assert isinstance(n, int)
    assert isinstance(clusters, int)
    assert isinstance(min_size, int)
    assert isinstance(noisy, bool)
    assert isinstance(dim, int)
    assert min_size * clusters <= n, "The minimum size is too great."
    # tailles de segments
    segments = uniform_with_constant_sum(clusters, n - min_size * clusters)
    segments += min_size

    assert clusters > 1, "There must be at least two regimes."

    signals = list()
    for _ in range(dim):
        # constantes
        constantes = list(range((int(clusters))))
        np.random.shuffle(constantes)

        # we create the signal
        signal1d = np.hstack(
            [c] * length for c,
            length in zip(
                constantes,
                segments))
        if noisy:
            std = snr * np.std(signal1d, dtype=float)
            signal1d = signal1d + \
                np.random.standard_normal(signal1d.size) * std

        signals.append(signal1d.reshape(-1, 1))

    res = np.hstack(signals)
    chg_pts = np.cumsum(segments)
    return res, chg_pts
