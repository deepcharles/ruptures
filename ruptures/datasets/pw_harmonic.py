import numpy as np
from ruptures.datasets import uniform_with_constant_sum
from math import ceil


def pw_harmonic(n=100, clusters=3, dim=1, min_size=None,
                min_wave_per_segment=5,
                min_points_per_wave=20,
                noisy=False, snr=0.1):
    """
    Piecewise constant signal.

    Args:
        n (int, optional): signal length
        clusters (int, optional): number of regimes
        dim (int, optional): dimension of the signal
        min_size (int or None, optional): minimum size of a regime. If None,
            automatically computed.
        min_wave_per_segment (int, optional): minimum number of periods per
            regime
        min_points_per_wave (int, optional): minimum number of points per
            period
        noisy (bool, optional): If True, noise is added
        snr (float, optional): signal-to-noise ratio (between 0 and 1)

    Returns:
        tuple: (list of changepoint indexes, signal)
    """
    # taille minimale de segment
    if min_size is None:
        min_size = ceil(n / clusters / 2)
    assert isinstance(n, int)
    assert isinstance(clusters, int)
    assert isinstance(min_size, int)
    assert isinstance(noisy, bool)
    assert isinstance(dim, int)
    assert clusters > 1, "There must be at least two regimes."
    assert min_size * clusters <= n, "The minimum size is too great."
    # tailles de segments
    segments = uniform_with_constant_sum(clusters, n - min_size * clusters)
    segments += max(min_size, min_points_per_wave * min_wave_per_segment)

    signals1D_list = list()  # list of 1D signals
    for _ in range(dim):

        last_previous_point = 0
        signal1d = list()
        for s in segments:
            # amplitude
            A = np.random.random()
            A *= (1 - abs(last_previous_point))
            A += abs(last_previous_point)
            # period
            nb_of_waves = np.random.randint(min_wave_per_segment,
                                            s // min_points_per_wave + 1)
            period = s // nb_of_waves
            # phase (to ensure continuity)
            phase = np.arcsin(last_previous_point / A)

            time = np.arange(s)
            x = A * np.sin(2 * np.pi * time / period + phase)
            last_previous_point = A * np.sin(2 * np.pi * s / period + phase)
            signal1d.append(x.reshape(-1, 1))

        # we concatenate the regimes
        signal1d = np.vstack(signal1d)

        if noisy:
            noise = snr * signal1d.std() * np.random.randn(signal1d.size)
            signal1d += noise.reshape(-1, 1)

        signals1D_list.append(signal1d)

    res = np.hstack(signals1D_list)
    chg_pts = np.cumsum(segments)
    return res, chg_pts
