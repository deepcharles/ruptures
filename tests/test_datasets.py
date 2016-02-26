import pytest
from ruptures.datasets import pw_constant, pw_linear, pw_harmonic
from itertools import product
import numpy as np


@pytest.mark.parametrize('method, n_samples, dim, n_regimes, noisy, snr',
                         product([pw_constant, pw_linear],
                                 range(20, 1000, 200),
                                 range(1, 4),
                                 range(2, 5, 3),
                                 [True, False],
                                 np.linspace(0, 1, 3)))
def test_constant(method, n_samples, dim, n_regimes, noisy, snr):
    signal, bkps = method(n=n_samples,
                          dim=dim,
                          clusters=n_regimes,
                          noisy=noisy,
                          snr=snr)
    assert signal.shape == (n_samples, dim)
    assert len(bkps) == n_regimes
    assert bkps[-1] == n_samples


@pytest.mark.parametrize('method', [pw_constant, pw_linear])
def test_exceptions(method):

    # should not raise any exception
    signal, bkps = method(n=10, clusters=5, noisy=False)
    signal, bkps = method(n=100, clusters=5, min_size=20)

    with pytest.raises(AssertionError):
        signal, bkps = method(n=9, clusters=5, noisy=False)
        signal, bkps = method(n=100, clusters=5, min_size=21)
        signal, bkps = method(n=100.0, clusters=5, min_size=21)
        signal, bkps = method(n=100, clusters=5, min_size=21, noisy=2)
        signal, bkps = method(n=100, clusters=5, min_size=21.0)
        signal, bkps = method(n=100, clusters=5.0)


@pytest.mark.parametrize('method', [pw_harmonic])
def test_exceptions_harmonic(method):

    # should not raise any exception
    signal, bkps = method(n=10, clusters=5, noisy=True,
                          min_wave_per_segment=1,
                          min_points_per_wave=1)

    signal, bkps = method(n=100, clusters=5, min_size=20,
                          min_wave_per_segment=1,
                          min_points_per_wave=1)

    with pytest.raises(AssertionError):
        signal, bkps = method(n=9, clusters=5, noisy=False)
        signal, bkps = method(n=100, clusters=5, min_size=21)
        signal, bkps = method(n=100.0, clusters=5, min_size=21)
        signal, bkps = method(n=100, clusters=5, min_size=21, noisy=2)
        signal, bkps = method(n=100, clusters=5, min_size=21.0)
        signal, bkps = method(n=100, clusters=5.0)
