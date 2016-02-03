import pytest
from ruptures.datasets import pw_constant, pw_linear
from itertools import product
import numpy as np


@pytest.mark.parametrize('method, n_samples, dim, n_regimes, noisy, snr',
                         product([pw_constant, pw_linear],
                                 range(20, 1000, 200),
                                 range(1, 4),
                                 range(2, 10, 3),
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

# @pytest.mark.parametrize('n_samples, n_regimes, noisy, snr',
#                          product(range(20, 1000, 200),
#                                  range(2, 10, 3),
#                                  [True, False],
#                                  np.linspace(0, 1, 3)))
# def test_linear(n_samples, n_regimes, noisy, snr):
#     signal, bkps = pw_linear(n=n_samples,
#                              clusters=n_regimes,
#                              noisy=noisy,
#                              snr=snr)
#     assert signal.shape[0] == n_samples
#     assert len(bkps) == n_regimes


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
