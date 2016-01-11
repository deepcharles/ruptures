from ruptures.costs import gaussmean, linear_mse, NotEnoughPoints
from ruptures.datasets import pw_constant
import numpy as np
from nose.tools import raises


@raises(NotEnoughPoints)
def test1_gaussmean():
    n_regimes = 2
    n_samples = 100

    # Piecewise constant signal
    signal, chg_pts = pw_constant(n=n_samples, clusters=n_regimes,
                                  min_size=50, noisy=True, snr=0.1)

    func_to_minimize = gaussmean(signal)
    func_to_minimize(0, 1)


@raises(NotEnoughPoints)
def test2_gaussmean():
    n_regimes = 2
    n_samples = 100

    # Piecewise constant signal
    signal, chg_pts = pw_constant(n=n_samples, clusters=n_regimes,
                                  min_size=50, noisy=True, snr=0.1)

    func_to_minimize = gaussmean(signal)
    func_to_minimize(4, 4)


def test3_gaussmean():
    n_samples = 100
    func = gaussmean(np.ones(n_samples))
    assert func(0, n_samples - 1) == np.inf


@raises(NotEnoughPoints)
def test1_linear_mse():
    n_regimes = 2
    n_samples = 100

    # Piecewise constant signal
    signal, chg_pts = pw_constant(n=n_samples, clusters=n_regimes,
                                  min_size=50, noisy=True, snr=0.1)

    func_to_minimize = linear_mse(signal)
    func_to_minimize(0, 2)


@raises(NotEnoughPoints)
def test2_linear_mse():
    n_regimes = 2
    n_samples = 100

    # Piecewise constant signal
    signal, chg_pts = pw_constant(n=n_samples, clusters=n_regimes,
                                  min_size=50, noisy=True, snr=0.1)

    func_to_minimize = linear_mse(signal)
    func_to_minimize(4, 4)
