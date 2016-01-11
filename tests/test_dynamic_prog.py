import numpy as np
from ruptures.dynamic_programming import dynp
from ruptures.costs import gaussmean, linear_mse, NotEnoughPoints
from ruptures.datasets import pw_linear, pw_constant
from nose.tools import raises


def test1_ruptures1D():
    """on vérifie que les algorithmes de programmation dynamique tournent."""
    n_regimes = 3
    n_samples = 500

    # Piecewise constant signal
    signal, chg_pts = pw_constant(n=n_samples, clusters=n_regimes,
                                  min_size=50, noisy=True, snr=0.1)

    func_to_minimize = gaussmean(signal)  # - log likelihood
    min_size, jump = 2, 1
    dp1 = dynp(error_func=func_to_minimize,
               n=n_samples, n_regimes=n_regimes,
               min_size=min_size, jump=jump)
    dp1.fit()
    # Piecewise linear signal
    signal, chg_pts = pw_linear(n=n_samples, clusters=n_regimes,
                                min_size=50, noisy=True, snr=0.1)

    func_to_minimize = linear_mse(signal)  # mean squared error
    min_size, jump = 3, 1
    dp2 = dynp(error_func=func_to_minimize,
               n=n_samples, n_regimes=n_regimes,
               min_size=min_size, jump=jump)
    dp2.fit()

    return dp1, dp2


@raises(NotEnoughPoints)
def test2_ruptures1D():
    """On spécifie une min_size trop petite pour vérifier que la bonne
    exception est appelée."""
    n_regimes = 3
    n_samples = 500

    # Piecewise constant signal
    signal, chg_pts = pw_constant(n=n_samples, clusters=n_regimes,
                                  min_size=50, noisy=True, snr=0.1)

    func_to_minimize = gaussmean(signal)  # - log likelihood
    min_size, jump = 1, 1
    dp = dynp(error_func=func_to_minimize,
              n=n_samples, n_regimes=n_regimes,
              min_size=min_size, jump=jump)
    dp.fit()
    return dp


@raises(NotEnoughPoints)
def test3_ruptures1D():
    """On spécifie une min_size trop petite pour vérifier que la bonne
    exception est appelée."""
    n_regimes = 3
    n_samples = 500

    # Piecewise linear signal
    signal, chg_pts = pw_linear(n=n_samples, clusters=n_regimes,
                                min_size=50, noisy=True, snr=0.1)

    func_to_minimize = linear_mse(signal)  # mean squared error
    min_size, jump = 2, 1
    dp = dynp(error_func=func_to_minimize,
              n=n_samples, n_regimes=n_regimes,
              min_size=min_size, jump=jump)
    dp.fit()
    return dp


def test4_ruptures1D():
    """On vérifie dans un cas simple que la détection de ruptures est bonne."""
    n_regimes = 3
    n_samples = 500

    # Piecewise constant signal
    signal, chg_pts = pw_constant(n=n_samples, clusters=n_regimes,
                                  min_size=50, noisy=True, snr=0.001)

    func_to_minimize = gaussmean(signal)  # - log likelihood
    min_size, jump = 2, 1
    dp = dynp(error_func=func_to_minimize,
              n=n_samples, n_regimes=n_regimes,
              min_size=min_size, jump=jump)
    my_chg_pts = dp.fit()

    assert np.all(abs(a - b) < 3 for (a, b)
                  in zip(sorted(chg_pts), sorted(my_chg_pts)))
