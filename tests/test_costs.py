import pytest
import numpy as np
from itertools import product
from ruptures.costs import ConstantMSE, GaussMLE, LinearMLE, HarmonicMSE
from ruptures.datasets import pw_constant
from ruptures.costs import NotEnoughPoints
PENALTIES = np.linspace(0.1, 100, 10)
ALGOS = [ConstantMSE, GaussMLE, LinearMLE, HarmonicMSE]


@pytest.fixture(scope="module")
def signal_bkps():
    n_samples = 300
    n_regimes = 3
    dim = 3
    signal, bkps = pw_constant(n=n_samples,
                               clusters=n_regimes,
                               dim=dim,
                               noisy=True,
                               snr=.01,
                               min_size=20)
    return signal, bkps


@pytest.mark.parametrize("penalty, algo", product(PENALTIES, ALGOS))
def test_pelt(signal_bkps, penalty, algo):
    signal, bkps = signal_bkps
    method = "pelt"
    a = algo(method)
    a.fit(signal, penalty=penalty, jump=5, min_size=20)
    assert a.bkps[-1] == signal.shape[
        0], "The last breakpoint must be equal to the signal length"


@pytest.mark.parametrize("algo", ALGOS)
def test_dynp(signal_bkps, algo):
    signal, bkps = signal_bkps
    n_regimes = len(bkps)
    method = "dynp"
    a = algo(method)
    a.fit(signal, n_regimes=n_regimes, jump=5, min_size=20)
    assert a.bkps[-1] == signal.shape[
        0], "The last breakpoint must be equal to the signal length"


@pytest.mark.parametrize("algo", ALGOS)
def test_exceptions(signal_bkps, algo):
    signal, bkps = signal_bkps
    n_regimes = len(bkps)

    method = "dynp"
    a = algo(method)
    with pytest.raises(NotEnoughPoints):
        a.fit(signal, n_regimes=n_regimes, min_size=1)

    method = "pelt"
    a = algo(method)
    with pytest.raises(NotEnoughPoints):
        a.fit(signal, penalty=1, min_size=1)
