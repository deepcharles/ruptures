import pytest
import numpy as np
from itertools import product
from ruptures.costs import KernelMSE
from ruptures.costs.kernel import valid_kernels
from ruptures.datasets import pw_constant
from ruptures.costs import NotEnoughPoints
PENALTIES = np.linspace(0.1, 100, 10)


@pytest.fixture(scope="module")
def signal_bkps():
    n_samples = 300
    n_regimes = 3
    signal, bkps = pw_constant(n=n_samples,
                               clusters=n_regimes,
                               noisy=True,
                               snr=.01)
    return signal, bkps


@pytest.mark.parametrize("penalty, kernel", product(PENALTIES, valid_kernels))
def test_pelt(signal_bkps, penalty, kernel):
    signal, bkps = signal_bkps
    method = "pelt"
    a = KernelMSE(method, kernel)
    a.fit(signal, penalty=penalty, jump=5, min_size=3)
    assert a.bkps[-1] == signal.shape[
        0], "The last breakpoint must be equal to the signal length"


@pytest.mark.parametrize("kernel", valid_kernels)
def test_dynp(signal_bkps, kernel):
    signal, bkps = signal_bkps
    n_regimes = len(bkps)
    method = "dynp"
    a = KernelMSE(method, kernel)
    a.fit(signal, n_regimes=n_regimes, jump=5, min_size=2)
    assert a.bkps[-1] == signal.shape[
        0], "The last breakpoint must be equal to the signal length"


@pytest.mark.parametrize("kernel", valid_kernels)
def test_exceptions(signal_bkps, kernel):
    signal, bkps = signal_bkps
    n_regimes = len(bkps)

    method = "dynp"
    a = KernelMSE(method, kernel)
    with pytest.raises(NotEnoughPoints):
        a.fit(signal, n_regimes=n_regimes, min_size=1)

    method = "pelt"
    a = KernelMSE(method, kernel)
    with pytest.raises(NotEnoughPoints):
        a.fit(signal, penalty=1, min_size=1)
