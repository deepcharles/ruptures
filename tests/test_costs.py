import pytest
import numpy as np
from itertools import product, accumulate
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


def has_holes(partition):
    """Returns the start and end of a partition. AssertionError if there is a
    hole."""
    def fusion(interval1, interval2):
        assert interval1[1] == interval2[0], "Hole in the partition."
        return (interval1[0], interval2[1])
    sorted_intervals = sorted((s, e) for s, e in partition)
    return list(accumulate(sorted_intervals, fusion))[-1]


@pytest.mark.parametrize("penalty, algo", product(PENALTIES, ALGOS))
def test_pelt(signal_bkps, penalty, algo):
    signal, bkps = signal_bkps
    method = "pelt"
    a = algo(method)
    my_bkps = a.fit(penalty=penalty, signal=signal, jump=5, min_size=20)
    assert my_bkps[-1] == signal.shape[
        0], "The last breakpoint must be equal to the signal length."

    assert has_holes(a._partition) == (
        0, signal.shape[0]), "The partition does not cover the signal."


@pytest.mark.parametrize("algo", ALGOS)
def test_dynp(signal_bkps, algo):
    signal, bkps = signal_bkps
    n_regimes = len(bkps)
    method = "dynp"
    a = algo(method)
    my_bkps = a.fit(n_regimes=n_regimes, signal=signal, jump=5, min_size=20)
    assert my_bkps[-1] == signal.shape[
        0], "The last breakpoint must be equal to the signal length"

    assert has_holes(a._partition) == (
        0, signal.shape[0]), "The partition does not cover the signal."

    for n_reg in range(2, n_regimes + 1):
        my_bkps = a.fit(n_regimes=n_reg, signal=signal, jump=5, min_size=20)
        assert my_bkps[-1] == signal.shape[
            0], "The last breakpoint must be equal to the signal length"
        assert has_holes(a._partition) == (
            0, signal.shape[0]), "The partition does not cover the signal."


@pytest.mark.parametrize("algo", ALGOS)
def test_exceptions(signal_bkps, algo):
    signal, bkps = signal_bkps
    n_regimes = len(bkps)

    method = "dynp"
    a = algo(method)
    with pytest.raises(NotEnoughPoints):
        a.fit(n_regimes=n_regimes, signal=signal, min_size=1)

    method = "pelt"
    a = algo(method)
    with pytest.raises(NotEnoughPoints):
        a.fit(penalty=1, signal=signal, min_size=1)


@pytest.mark.parametrize("algo", ALGOS)
def test_dynp_bis(signal_bkps, algo):
    signal, bkps = signal_bkps
    n_regimes = len(bkps)
    method = "dynp"
    a = algo(method)
    my_bkps = a.fit(n_regimes=n_regimes, signal=signal, jump=5, min_size=20)
    assert my_bkps[-1] == signal.shape[
        0], "The last breakpoint must be equal to the signal length"

    assert has_holes(a._partition) == (
        0, signal.shape[0]), "The partition does not cover the signal."

    for n_reg in range(2, n_regimes + 1):
        my_bkps = a.fit(n_regimes=n_reg, jump=5, min_size=20)
        assert my_bkps[-1] == signal.shape[
            0], "The last breakpoint must be equal to the signal length"
        assert has_holes(a._partition) == (
            0, signal.shape[0]), "The partition does not cover the signal."


@pytest.mark.parametrize("algo", ALGOS)
def test_pelt_bis(signal_bkps, algo):
    signal, bkps = signal_bkps
    method = "pelt"
    a = algo(method)
    my_bkps = a.fit(penalty=1, signal=signal, jump=5, min_size=20)
    for pen in PENALTIES:
        my_bkps = a.fit(penalty=pen, jump=5, min_size=20)
        assert my_bkps[-1] == signal.shape[
            0], "The last breakpoint must be equal to the signal length."
        assert has_holes(a._partition) == (
            0, signal.shape[0]), "The partition does not cover the signal."
