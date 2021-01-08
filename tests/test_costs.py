import pytest
from ruptures.costs import (
    CostAR,
    CostCLinear,
    CostL1,
    CostL2,
    CostLinear,
    CostNormal,
    CostRank,
    CostRbf,
    cost_factory,
)
from ruptures.datasets import pw_constant
from ruptures.exceptions import NotEnoughPoints
import numpy as np


@pytest.fixture(scope="module")
def signal_bkps_1D():
    signal, bkps = pw_constant(n_features=1)
    return signal, bkps


@pytest.fixture(scope="module")
def signal_bkps_1D_noisy():
    signal, bkps = pw_constant(n_features=1, noise_std=1)
    return signal, bkps


@pytest.fixture(scope="module")
def signal_bkps_5D():
    signal, bkps = pw_constant(n_features=5)
    return signal, bkps


@pytest.fixture(scope="module")
def signal_bkps_5D_noisy():
    signal, bkps = pw_constant(n_features=5, noise_std=1)
    return signal, bkps


cost_classes = {CostAR, CostL1, CostL2, CostNormal, CostRbf, CostRank}
cost_names = {"ar", "l1", "l2", "normal", "rbf", "rank"}


@pytest.mark.parametrize("cost", cost_classes)
def test_costs_1D(signal_bkps_1D, cost):
    signal, bkps = signal_bkps_1D
    cost.fit(signal)
    cost.fit(signal.flatten())
    cost.error(0, 100)
    cost.error(100, signal.shape[0])
    cost.error(10, 50)
    cost.sum_of_costs(bkps)
    with pytest.raises(NotEnoughPoints):
        cost.error(1, 2)


@pytest.mark.parametrize("cost", cost_classes)
def test_costs_1D_noisy(signal_bkps_1D_noisy, cost):
    signal, bkps = signal_bkps_1D_noisy
    cost.fit(signal)
    cost.fit(signal.flatten())
    cost.error(0, 100)
    cost.error(100, signal.shape[0])
    cost.error(10, 50)
    cost.sum_of_costs(bkps)
    with pytest.raises(NotEnoughPoints):
        cost.error(1, 2)


@pytest.mark.parametrize("cost", cost_classes)
def test_costs_5D(signal_bkps_5D, cost):
    signal, bkps = signal_bkps_5D
    cost.fit(signal)
    cost.error(0, 100)
    cost.error(100, signal.shape[0])
    cost.error(10, 50)
    cost.sum_of_costs(bkps)
    with pytest.raises(NotEnoughPoints):
        cost.error(1, 2)


@pytest.mark.parametrize("cost", cost_classes)
def test_costs_5D_noisy(signal_bkps_5D_noisy, cost):
    signal, bkps = signal_bkps_5D_noisy
    cost.fit(signal)
    cost.error(0, 100)
    cost.error(100, signal.shape[0])
    cost.error(10, 50)
    cost.sum_of_costs(bkps)
    with pytest.raises(NotEnoughPoints):
        cost.error(1, 2)


@pytest.mark.parametrize("cost_name", cost_names)
def test_costs_1D(signal_bkps_1D, cost_name):
    signal, bkps = signal_bkps_1D
    cost = cost_factory(cost_name)
    cost.fit(signal)
    cost.fit(signal.flatten())
    cost.error(0, 100)
    cost.error(100, signal.shape[0])
    cost.error(10, 50)
    cost.sum_of_costs(bkps)
    with pytest.raises(NotEnoughPoints):
        cost.error(1, 2)


@pytest.mark.parametrize("cost_name", cost_names)
def test_costs_1D_noisy(signal_bkps_1D_noisy, cost_name):
    signal, bkps = signal_bkps_1D_noisy
    cost = cost_factory(cost_name)
    cost.fit(signal)
    cost.fit(signal.flatten())
    cost.error(0, 100)
    cost.error(100, signal.shape[0])
    cost.error(10, 50)
    cost.sum_of_costs(bkps)
    with pytest.raises(NotEnoughPoints):
        cost.error(1, 2)


@pytest.mark.parametrize("cost_name", cost_names)
def test_costs_5D(signal_bkps_5D, cost_name):
    signal, bkps = signal_bkps_5D
    cost = cost_factory(cost_name)
    cost.fit(signal)
    cost.error(0, 100)
    cost.error(100, signal.shape[0])
    cost.error(10, 50)
    cost.sum_of_costs(bkps)
    with pytest.raises(NotEnoughPoints):
        cost.error(1, 2)


@pytest.mark.parametrize("cost_name", cost_names)
def test_costs_5D_noisy(signal_bkps_5D_noisy, cost_name):
    signal, bkps = signal_bkps_5D_noisy
    cost = cost_factory(cost_name)
    cost.fit(signal)
    cost.error(0, 100)
    cost.error(100, signal.shape[0])
    cost.error(10, 50)
    cost.sum_of_costs(bkps)
    with pytest.raises(NotEnoughPoints):
        cost.error(1, 2)


def test_factory_exception():
    with pytest.raises(ValueError):
        cost_factory("bkd;s")


# Test CostLinear
def test_costlinear(signal_bkps_5D):
    signal, bkps = signal_bkps_5D
    # creation of data
    n = signal.shape[0]  # number of samples
    # regressors
    tt = np.linspace(0, 10 * np.pi, n)
    X = np.vstack(
        (np.sin(tt), np.sin(2 * tt), np.sin(3 * tt), np.sin(4 * tt), np.ones(n))
    ).T
    # observed signal
    y = np.sum(X * signal, axis=1)
    y += np.random.normal(size=y.shape)
    # stack observed signal and regressors.
    # first dimension is the observed signal.
    s = np.column_stack((y.reshape(-1, 1), X))
    # compute error
    c = CostLinear().fit(s)
    c.error(0, 100)
    c.error(100, signal.shape[0])
    c.error(10, 50)
    c.sum_of_costs(bkps)
    with pytest.raises(NotEnoughPoints):
        c.error(10, 11)
