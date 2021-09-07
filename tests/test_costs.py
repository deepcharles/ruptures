import pytest
import numpy as np
from ruptures.costs import cost_factory, CostLinear
from ruptures.costs.costnormal import CostNormal
from ruptures.datasets import pw_constant
from ruptures.exceptions import NotEnoughPoints
import numpy as np


@pytest.fixture(scope="module")
def signal_bkps_1D():
    signal, bkps = pw_constant(n_features=1, seed=1234567890)
    return signal, bkps


@pytest.fixture(scope="module")
def signal_bkps_1D_noisy():
    signal, bkps = pw_constant(n_features=1, noise_std=1, seed=1234567890)
    return signal, bkps


@pytest.fixture(scope="module")
def signal_bkps_5D():
    signal, bkps = pw_constant(n_features=5, seed=1234567890)
    return signal, bkps


@pytest.fixture(scope="module")
def signal_bkps_5D_noisy():
    signal, bkps = pw_constant(n_features=5, noise_std=1, seed=1234567890)
    return signal, bkps


cost_names = {
    "ar",
    "l1",
    "l2",
    "normal",
    "rbf",
    "rank",
    "clinear",
    "mahalanobis",
    "cosine",
}


@pytest.mark.parametrize("cost_name", cost_names)
def test_costs_1D_names(signal_bkps_1D, cost_name):
    signal, bkps = signal_bkps_1D
    cost = cost_factory(cost_name)
    cost.fit(signal)
    cost.fit(signal.flatten())
    cost.error(0, 100)
    cost.error(100, signal.shape[0])
    cost.error(10, 50)
    cost.sum_of_costs(bkps)
    with pytest.raises(NotEnoughPoints):
        if cost_name == "cosine":
            cost.min_size = 4
            cost.error(1, 2)
        else:
            cost.error(1, 2)


@pytest.mark.parametrize("cost_name", cost_names)
def test_costs_1D_noisy_names(signal_bkps_1D_noisy, cost_name):
    signal, bkps = signal_bkps_1D_noisy
    cost = cost_factory(cost_name)
    cost.fit(signal)
    cost.fit(signal.flatten())
    cost.error(0, 100)
    cost.error(100, signal.shape[0])
    cost.error(10, 50)
    cost.sum_of_costs(bkps)
    with pytest.raises(NotEnoughPoints):
        if cost_name == "cosine":
            cost.min_size = 4
            cost.error(1, 2)
        else:
            cost.error(1, 2)


@pytest.mark.parametrize("cost_name", cost_names)
def test_costs_5D_names(signal_bkps_5D, cost_name):
    signal, bkps = signal_bkps_5D
    cost = cost_factory(cost_name)
    cost.fit(signal)
    cost.error(0, 100)
    cost.error(100, signal.shape[0])
    cost.error(10, 50)
    cost.sum_of_costs(bkps)
    with pytest.raises(NotEnoughPoints):
        if cost_name == "cosine":
            cost.min_size = 4
            cost.error(1, 2)
        else:
            cost.error(1, 2)


@pytest.mark.parametrize("cost_name", cost_names)
def test_costs_5D_noisy_names(signal_bkps_5D_noisy, cost_name):
    signal, bkps = signal_bkps_5D_noisy
    cost = cost_factory(cost_name)
    cost.fit(signal)
    cost.error(0, 100)
    cost.error(100, signal.shape[0])
    cost.error(10, 50)
    cost.sum_of_costs(bkps)
    with pytest.raises(NotEnoughPoints):
        if cost_name == "cosine":
            cost.min_size = 4
            cost.error(1, 2)
        else:
            cost.error(1, 2)


def test_factory_exception():
    with pytest.raises(ValueError):
        cost_factory("bkd;s")


# Test CostLinear
def test_costlinear(signal_bkps_5D_noisy, signal_bkps_1D_noisy):
    # Creation of data. For convenience, we use
    # already generated signal_bkps_5D and signal_bkps_1D
    signal_regressors, _ = signal_bkps_5D_noisy  # regressors
    signal, bkps = signal_bkps_1D_noisy  # observed signal
    n = signal.shape[0]  # number of samples

    # First dimension is the observed signal.
    # Stack observed signal and regressors.
    # Add intercept to regressors
    s = np.c_[signal, signal_regressors, np.ones(n)]
    # Compute error
    c = CostLinear().fit(s)
    c.error(0, 100)
    c.error(100, n)
    c.error(10, 50)
    c.sum_of_costs(bkps)
    with pytest.raises(NotEnoughPoints):
        c.error(10, 11)

# Test CostNormal
def test_costnormal():
    # For signals that have truly constant segments, CostNormal should not fail.
    signal_1D = np.r_[np.ones(100), np.arange(100), np.zeros(100)]
    signal_2D = np.c_[signal_1D, signal_1D[::-1]]
    n_samples = signal_1D.shape[0]
    bkps = [100, 200, n_samples]

    # with the correction on the diagonal of the covariance matrix.
    for signal in (signal_1D, signal_2D):
        c = CostNormal(add_small_diag=True).fit(signal=signal)
        c.error(0, 100)
        c.error(100, n_samples)
        c.error(10, 50)
        c.sum_of_costs(bkps)
        with pytest.raises(NotEnoughPoints):
            c.error(10, 11)
