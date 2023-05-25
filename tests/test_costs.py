import numpy as np
import pytest
from ruptures import Binseg
from ruptures.costs import CostLinear, CostNormal, cost_factory
from ruptures.costs.costml import CostMl
from ruptures.datasets import pw_constant
from ruptures.exceptions import NotEnoughPoints


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
        elif cost_name == "l2":
            cost.error(1, 1)
        elif cost_name == "rbf":
            cost.error(1, 1)
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
        elif cost_name == "l2":
            cost.error(1, 1)
        elif cost_name == "rbf":
            cost.error(1, 1)
        else:
            cost.error(1, 2)


@pytest.mark.parametrize("cost_name", cost_names)
def test_costs_5D_names(signal_bkps_5D, cost_name):
    signal, bkps = signal_bkps_5D
    cost = cost_factory(cost_name)
    try:
        cost.fit(signal)
    except np.linalg.LinAlgError as e:
        if cost_name == "mahalanobis":
            reason_msg = (
                "The Mahalanobis metric matrix cannot be computed "
                "because the signal has a singular covariance matrix"
            )
            pytest.skip(reason_msg)
        else:
            raise
    cost.error(0, 100)
    cost.error(100, signal.shape[0])
    cost.error(10, 50)
    cost.sum_of_costs(bkps)
    if cost_name == "l2":
        assert cost.error(1, 2) == 0.0, "CostL2 on segment of size 1 returns 0."

    with pytest.raises(NotEnoughPoints):
        if cost_name == "cosine":
            cost.min_size = 4
            cost.error(1, 2)
        elif cost_name == "l2":
            cost.error(1, 1)
        elif cost_name == "rbf":
            cost.error(1, 1)
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
        elif cost_name == "l2":
            cost.error(1, 1)
        elif cost_name == "rbf":
            cost.error(1, 1)
        else:
            cost.error(1, 2)


def test_factory_exception():
    with pytest.raises(ValueError):
        cost_factory("Dummy cost name")


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
    # For signals that have truly constant segments, CostNormal should not fail
    # with the correction on the diagonal of the covariance matrix.

    # generate data
    signal_1D = np.r_[np.ones(100), np.arange(100), np.zeros(100)]
    signal_2D = np.c_[signal_1D, signal_1D[::-1]]
    n_samples = signal_1D.shape[0]
    bkps = [100, 200, n_samples]

    # test cost function
    for signal in (signal_1D, signal_2D):
        c = CostNormal(add_small_diag=True).fit(signal=signal)
        c.error(0, 100)
        c.error(100, n_samples)
        c.error(10, 50)
        c.sum_of_costs(bkps)
        with pytest.raises(NotEnoughPoints):
            c.error(10, 11)

    # test cost function without correction
    c = CostNormal(add_small_diag=False).fit(signal=signal_1D)
    assert np.isinf(c.error(0, 100))


def test_costml(signal_bkps_5D_noisy, signal_bkps_1D_noisy):
    """Test if `CostMl.fit` actually (re-)fits the metric matrix.

    Refitting the metric matrix should only happen if the user did not
    provide a metric matrix.
    """
    # no user-defined metric matrix
    c = CostMl()
    for signal, bkps in (signal_bkps_5D_noisy, signal_bkps_1D_noisy):
        c.fit(signal=signal)
        c.error(0, 100)
        c.sum_of_costs(bkps)
    # with a user-defined metric matrix
    signal, bkps = signal_bkps_5D_noisy
    _, n_dims = signal.shape
    c = CostMl(metric=np.eye(n_dims))
    c.fit(signal)
    c.error(10, 50)
    assert np.allclose(c.metric, np.eye(n_dims))


def test_costl2_small_data():
    """Test if CostL2 returns the correct segmentation for small data."""
    signal = np.array([0.0, 0.1, 1.2, 1.0])
    n_samples = signal.shape[0]
    algo = Binseg(model="l2", min_size=1, jump=1).fit(signal)
    computed_break_dict = {}
    for n_bkps in range(n_samples):
        try:
            result = algo.predict(n_bkps=n_bkps)
        except Exception as e:
            result = e
            computed_break_dict
        computed_break_dict[n_bkps] = result

    expected_break_dict = {0: [4], 1: [2, 4], 2: [2, 3, 4], 3: [1, 2, 3, 4]}
    err_msg = (
        f"Wrong segmentation, expected {expected_break_dict}, ",
        "got {computed_break_dict}.",
    )
    assert expected_break_dict == computed_break_dict, err_msg
