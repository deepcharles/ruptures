from copy import deepcopy
from itertools import product

import numpy as np
import pytest

from ruptures.costs import CostAR
from ruptures.datasets import pw_constant
from ruptures.detection import Binseg, BottomUp, Dynp, Pelt, Window, KernelCPD
from ruptures.exceptions import BadSegmentationParameters


@pytest.fixture(scope="module")
def signal_bkps_5D_n10():
    signal, bkps = pw_constant(n_samples=10, n_features=5, noise_std=1)
    return signal, bkps


@pytest.fixture(scope="module")
def signal_bkps_5D():
    signal, bkps = pw_constant(n_features=5, noise_std=1)
    return signal, bkps


@pytest.fixture(scope="module")
def signal_bkps_1D():
    signal, bkps = pw_constant(noise_std=1)
    return signal.astype(np.float32), bkps


@pytest.fixture(scope="module")
def signal_bkps_5D_no_noise():
    signal, bkps = pw_constant(n_features=5, noise_std=0)
    return signal, bkps


@pytest.fixture(scope="module")
def signal_bkps_1D_no_noise():
    signal, bkps = pw_constant(noise_std=0)
    return signal, bkps


@pytest.fixture(scope="module")
def signal_bkps_1D_constant():
    signal, bkps = np.zeros(200), [200]
    return signal, bkps


@pytest.mark.parametrize("algo", [Binseg, BottomUp, Dynp, Pelt, Window])
def test_empty(signal_bkps_1D, algo):
    signal, _ = signal_bkps_1D
    algo().fit(signal).predict(1)
    algo().fit_predict(signal, 1)


@pytest.mark.parametrize(
    "algo, model",
    product(
        [Binseg, BottomUp, Window],
        ["l1", "l2", "ar", "normal", "rbf", "rank", "mahalanobis"],
    ),
)
def test_model_1D(signal_bkps_1D, algo, model):
    signal, _ = signal_bkps_1D
    algo(model=model).fit_predict(signal, pen=1)
    ret = algo(model=model).fit_predict(signal, n_bkps=1)
    assert len(ret) == 2
    assert ret[-1] == signal.shape[0]
    algo(model=model).fit_predict(signal, epsilon=10)


@pytest.mark.parametrize(
    "algo, model",
    product([Dynp, Pelt], ["l1", "l2", "ar", "normal", "rbf", "rank", "mahalanobis"]),
)
def test_model_1D_bis(signal_bkps_1D, algo, model):
    signal, _ = signal_bkps_1D
    algo_t = algo(model=model)
    ret = algo_t.fit_predict(signal, 1)
    if isinstance(algo_t, Dynp):
        assert len(ret) == 2
    assert ret[-1] == signal.shape[0]


@pytest.mark.parametrize(
    "algo, model",
    product(
        [Dynp, Pelt, Binseg, BottomUp, Window],
        ["l1", "l2", "ar", "normal", "rbf", "rank"],
    ),
)
def test_model_1D_constant(signal_bkps_1D_constant, algo, model):
    signal, _ = signal_bkps_1D_constant
    algo_t = algo(model=model)
    ret = algo_t.fit_predict(signal, 1)
    if isinstance(algo_t, Dynp) or isinstance(algo_t, BottomUp):
        # With constant signal, those search methods
        # will return another break points alongside signal.shape[0]
        assert len(ret) == 2
    if isinstance(algo_t, Binseg):
        if model == "normal":
            # With constant signal, this search method with normal cost
            # will return only signal.shape[0] as breaking points
            assert len(ret) == 1
        else:
            # With constant signal, this search method with another cost
            # will return another break points alongside signal.shape[0]
            assert len(ret) == 2
    if isinstance(algo_t, Window):
        # With constant signal, this search methods
        # will return only signal.shape[0] as breaking points
        assert len(ret) == 1
    if isinstance(algo_t, Pelt):
        assert len(ret) <= 2
    assert ret[-1] == signal.shape[0]


@pytest.mark.parametrize(
    "algo, model",
    product(
        [Binseg, BottomUp, Window],
        ["l1", "l2", "linear", "normal", "rbf", "rank", "mahalanobis"],
    ),
)
def test_model_5D(signal_bkps_5D, algo, model):
    signal, _ = signal_bkps_5D
    algo(model=model).fit_predict(signal, pen=1)
    ret = algo(model=model).fit_predict(signal, n_bkps=1)
    assert len(ret) == 2
    algo(model=model).fit_predict(signal, epsilon=10)


@pytest.mark.parametrize(
    "algo, model",
    product(
        [Dynp, Pelt], ["l1", "l2", "linear", "normal", "rbf", "rank", "mahalanobis"]
    ),
)
def test_model_5D_bis(signal_bkps_5D, algo, model):
    signal, _ = signal_bkps_5D
    algo_t = algo(model=model)
    ret = algo_t.fit_predict(signal, 1)
    if isinstance(algo_t, Dynp):
        assert len(ret) == 2


@pytest.mark.parametrize("algo", [Binseg, BottomUp, Window, Dynp, Pelt])
def test_custom_cost(signal_bkps_1D, algo):
    signal, _ = signal_bkps_1D
    c = CostAR(order=10)
    algo_t = algo(custom_cost=c)
    ret = algo_t.fit_predict(signal, 1)
    if isinstance(algo_t, Pelt):
        assert len(ret) >= 2
    else:
        assert len(ret) == 2


@pytest.mark.parametrize("algo", [Binseg, BottomUp, Window, Dynp, Pelt])
def test_pass_param_to_cost(signal_bkps_1D, algo):
    signal, _ = signal_bkps_1D
    algo_t = algo(model="ar", params={"order": 10})
    ret = algo_t.fit_predict(signal, 1)
    if isinstance(algo_t, Pelt):
        assert len(ret) >= 2
    else:
        assert len(ret) == 2


@pytest.mark.parametrize(
    "kernel, min_size",
    product(["linear"], [2, 5]),
)
def test_kernelcpd_1D_linear(signal_bkps_1D, kernel, min_size):
    signal, bkps = signal_bkps_1D
    ret = (
        KernelCPD(kernel=kernel, min_size=min_size, jump=1)
        .fit(signal)
        .predict(n_bkps=len(bkps) - 1)
    )
    assert len(ret) == len(bkps)


@pytest.mark.parametrize(
    "kernel, min_size",
    product(["linear"], [2, 5]),
)
def test_kernelcpd_5D_linear(signal_bkps_5D, kernel, min_size):
    signal, bkps = signal_bkps_5D
    ret = (
        KernelCPD(kernel=kernel, min_size=min_size, jump=1)
        .fit(signal)
        .predict(n_bkps=len(bkps) - 1)
    )
    assert len(ret) == len(bkps)


@pytest.mark.parametrize(
    "kernel, min_size",
    product(["rbf"], [2, 5]),
)
def test_kernelcpd_1D_rbf(signal_bkps_1D, kernel, min_size):
    signal, bkps = signal_bkps_1D
    ret = (
        KernelCPD(kernel=kernel, min_size=min_size, jump=1, params={"gamma": 1.5})
        .fit(signal)
        .predict(n_bkps=len(bkps) - 1)
    )
    assert len(ret) == len(bkps)


@pytest.mark.parametrize(
    "kernel, min_size",
    product(["rbf"], [2, 5]),
)
def test_kernelcpd_5D_rbf(signal_bkps_5D, kernel, min_size):
    signal, bkps = signal_bkps_5D
    ret = (
        KernelCPD(kernel=kernel, min_size=min_size, jump=1, params={"gamma": 1.5})
        .fit(signal)
        .predict(n_bkps=len(bkps) - 1)
    )
    assert len(ret) == len(bkps)


@pytest.mark.parametrize(
    "kernel, min_size",
    product(["linear"], [2, 5]),
)
def test_kernelcpd_1D_no_noise_linear(signal_bkps_1D_no_noise, kernel, min_size):
    signal, bkps = signal_bkps_1D_no_noise
    res = (
        KernelCPD(kernel=kernel, min_size=min_size, jump=1)
        .fit(signal)
        .predict(n_bkps=len(bkps) - 1)
    )
    assert res == bkps


@pytest.mark.parametrize(
    "kernel, min_size",
    product(["linear"], [2, 5]),
)
def test_kernelcpd_5D_no_noise_linear(signal_bkps_5D_no_noise, kernel, min_size):
    signal, bkps = signal_bkps_5D_no_noise
    res = (
        KernelCPD(kernel=kernel, min_size=min_size, jump=1)
        .fit(signal)
        .predict(n_bkps=len(bkps) - 1)
    )
    assert res == bkps


@pytest.mark.parametrize(
    "kernel, min_size",
    product(["rbf"], [2, 5]),
)
def test_kernelcpd_1D_no_noise_rbf(signal_bkps_1D_no_noise, kernel, min_size):
    signal, bkps = signal_bkps_1D_no_noise
    res = (
        KernelCPD(kernel=kernel, min_size=min_size, jump=1, params={"gamma": 1.5})
        .fit(signal)
        .predict(n_bkps=len(bkps) - 1)
    )
    assert res == bkps


@pytest.mark.parametrize(
    "kernel, min_size",
    product(["rbf"], [2, 5]),
)
def test_kernelcpd_5D_no_noise_rbf(signal_bkps_5D_no_noise, kernel, min_size):
    signal, bkps = signal_bkps_5D_no_noise
    res = (
        KernelCPD(kernel=kernel, min_size=min_size, jump=1, params={"gamma": 1.5})
        .fit(signal)
        .predict(n_bkps=len(bkps) - 1)
    )
    assert res == bkps


# Exhaustive test of KernelCPD
@pytest.mark.parametrize("kernel", ["linear", "rbf", "cosine"])
def test_kernelcpd(signal_bkps_5D, kernel):
    signal, bkps = signal_bkps_5D
    # Test we do not compute if intermediary results exist
    algo_temp = KernelCPD(kernel=kernel)
    algo_temp.fit(signal).predict(n_bkps=len(bkps) - 1)
    algo_temp.predict(n_bkps=1)
    # Test penalized version
    KernelCPD(kernel=kernel).fit(signal).predict(pen=0.2)
    # Test fit_predict
    KernelCPD(kernel=kernel).fit_predict(signal, pen=0.2)


@pytest.mark.parametrize("kernel", ["linear", "rbf", "cosine"])
def test_kernelcpd_small_signal(signal_bkps_5D_n10, kernel):
    signal, _ = signal_bkps_5D_n10
    algo_temp = KernelCPD(kernel=kernel)
    with pytest.raises(BadSegmentationParameters):
        KernelCPD(kernel=kernel, min_size=10, jump=2).fit_predict(signal, n_bkps=2)
    with pytest.raises(AssertionError):
        KernelCPD(kernel=kernel, min_size=10, jump=2).fit_predict(signal, n_bkps=0)
    with pytest.raises(BadSegmentationParameters):
        KernelCPD(kernel=kernel, min_size=10, jump=2).fit_predict(signal, pen=0.2)
    assert (
        len(KernelCPD(kernel=kernel, min_size=5, jump=2).fit_predict(signal, pen=0.2))
        > 0
    )


@pytest.mark.parametrize(
    "algo, model",
    product(
        [Binseg, BottomUp, Window],
        ["l1", "l2", "ar", "normal", "rbf", "rank", "mahalanobis"],
    ),
)
def test_model_small_signal(signal_bkps_5D_n10, algo, model):
    signal, _ = signal_bkps_5D_n10
    with pytest.raises(BadSegmentationParameters):
        algo(model=model, min_size=5, jump=2).fit_predict(signal, n_bkps=2)
    assert (
        len(algo(model=model, min_size=5, jump=2).fit_predict(signal, pen=10 ** 6)) > 0
    )
    assert (
        len(algo(model=model, min_size=5, jump=2).fit_predict(signal, epsilon=10)) > 0
    )
    assert (
        len(algo(model=model, min_size=9, jump=2).fit_predict(signal, pen=10 ** 6)) > 0
    )


@pytest.mark.parametrize(
    "model", ["l1", "l2", "ar", "normal", "rbf", "rank", "mahalanobis"]
)
def test_model_small_signal_dynp(signal_bkps_5D_n10, model):
    signal, _ = signal_bkps_5D_n10
    with pytest.raises(BadSegmentationParameters):
        Dynp(model=model, min_size=5, jump=2).fit_predict(signal, 2)
    with pytest.raises(BadSegmentationParameters):
        Dynp(model=model, min_size=9, jump=2).fit_predict(signal, 2)
    with pytest.raises(BadSegmentationParameters):
        Dynp(model=model, min_size=11, jump=2).fit_predict(signal, 2)


@pytest.mark.parametrize(
    "model", ["l1", "l2", "ar", "normal", "rbf", "rank", "mahalanobis"]
)
def test_model_small_signal_pelt(signal_bkps_5D_n10, model):
    signal, _ = signal_bkps_5D_n10
    with pytest.raises(BadSegmentationParameters):
        Pelt(model=model, min_size=11, jump=2).fit_predict(signal, 2)
    assert len(Pelt(model=model, min_size=10, jump=2).fit_predict(signal, 1.0)) > 0


def test_binseg_deepcopy():
    binseg = Binseg()
    binseg_copy = deepcopy(binseg)
    assert id(binseg.single_bkp) != id(binseg_copy.single_bkp)
