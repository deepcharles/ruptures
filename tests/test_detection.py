from itertools import product

import numpy as np
import pytest

from ruptures.costs import NotEnoughPoints, CostAR
from ruptures.datasets import pw_constant
from ruptures.detection import Binseg, BottomUp, Dynp, Pelt, Window


@pytest.fixture(scope="module")
def signal_bkps_5D():
    signal, bkps = pw_constant(n_features=5, noise_std=1)
    return signal, bkps


@pytest.fixture(scope="module")
def signal_bkps_1D():
    signal, bkps = pw_constant(noise_std=1)
    return signal, bkps


@pytest.fixture(scope="module")
def signal_bkps_1D_constant():
    signal, bkps = np.zeros(200), [200]
    return signal, bkps


@pytest.mark.parametrize("algo", [Binseg, BottomUp, Dynp, Pelt, Window])
def test_empty(signal_bkps_1D, algo):
    signal, bkps = signal_bkps_1D
    algo().fit(signal).predict(1)
    algo().fit_predict(signal, 1)


@pytest.mark.parametrize(
    "algo, model",
    product([Binseg, BottomUp, Window], ["l1", "l2", "ar", "normal", "rbf", "rank"]),
)
def test_model_1D(signal_bkps_1D, algo, model):
    signal, bkps = signal_bkps_1D
    algo().fit_predict(signal, pen=1)
    algo().fit_predict(signal, n_bkps=1)
    algo().fit_predict(signal, epsilon=10)


@pytest.mark.parametrize(
    "algo, model", product([Dynp, Pelt], ["l1", "l2", "ar", "normal", "rbf", "rank"])
)
def test_model_1D_bis(signal_bkps_1D, algo, model):
    signal, bkps = signal_bkps_1D
    algo().fit_predict(signal, 1)


@pytest.mark.parametrize(
    "algo, model",
    product(
        [Dynp, Pelt, Binseg, BottomUp, Window],
        ["l1", "l2", "ar", "normal", "rbf", "rank"],
    ),
)
def test_model_1D_constant(signal_bkps_1D_constant, algo, model):
    signal, bkps = signal_bkps_1D_constant
    algo().fit_predict(signal, 1)


@pytest.mark.parametrize(
    "algo, model",
    product(
        [Binseg, BottomUp, Window], ["l1", "l2", "linear", "normal", "rbf", "rank"]
    ),
)
def test_model_5D(signal_bkps_5D, algo, model):
    signal, bkps = signal_bkps_5D
    algo().fit_predict(signal, pen=1)
    algo().fit_predict(signal, n_bkps=1)
    algo().fit_predict(signal, epsilon=10)


@pytest.mark.parametrize(
    "algo, model",
    product([Dynp, Pelt], ["l1", "l2", "linear", "normal", "rbf", "rank"]),
)
def test_model_5D_bis(signal_bkps_5D, algo, model):
    signal, bkps = signal_bkps_5D
    algo().fit_predict(signal, 1)


@pytest.mark.parametrize("algo", [Binseg, BottomUp, Window, Dynp, Pelt])
def test_custom_cost(signal_bkps_1D, algo):
    signal, bkps = signal_bkps_1D
    c = CostAR(order=10)
    algo(custom_cost=c).fit_predict(signal, 1)


@pytest.mark.parametrize("algo", [Binseg, BottomUp, Window, Dynp, Pelt])
def test_pass_param_to_cost(signal_bkps_1D, algo):
    signal, bkps = signal_bkps_1D
    algo(model="ar", params={"order": 10}).fit_predict(signal, 1)
