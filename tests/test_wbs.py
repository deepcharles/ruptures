"""Tests for Wild Binary Segmentation (WBS)."""

from itertools import product

import numpy as np
import pytest

import ruptures as rpt
from ruptures.costs import CostAR
from ruptures.datasets import pw_constant
from ruptures.detection import WBS
from ruptures.exceptions import BadSegmentationParameters


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def signal_1d():
    signal, bkps = pw_constant(n_samples=200, n_bkps=3, noise_std=1, seed=0)
    return signal, bkps


@pytest.fixture(scope="module")
def signal_5d():
    signal, bkps = pw_constant(n_features=5, noise_std=1, seed=1)
    return signal, bkps


@pytest.fixture(scope="module")
def signal_1d_constant():
    return np.zeros(200), [200]


# ---------------------------------------------------------------------------
# Export / API
# ---------------------------------------------------------------------------


def test_top_level_export():
    assert rpt.WBS is WBS


def test_fit_returns_self(signal_1d):
    signal, _ = signal_1d
    algo = WBS(seed=0)
    ret = algo.fit(signal)
    assert ret is algo


# ---------------------------------------------------------------------------
# Stopping rules
# ---------------------------------------------------------------------------


def test_n_bkps(signal_1d):
    signal, _ = signal_1d
    bkps = WBS(seed=0).fit(signal).predict(n_bkps=3)
    assert len(bkps) == 4
    assert bkps[-1] == signal.shape[0]


def test_pen(signal_1d):
    signal, _ = signal_1d
    bkps = WBS(seed=0).fit(signal).predict(pen=1)
    assert bkps[-1] == signal.shape[0]


def test_epsilon(signal_1d):
    signal, _ = signal_1d
    bkps = WBS(seed=0).fit(signal).predict(epsilon=10)
    assert bkps[-1] == signal.shape[0]


def test_fit_predict(signal_1d):
    signal, _ = signal_1d
    bkps = WBS(seed=0).fit_predict(signal, n_bkps=3)
    assert len(bkps) == 4
    assert bkps[-1] == signal.shape[0]


def test_no_param_raises(signal_1d):
    signal, _ = signal_1d
    algo = WBS(seed=0).fit(signal)
    with pytest.raises(AssertionError):
        algo.predict()


# ---------------------------------------------------------------------------
# Multi-dimensional signal
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", ["l1", "l2", "rbf", "normal", "rank"])
def test_models_1d(signal_1d, model):
    signal, _ = signal_1d
    bkps = WBS(model=model, seed=42).fit_predict(signal, n_bkps=1)
    assert len(bkps) == 2
    assert bkps[-1] == signal.shape[0]


@pytest.mark.parametrize("model", ["l1", "l2", "rbf", "normal", "rank"])
def test_models_5d(signal_5d, model):
    signal, _ = signal_5d
    bkps = WBS(model=model, seed=42).fit_predict(signal, n_bkps=1)
    assert len(bkps) == 2
    assert bkps[-1] == signal.shape[0]


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_seed_gives_same_result(signal_1d):
    signal, _ = signal_1d
    bkps1 = WBS(seed=7).fit_predict(signal, n_bkps=3)
    bkps2 = WBS(seed=7).fit_predict(signal, n_bkps=3)
    assert bkps1 == bkps2


def test_different_seeds_may_differ(signal_1d):
    """Two different seeds should not always give identical results."""
    signal, _ = signal_1d
    results = {
        tuple(WBS(seed=s).fit_predict(signal, n_bkps=5)) for s in range(10)
    }
    # With 10 different seeds there should be at least 2 distinct results
    # (this could theoretically fail but is astronomically unlikely)
    assert len(results) >= 2


# ---------------------------------------------------------------------------
# Statistical accuracy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_recovers_single_change_point(seed):
    """WBS should reliably find a single large change point."""
    rng = np.random.default_rng(seed)
    n = 200
    signal = np.concatenate([rng.normal(0, 0.1, n), rng.normal(5, 0.1, n)])
    bkps = WBS(model="l2", seed=seed).fit_predict(signal, n_bkps=1)
    # True change point is at 200; allow ±5 samples of tolerance
    assert abs(bkps[0] - n) <= 5


# ---------------------------------------------------------------------------
# Custom cost
# ---------------------------------------------------------------------------


def test_custom_cost(signal_1d):
    signal, _ = signal_1d
    c = CostAR(order=2)
    bkps = WBS(custom_cost=c, seed=0).fit_predict(signal, n_bkps=1)
    assert len(bkps) == 2
    assert bkps[-1] == signal.shape[0]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_constant_signal(signal_1d_constant):
    signal, _ = signal_1d_constant
    bkps = WBS(seed=0).fit(signal).predict(n_bkps=1)
    assert len(bkps) == 2
    assert bkps[-1] == signal.shape[0]


def test_bad_segmentation_raises(signal_1d):
    signal, _ = signal_1d
    with pytest.raises(BadSegmentationParameters):
        WBS(min_size=200, seed=0).fit_predict(signal, n_bkps=5)


def test_float32_signal():
    rng = np.random.default_rng(0)
    signal = np.concatenate([rng.normal(0, 0.1, 50), rng.normal(5, 0.1, 50)]).astype(
        np.float32
    )
    bkps = WBS(seed=0).fit_predict(signal, n_bkps=1)
    assert bkps[-1] == signal.shape[0]


def test_n_draws_parameter(signal_1d):
    """n_draws=1 is a degenerate but valid call."""
    signal, _ = signal_1d
    bkps = WBS(n_draws=1, seed=0).fit_predict(signal, n_bkps=1)
    assert bkps[-1] == signal.shape[0]


def test_2d_signal_shape(signal_5d):
    signal, _ = signal_5d
    bkps = WBS(seed=0).fit_predict(signal, n_bkps=2)
    assert bkps[-1] == signal.shape[0]


def test_predict_after_refit_uses_new_intervals():
    """Re-fitting should clear the lru_cache and re-draw intervals."""
    signal, _ = pw_constant(n_samples=100, n_bkps=2, noise_std=1, seed=5)
    algo = WBS(seed=99)
    bkps1 = algo.fit_predict(signal, n_bkps=2)
    signal2, _ = pw_constant(n_samples=100, n_bkps=2, noise_std=1, seed=6)
    bkps2 = algo.fit_predict(signal2, n_bkps=2)
    # Both should end at their respective n_samples
    assert bkps1[-1] == signal.shape[0]
    assert bkps2[-1] == signal2.shape[0]
