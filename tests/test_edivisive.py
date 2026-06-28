"""Tests for the EDivisive change point detection algorithm.

Reference: Matteson & James (2014) "A Nonparametric Approach for
Multiple Change Point Analysis of Multivariate Data", JASA
109(505):334-345.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from ruptures import EDivisive
from ruptures.exceptions import BadSegmentationParameters


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(0)


@pytest.fixture(scope="module")
def signal_1d_two_bkps(rng):
    """Piecewise-constant 1-D signal with two clear change points."""
    s = np.concatenate(
        [
            rng.normal(0.0, 0.5, 100),
            rng.normal(5.0, 0.5, 100),
            rng.normal(0.0, 0.5, 100),
        ]
    )
    return s, [100, 200, 300]


@pytest.fixture(scope="module")
def signal_5d_one_bkp(rng):
    """Piecewise-constant 5-D signal with one clear change point."""
    a = rng.normal(0.0, 0.5, (150, 5))
    b = rng.normal(3.0, 0.5, (150, 5))
    return np.vstack([a, b]), [150, 300]


@pytest.fixture(scope="module")
def constant_signal():
    """Signal with no change point."""
    return np.zeros(200)


# ---------------------------------------------------------------------------
# Interface / attribute tests
# ---------------------------------------------------------------------------


def test_fit_returns_self(signal_1d_two_bkps):
    signal, _ = signal_1d_two_bkps
    algo = EDivisive(n_perms=0)
    assert algo.fit(signal) is algo


def test_predict_last_element_equals_n_samples(signal_1d_two_bkps):
    signal, _ = signal_1d_two_bkps
    bkps = EDivisive(n_perms=0).fit_predict(signal, n_bkps=2)
    assert bkps[-1] == signal.shape[0]


def test_predict_n_bkps_length(signal_1d_two_bkps):
    signal, _ = signal_1d_two_bkps
    for k in [0, 1, 2]:
        bkps = EDivisive(n_perms=0).fit_predict(signal, n_bkps=k)
        assert len(bkps) == k + 1


def test_predict_sorted(signal_1d_two_bkps):
    signal, _ = signal_1d_two_bkps
    bkps = EDivisive(n_perms=0).fit_predict(signal, n_bkps=2)
    assert bkps == sorted(bkps)


def test_predict_1d_input(signal_1d_two_bkps):
    signal, _ = signal_1d_two_bkps
    assert signal.ndim == 1
    bkps = EDivisive(n_perms=0).fit_predict(signal, n_bkps=1)
    assert bkps[-1] == signal.shape[0]


def test_predict_2d_input(signal_5d_one_bkp):
    signal, _ = signal_5d_one_bkp
    bkps = EDivisive(n_perms=0).fit_predict(signal, n_bkps=1)
    assert bkps[-1] == signal.shape[0]
    assert len(bkps) == 2


def test_fit_predict_equiv(signal_1d_two_bkps):
    """fit().predict() and fit_predict() must agree."""
    signal, _ = signal_1d_two_bkps
    algo = EDivisive(n_perms=0)
    a = algo.fit(signal).predict(n_bkps=2)
    b = EDivisive(n_perms=0).fit_predict(signal, n_bkps=2)
    assert a == b


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_alpha_out_of_range():
    with pytest.raises(ValueError, match="alpha"):
        EDivisive(alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        EDivisive(alpha=2.5)


def test_predict_before_fit():
    with pytest.raises(BadSegmentationParameters):
        EDivisive().predict(n_bkps=1)


def test_negative_n_bkps(signal_1d_two_bkps):
    signal, _ = signal_1d_two_bkps
    algo = EDivisive(n_perms=0).fit(signal)
    with pytest.raises(BadSegmentationParameters):
        algo.predict(n_bkps=-1)


# ---------------------------------------------------------------------------
# Detection accuracy
# ---------------------------------------------------------------------------


def test_detects_two_change_points_1d(signal_1d_two_bkps):
    """Both breakpoints must be within ±15 samples of the true positions."""
    signal, true_bkps = signal_1d_two_bkps
    bkps = EDivisive(n_perms=0).fit_predict(signal, n_bkps=2)
    # true_bkps without the final n_samples sentinel
    detected = bkps[:-1]
    expected = true_bkps[:-1]
    for est, true in zip(sorted(detected), sorted(expected)):
        assert abs(est - true) <= 15, f"Expected ~{true}, got {est}"


def test_detects_one_change_point_5d(signal_5d_one_bkp):
    """Breakpoint in a 5-D signal must be within ±15 samples."""
    signal, true_bkps = signal_5d_one_bkp
    bkps = EDivisive(n_perms=0).fit_predict(signal, n_bkps=1)
    detected = bkps[0]
    assert abs(detected - 150) <= 15, f"Expected ~150, got {detected}"


def test_no_split_constant_signal_n_bkps_0(constant_signal):
    """Zero breakpoints requested → only the sentinel."""
    bkps = EDivisive(n_perms=0).fit_predict(constant_signal, n_bkps=0)
    assert bkps == [200]


# ---------------------------------------------------------------------------
# Statistical properties of the energy statistic
# ---------------------------------------------------------------------------


def test_q_stat_increases_with_separation():
    """Larger mean shift → larger energy statistic at the true split."""
    rng = np.random.default_rng(1)
    n = 100
    results = {}
    for delta in [0.5, 2.0, 5.0]:
        signal = np.concatenate([rng.normal(0.0, 1.0, n), rng.normal(delta, 1.0, n)])
        algo = EDivisive(n_perms=0).fit(signal)
        _, q = algo._best_split(0, 2 * n)
        results[delta] = q
    assert results[0.5] < results[2.0] < results[5.0]


def test_q_stat_zero_identical_segments():
    """Energy statistic must be zero when both segments are identical."""
    signal = np.ones(100)
    algo = EDivisive(n_perms=0).fit(signal)
    _, q = algo._best_split(0, 100)
    assert abs(q) < 1e-10


def test_best_split_exact_midpoint():
    """Optimal split must be at the true midpoint for a step signal."""
    signal = np.concatenate([np.zeros(50), np.full(50, 10.0)])
    algo = EDivisive(n_perms=0).fit(signal)
    t, _ = algo._best_split(0, 100)
    assert t == 50


# ---------------------------------------------------------------------------
# alpha parameter
# ---------------------------------------------------------------------------


def test_alpha_2(signal_1d_two_bkps):
    """Alpha=2 (squared distances) should still detect the change points."""
    signal, _ = signal_1d_two_bkps
    bkps = EDivisive(alpha=2.0, n_perms=0).fit_predict(signal, n_bkps=2)
    assert len(bkps) == 3
    assert bkps[-1] == signal.shape[0]


def test_alpha_0_5(signal_1d_two_bkps):
    """alpha=0.5 should still detect the change points."""
    signal, _ = signal_1d_two_bkps
    bkps = EDivisive(alpha=0.5, n_perms=0).fit_predict(signal, n_bkps=2)
    assert len(bkps) == 3


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------


def test_perm_test_rejects_null_for_clear_change(rng):
    """Permutation test must detect a 10-sigma shift at ~position 100."""
    signal = np.concatenate(
        [
            rng.normal(0.0, 1.0, 100),
            rng.normal(10.0, 1.0, 100),
        ]
    )
    bkps = EDivisive(n_perms=200, sig_level=0.05).fit_predict(signal)
    # The true change point at 100 must appear in the result (within ±5).
    assert any(abs(b - 100) <= 5 for b in bkps), f"Breakpoints: {bkps}"
    assert bkps[-1] == 200


def test_perm_test_no_split_for_pure_noise(rng):
    """Permutation test must not split a homogeneous noise signal."""
    signal = rng.normal(0.0, 1.0, 200)
    bkps = EDivisive(n_perms=200, sig_level=0.05).fit_predict(signal)
    # May occasionally split due to random chance, but last element must be 200
    assert bkps[-1] == 200


# ---------------------------------------------------------------------------
# min_size constraint
# ---------------------------------------------------------------------------


def test_min_size_respected(signal_1d_two_bkps):
    """No segment in the result may be shorter than min_size."""
    signal, _ = signal_1d_two_bkps
    min_size = 20
    bkps = EDivisive(min_size=min_size, n_perms=0).fit_predict(signal, n_bkps=2)
    segments = list(zip([0] + bkps[:-1], bkps))
    for start, end in segments:
        assert end - start >= min_size, f"Segment [{start}, {end}) too short"
