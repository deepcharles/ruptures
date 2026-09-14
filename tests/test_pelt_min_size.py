"""Regression tests for PELT pruning with minimum segment lengths."""

from itertools import combinations, product

import numpy as np
import pytest

from ruptures import Pelt


@pytest.mark.parametrize(
    "values,minimum,jump,penalty,expected",
    [
        pytest.param(
            [0, 9, 2, 8, 3, 0, 10], 3, 1, 0.1, [3, 7], id="rounding-sensitive"
        ),
        pytest.param(
            [3, 8, 7, 0, 4, 9, 6, 0, 8, 8], 3, 1, 0.1, [5, 10], id="multiple-splits"
        ),
        pytest.param(
            [0, 0, 1, 1, 2, 2, 2, 0], 3, 1, 0.1, [3, 8], id="strict-minimum-three"
        ),
        pytest.param([0, 1, 0, 0, 1], 2, 1, 0.1, [5], id="strict-minimum-two"),
        pytest.param([0, 1, 0, 0, 1], 2, 2, 0.1, [5], id="off-grid-terminal"),
    ],
)
def test_keep_start_until_replacement_is_legal(
    values, minimum, jump, penalty, expected
):
    """Retain starts whose dominating prefix cannot yet start a legal
    segment."""
    algo = Pelt(model="l2", min_size=minimum, jump=jump)
    assert algo.fit_predict(np.array(values, dtype=float), pen=penalty) == expected


def test_off_grid_final_endpoint_with_default_settings():
    """Handle a short terminal grid interval even when min_size is below
    jump."""
    signal = np.array([0] * 5 + [1] * 5 + [0] * 5 + [3], dtype=float)
    assert Pelt(model="l2").fit_predict(signal, pen=1) == [5, 16]


@pytest.mark.parametrize(
    "minimum,jump,n", [(1, 1, 1), (3, 5, 3), (3, 1, 4), (10, 100, 10)]
)
def test_only_one_segment_is_feasible(minimum, jump, n):
    """Keep the unsplit candidate when no internal boundary is feasible."""
    assert Pelt(min_size=minimum, jump=jump).fit_predict(np.zeros(n), pen=1) == [n]


def test_repeated_penalties_and_refit():
    """Keep pruning state local to each prediction and allow refitting."""
    algo = Pelt(min_size=3, jump=1).fit(np.array([0, 0, 1, 1, 2, 2, 2, 0]))
    assert algo.predict(0.1) == [3, 8]
    assert algo.predict(1000) == [8]
    assert algo.predict(0.1) == [3, 8]
    assert algo.fit_predict(np.zeros((8, 2)), 0.1) == [8]


@pytest.mark.parametrize(
    "minimum,jump,dimensions", list(product(range(1, 5), [1, 2, 3, 5], [1, 3]))
)
def test_l2_objective_matches_exhaustive_enumeration(minimum, jump, dimensions):
    """Compare PELT with every feasible partition, including off-grid ends."""
    rng = np.random.default_rng(20260914)
    for n in (5, 8, 11):
        grid = list(range(jump, n, jump))
        partitions = []
        for count in range(len(grid) + 1):
            for internal in combinations(grid, count):
                ends = internal + (n,)
                starts = (0,) + internal
                if all(end - start >= minimum for start, end in zip(starts, ends)):
                    partitions.append(ends)

        signals = [np.zeros((n, dimensions))]
        signals.extend(rng.integers(-2, 3, size=(4, n, dimensions)))
        for signal in signals:
            # Compute residual SSE directly, independently of CostL2.error.
            costs = {
                (start, end): np.sum(
                    (signal[start:end] - signal[start:end].mean(axis=0)) ** 2
                )
                for start in range(n)
                for end in range(start + minimum, n + 1)
            }

            def objective(ends, penalty):
                """Return squared error plus a penalty per internal
                boundary."""
                starts = (0,) + tuple(ends[:-1])
                return sum(
                    costs[start, end] for start, end in zip(starts, ends)
                ) + penalty * (len(ends) - 1)

            algo = Pelt(model="l2", min_size=minimum, jump=jump).fit(signal)
            for penalty in (0.1, 1.0):
                actual = tuple(algo.predict(penalty))
                assert actual in partitions
                expected = min(objective(ends, penalty) for ends in partitions)
                assert objective(actual, penalty) == pytest.approx(
                    expected, rel=1e-12, abs=1e-12
                )
