import numpy as np
import pytest

import ruptures as rpt


@pytest.mark.parametrize("n_features", [1, 2])
@pytest.mark.parametrize(
    "epsilon, n_segments",
    [(1, 4), (3.99, 4), (4, 3), (7.99, 3), (8, 2), (9, 2), (208, 1), (209, 1)],
)
def test_bottomup_reconstruction_budget(n_features, epsilon, n_segments):
    signal = np.array([0, 0, 2, 2, 10, 10, 12, 12], dtype=float)
    if n_features == 2:
        signal = np.column_stack((signal, np.zeros_like(signal)))
    algo = rpt.BottomUp(model="l2", min_size=2, jump=1).fit(signal)

    bkps = algo.predict(epsilon=epsilon)
    residual = sum(
        np.square(segment - segment.mean(axis=0)).sum()
        for segment in np.split(signal, bkps[:-1])
    )

    assert residual <= epsilon
    assert len(bkps) == n_segments
    assert bkps == sorted(set(bkps))
    assert bkps[-1] == len(signal)
    assert algo.predict(epsilon=epsilon) == bkps


def test_bottomup_budget_includes_existing_residual():
    signal = np.array([0, 2, 0, 2, 10, 12, 10, 12], dtype=float)
    bkps = rpt.BottomUp(model="l2", min_size=2, jump=1).fit_predict(signal, epsilon=9)
    assert bkps == [4, 8]


@pytest.mark.parametrize(
    "stopping_rule, expected",
    [
        ({"n_bkps": 1}, [4, 8]),
        ({"pen": 1}, [2, 4, 6, 8]),
        ({"pen": 5}, [4, 8]),
        ({"epsilon": 209}, [8]),
    ],
)
def test_bottomup_stopping_rule_controls(stopping_rule, expected):
    signal = np.array([0, 0, 2, 2, 10, 10, 12, 12], dtype=float)
    bkps = rpt.BottomUp(model="l2", min_size=2, jump=1).fit_predict(
        signal, **stopping_rule
    )
    assert bkps == expected


def test_bottomup_budget_single_initial_segment():
    signal = np.array([1.0, 1.0])
    assert rpt.BottomUp(min_size=2, jump=1).fit_predict(signal, epsilon=1) == [2]
