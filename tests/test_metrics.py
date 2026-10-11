import pytest

from ruptures.metrics import (
    hamming,
    hausdorff,
    meantime,
    precision_recall,
    randindex,
)
from ruptures.metrics.sanity_check import BadPartitions


@pytest.fixture(scope="module")
def b_mb():
    return [100, 200, 350, 400, 500], [101, 201, 301, 401, 500]


def test_hausdorff(b_mb):
    b, mb = b_mb
    m = hausdorff(b, mb)
    assert m > 0
    m = hausdorff(b, b)
    assert m == 0


def test_randindex(b_mb):
    b, mb = b_mb
    m = randindex(b, mb)
    assert 1 > m > 0
    m = randindex(b, b)
    assert m == 1


def test_meantime(b_mb):
    b, mb = b_mb
    m = meantime(b, mb)
    assert m > 0
    m = meantime(b, b)
    assert m == 0


@pytest.mark.parametrize("margin", range(1, 20, 2))
def test_precision_recall(b_mb, margin):
    b, mb = b_mb
    p, r = precision_recall(b, mb, margin=margin)
    assert 0 <= p < 1
    assert 0 <= r < 1
    p, r = precision_recall(b, b, margin=margin)
    assert (p, r) == (1, 1)
    p, r = precision_recall(b, [b[-1]], margin=margin)


@pytest.mark.parametrize(
    "metric", [hamming, hausdorff, meantime, precision_recall, randindex]
)
def test_exception(b_mb, metric):
    true_bkps, my_bkps = b_mb
    with pytest.raises(BadPartitions):
        m = metric(true_bkps, [])
    with pytest.raises(BadPartitions):
        m = metric([], my_bkps)
    with pytest.raises(BadPartitions):
        m = metric([10, 10, 500], [10, 500])
    with pytest.raises(BadPartitions):
        m = metric([10, 500], [10, 501])


@pytest.mark.parametrize(
    "true_bkps,my_bkps,margin,precision,recall",
    [
        ([10, 20, 100], [10, 20, 100], 11, 1, 1),
        ([10, 20, 100], [5, 14, 100], 7, 1, 1),
        ([10, 20, 30, 100], [5, 14, 24, 100], 7, 1, 1),
        ([10, 20, 100], [9, 11, 20, 100], 3, 2 / 3, 1),
        ([10, 20, 100], [10, 20, 100], 3, 1, 1),
        ([10, 100], [9, 11, 100], 3, 1 / 2, 1),
        ([10, 20, 100], [14, 100], 7, 1, 1 / 2),
        ([10, 20, 100], [1, 30, 100], 3, 0, 0),
        ([10, 20, 100], [100], 3, 0, 0),
        ([10, 100], [13, 100], 3, 0, 0),
        ([10, 100], [7, 100], 3, 0, 0),
        ([10, 100], [12, 100], 3, 1, 1),
        ([10, 100], [8, 100], 3, 1, 1),
    ],
)
def test_precision_recall_unique_matches(true_bkps, my_bkps, margin, precision, recall):
    assert precision_recall(true_bkps, my_bkps, margin=margin) == pytest.approx(
        (precision, recall)
    )
