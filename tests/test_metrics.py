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


def test_hausdorff_empty_intermediate_bkps():
    """Hausdorff must not crash when a partition has no intermediate
    breakpoints."""
    # Both no intermediate breakpoints -> distance 0
    assert hausdorff([500], [500]) == 0.0
    # One side empty, the other not -> infinity
    import math

    assert math.isinf(hausdorff([500], [200, 500]))
    assert math.isinf(hausdorff([200, 500], [500]))


def test_meantime_empty_intermediate_bkps():
    """Meantime must not crash when a partition has no intermediate
    breakpoints."""
    # Both no intermediate breakpoints -> distance 0
    assert meantime([500], [500]) == 0.0
    # One side empty -> infinity
    import math

    assert math.isinf(meantime([500], [200, 500]))
    assert math.isinf(meantime([200, 500], [500]))


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
