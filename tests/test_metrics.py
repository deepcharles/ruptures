import numpy as np
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
    "dtype,n_samples",
    [
        (int, 50000),
        (np.int32, 50000),
        (np.uint32, 100000),
        (np.int64, 4000000000),
        (np.uint64, 10000000000),
    ],
)
def test_randindex_integer_breakpoints(dtype, n_samples):
    # Two equal segments versus one: only within-segment pairs agree.
    expected = (n_samples - 2) / (2 * (n_samples - 1))
    split = [dtype(n_samples // 2), dtype(n_samples)]
    whole = [dtype(n_samples)]
    assert randindex(split, whole) == pytest.approx(expected)
    assert randindex(whole, split) == pytest.approx(expected)
    assert randindex(split, split) == 1.0
