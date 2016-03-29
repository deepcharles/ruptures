import numpy as np
import pytest
from ruptures.metrics import hamming, meantime, precision_recall
from ruptures.metrics import hausdorff, zero_one_loss
from ruptures.metrics.sanity_check import BadPartitions
METRICS = [hamming, meantime, precision_recall, zero_one_loss, hausdorff]


@pytest.fixture(scope="module")
def bkps():
    return [100, 200, 300, 400, 500], [101, 201, 301, 401, 500]


@pytest.mark.parametrize("metric", METRICS)
def test_simple(bkps, metric):
    true_bkps, my_bkps = bkps
    m = metric(true_bkps, my_bkps)
    try:
        assert m > 0, "Partitions are different, so the metric must be"
        "positive."
    except TypeError:
        pr, re = m
        assert pr > 0 and re > 0, "Partitions are different, so the metric"
        "must be positive."
    return m


@pytest.mark.parametrize("metric", METRICS)
def test_exception(bkps, metric):
    true_bkps, my_bkps = bkps
    with pytest.raises(BadPartitions):
        m = metric(true_bkps, [])
    with pytest.raises(BadPartitions):
        m = metric([], my_bkps)
    with pytest.raises(BadPartitions):
        m = metric([10, 10, 500], [10, 500])
    with pytest.raises(BadPartitions):
        m = metric([10, 500], [10, 501])


def test_pr_re(bkps):
    true_bkps, my_bkps = bkps

    # wrong margin
    with pytest.raises(AssertionError):
        precision_recall(true_bkps, my_bkps, 0)
        precision_recall(true_bkps, my_bkps, -1)

    # same partition
    for margin in range(1, 15):
        assert precision_recall(true_bkps, true_bkps, margin) == (1, 1)

    # with a delta
    for delta in range(2, 15):
        my_bkps = [b + delta for b in true_bkps[:-1]] + [true_bkps[-1]]
        # delta too small
        assert precision_recall(true_bkps, my_bkps,
                                margin=delta) == (0, 0)
        # delta too great
        assert precision_recall(true_bkps, my_bkps,
                                margin=delta + 1) == (1, 1)

    # test on a few values
    true_bkps = [100, 200, 300, 400, 500]
    my_bkps = list(range(2, 501, 2))
    pr, re = precision_recall(true_bkps, my_bkps, margin=10)
    assert pr == 4 / 249
    assert re == 1

    pr, re = precision_recall(my_bkps, true_bkps, margin=10)
    assert pr == 1
    assert re == 4 / 249


def test_zerooneloss(bkps):
    true_bkps, my_bkps = [100, 200, 300, 400, 500], [101, 201, 301, 401, 500]
    assert zero_one_loss(true_bkps, my_bkps)

    true_bkps, my_bkps = [100, 200, 300, 500], [101, 201, 301, 401, 500]
    assert not zero_one_loss(true_bkps, my_bkps)

    true_bkps, my_bkps = np.arange(10), np.arange(10)
    assert zero_one_loss(true_bkps, my_bkps)

    true_bkps, my_bkps = np.arange(100, step=9), np.arange(100)
    assert not zero_one_loss(true_bkps, my_bkps)


def test_hausdorff(bkps):
    true_bkps, my_bkps = [100, 200, 300, 400, 500], [101, 201, 301, 401, 500]
    assert hausdorff(true_bkps, my_bkps) == 1

    true_bkps, my_bkps = [100, 200, 300, 400, 500], [100, 200, 300, 405, 500]
    assert hausdorff(true_bkps, my_bkps) == 5

    true_bkps, my_bkps = [100, 200, 300, 500], [100, 200, 300, 405, 500]
    assert hausdorff(true_bkps, my_bkps) == 105

    assert hausdorff(true_bkps, true_bkps) == 0


def test_timeerror():
    true_bkps, my_bkps = [100, 200, 300, 400, 500], [101, 201, 301, 401, 500]
    assert meantime(true_bkps, my_bkps) == 1

    true_bkps, my_bkps = [100, 200, 300, 400, 500], [101, 201, 301, 401, 500]
    assert meantime(true_bkps, true_bkps) == 0

    true_bkps, my_bkps = [100, 200, 300, 400, 500], [104, 200, 300, 400, 500]
    assert meantime(true_bkps, my_bkps) == 1

    true_bkps, my_bkps = np.array([100, 200, 300, 400, 500]), np.array([
        101, 201, 301, 401, 500])
    assert meantime(true_bkps, my_bkps) == 1

    true_bkps, my_bkps = [400, 500], [100, 200, 300, 500]
    assert meantime(my_bkps, true_bkps) == 100

    true_bkps, my_bkps = [100, 200, 300, 500], [450, 500]
    assert meantime(my_bkps, true_bkps) == 250


def test_hamming(bkps):
    # TODO: faire des tests pour hamming
    true_bkps, my_bkps = bkps
    assert hamming(true_bkps, true_bkps) == 0
