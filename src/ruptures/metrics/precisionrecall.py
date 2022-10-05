r"""Precision and recall."""
from itertools import product

from ruptures.metrics.sanity_check import sanity_check


def precision_recall(true_bkps, my_bkps, margin=10):
    """Calculate the precision/recall of an estimated segmentation compared
    with the true segmentation.

    Args:
        true_bkps (list): list of the last index of each regime (true
            partition).
        my_bkps (list): list of the last index of each regime (computed
            partition).
        margin (int, optional): allowed error (in points).

    Returns:
        tuple: (precision, recall)
    """
    sanity_check(true_bkps, my_bkps)
    assert margin > 0, "Margin of error must be positive (margin = {})".format(margin)

    if len(my_bkps) == 1:
        return 0, 0

    used = set()
    true_pos = set(
        true_b
        for true_b, my_b in product(true_bkps[:-1], my_bkps[:-1])
        if my_b - margin < true_b < my_b + margin
        and not (my_b in used or used.add(my_b))
    )

    tp_ = len(true_pos)
    precision = tp_ / (len(my_bkps) - 1)
    recall = tp_ / (len(true_bkps) - 1)
    return precision, recall
