from itertools import product
from ruptures.metrics.sanity_check import sanity_check


def precision_recall(true_bkps, my_bkps, margin=10):
    """Calculates the precision/recall of a segmentation compared with the true
        segmentation.
        A true changepoint is declared "detected" (or positive) if there is at
        least one computed changepoint at (strictly) less than margin points
        from it.

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
    assert margin > 0, "Margin of error must be positive (margin = {})".format(
        margin)

    if len(my_bkps) == 1:
        return 0, 0

    used = set()
    true_pos = set(true_b
                   for true_b, my_b in product(true_bkps[:-1], my_bkps[:-1])
                   if my_b - margin < true_b < my_b + margin and
                   not (my_b in used or used.add(my_b)))
    precision = len(true_pos) / (len(my_bkps) - 1)
    recall = len(true_pos) / (len(true_bkps) - 1)
    return precision, recall
