from itertools import product


def precision_recall(true_bkps, my_bkps, margin=10):
    """Calculates the precision/recall of a segmentation compared with the true
        segmentation.
        A true changepoint is declared "detected" (or positive) if there is at
        least one computed changepoint at less than margin points from it.

    Args:
        true_bkps (list): list of the last index of each regime (true
            partition).
        my_bkps (list): list of the last index of each regime (computed
            partition).
        margin (int, optional): allowed error (in points).

    Returns:
        tuple: (precision, recall)
    """
    true_pos = set(true_b
                   for true_b, my_b in product(true_bkps, my_bkps)
                   if my_b - margin < true_b < my_b + margin)
    precision = len(true_pos) / len(my_bkps)
    recall = len(true_pos) / len(true_bkps)
    return precision, recall
