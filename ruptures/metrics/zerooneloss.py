from ruptures.metrics.sanity_check import sanity_check


def zero_one_loss(bkps1, bkps2):
    """Zero-one loss: 1 if bkps have the same number of breakpoints, 0 if not.

    Args:
        bkps1 (list): list of the last index of each regime.
        bkps2 (list): list of the last index of each regime.

    Returns:
        int: 0 or 1
    """
    sanity_check(bkps1, bkps2)
    return int(len(bkps1) == len(bkps2))
