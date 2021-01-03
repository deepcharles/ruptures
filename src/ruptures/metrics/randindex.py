r"""Rand index (`randindex`)"""
from ruptures.metrics import hamming


def randindex(bkps1, bkps2):
    """Rand index for two partitions. The result is scaled to be within 0 and
    1.

    Args:
        bkps1 (list): list of the last index of each regime.
        bkps2 (list): list of the last index of each regime.

    Returns:
        float: Rand index
    """
    return 1 - hamming(bkps1, bkps2)
