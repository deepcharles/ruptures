"""Hamming metric for segmentation."""
from ruptures.metrics.randindex import randindex


def hamming(bkps1, bkps2):
    """Modified Hamming distance for partitions. For all pair of points (x, y),
    x != y, the functions computes the number of times the two partitions
    disagree. The result is scaled to be within 0 and 1.

    Args:
        bkps1 (list): list of the last index of each regime.
        bkps2 (list): list of the last index of each regime.

    Returns:
        float: Hamming distance.
    """
    return 1 - randindex(bkps1=bkps1, bkps2=bkps2)
