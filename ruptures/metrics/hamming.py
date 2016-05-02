from ruptures.metrics.sanity_check import sanity_check
from scipy.sparse import block_diag
import numpy as np


def membership_mat(bkps):
    return block_diag([np.ones((k, k)) for k in [bkps[0]] +
                       list(np.diff(bkps))], dtype=bool)


def hamming(bkps1, bkps2):
    """Modified Hamming distance for partitions.
        For all pair of points (x, y), x != y, the functions computes the
        number of times the two partitions disagree.
        The result is scaled to be within 0 and 1.

    Args:
        bkps1 (list): list of the last index of each regime.
        bkps2 (list): list of the last index of each regime.

    Returns:
        float: Hamming distance.
    """
    sanity_check(bkps1, bkps2)
    n = max(bkps1)

    disagreement = abs(membership_mat(bkps1) - membership_mat(bkps2)).sum()
    disagreement /= n * (n - 1)  # scaling
    return disagreement
