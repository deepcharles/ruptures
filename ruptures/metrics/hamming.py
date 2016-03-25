from itertools import product
from ruptures.metrics.sanity_check import sanity_check


def in_same_cluster(bkps):
    bkps_sorted = sorted(bkps)

    def res_func(x, y):
        return all(not min(x, y) < b <= max(x, y) for b in bkps_sorted)
    return res_func


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
    sanity_check(bkps1, bkps1)
    n = max(bkps1)
    membership1 = in_same_cluster(bkps1)
    membership2 = in_same_cluster(bkps2)
    disagreement = sum(membership1(i, j) != membership2(i, j)
                       for i, j in product(range(n), range(n))
                       if i != j)
    disagreement /= n * (n - 1)  # scaling
    return disagreement
