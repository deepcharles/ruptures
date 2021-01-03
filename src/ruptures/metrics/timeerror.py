"""Mean time error."""
import numpy as np
from scipy.spatial.distance import cdist

from ruptures.metrics.sanity_check import sanity_check


def meantime(true_bkps, my_bkps):
    """For each computed changepoint, the mean time error is the average number
    of points to the closest true changepoint. Not a symetric funtion.

    Args:
        true_bkps (list): list of the last index of each regime (true
            partition).
        my_bkps (list): list of the last index of each regime (computed
            partition)

    Returns:
        float: mean time error.
    """
    sanity_check(true_bkps, my_bkps)
    true_bkps_arr = np.array(true_bkps[:-1]).reshape(-1, 1)
    my_bkps_arr = np.array(my_bkps[:-1]).reshape(-1, 1)
    pw_dist = cdist(true_bkps_arr, my_bkps_arr)

    dist_from_true = pw_dist.min(axis=0)
    assert len(dist_from_true) == len(my_bkps) - 1

    return dist_from_true.mean()
