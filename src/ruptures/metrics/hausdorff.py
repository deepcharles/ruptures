r"""Hausdorff metric."""

import numpy as np
from scipy.spatial.distance import cdist
from ruptures.metrics.sanity_check import sanity_check


def hausdorff(bkps1, bkps2):
    """Compute the Hausdorff distance between changepoints.

    Args:
        bkps1 (list): list of the last index of each regime.
        bkps2 (list): list of the last index of each regime.

    Returns:
        float: Hausdorff distance.  Returns ``np.inf`` when one partition
        has no intermediate changepoints (i.e. it predicts a single segment)
        and the other does.
    """
    sanity_check(bkps1, bkps2)
    bkps1_arr = np.array(bkps1[:-1]).reshape(-1, 1)
    bkps2_arr = np.array(bkps2[:-1]).reshape(-1, 1)
    if bkps1_arr.size == 0 or bkps2_arr.size == 0:
        # If one partition has no intermediate changepoints and the other does,
        # the sets are not comparable; return infinity by convention.
        if bkps1_arr.size == 0 and bkps2_arr.size == 0:
            return 0.0
        return np.inf
    pw_dist = cdist(bkps1_arr, bkps2_arr)
    res = max(pw_dist.min(axis=0).max(), pw_dist.min(axis=1).max())
    return res
