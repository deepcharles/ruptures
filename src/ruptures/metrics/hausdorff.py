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
        float: Hausdorff distance.
    """
    sanity_check(bkps1, bkps2)
    bkps1_arr = np.array(bkps1[:-1]).reshape(-1, 1)
    bkps2_arr = np.array(bkps2[:-1]).reshape(-1, 1)
    pw_dist = cdist(bkps1_arr, bkps2_arr)
    res = max(pw_dist.min(axis=0).max(), pw_dist.min(axis=1).max())
    return res
