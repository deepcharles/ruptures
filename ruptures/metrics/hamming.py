"""Hamming metric for segmentation."""

import numpy as np
from scipy.sparse import block_diag, triu

from ruptures.metrics.sanity_check import sanity_check
from ruptures.utils import pairwise


def membership_mat(bkps):
    """Return membership matrix for the given segmentation."""
    n_samples = bkps[-1]
    m_mat = np.zeros((n_samples, n_samples))
    for start, end in pairwise([0] + bkps):
        m_mat[start:end, start:end] = 1
    return m_mat


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
    n_samples = max(bkps1)

    disagreement = abs(membership_mat(bkps1) - membership_mat(bkps2))
    disagreement = triu(disagreement, k=1).sum() * 1.
    disagreement /= n_samples * n_samples / 2  # scaling
    return disagreement
