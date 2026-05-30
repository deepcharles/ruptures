r"""Adjusted Rand index (`adjusted_randindex`)"""
import numpy as np
from ruptures.metrics.sanity_check import sanity_check
from sklearn.metrics import adjusted_rand_score


def chpt_to_label(bkps):
    """Return the segment index each sample belongs to.

    Example:
    -------
    >>> chpt_to_label([4, 10])
    array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    """
    duration = np.diff([0] + bkps)
    return np.repeat(np.arange(len(bkps)), duration)


def adjusted_randindex(bkps1, bkps2):
    """Compute the adjusted Rand index (between -0.5 and 1.) between two
    segmentations.

    The Rand index (RI) measures the similarity between two segmentations and
    is equal to the proportion of aggreement between two partitions.

    The metric implemented here is RI variant, adjusted for chance, and based
    on [scikit-learn's implementation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html).

    Args:
    ----
        bkps1 (list): sorted list of the last index of each regime.
        bkps2 (list): sorted list of the last index of each regime.

    Return:
    ------
        float: Adjusted Rand index
    """  # noqa E501
    sanity_check(bkps1, bkps2)
    label1 = chpt_to_label(bkps1)
    label2 = chpt_to_label(bkps2)
    return adjusted_rand_score(label1, label2)
