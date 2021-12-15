r"""Rand index (`randindex`)"""
from ruptures.metrics.sanity_check import sanity_check
from ruptures.utils import pairwise


def randindex(bkps1, bkps2):
    """Computes efficiently the Rand index for change-point detection given two
    sorted partitions.

    Args:
        bkps1 (list): sorted list of the last index of each regime.
        bkps2 (list): sorted list of the last index of each regime.

    Returns:
        float: Rand index
    """
    sanity_check(bkps1, bkps2)
    n_samples = bkps1[-1]
    bkps1_with_0 = [0] + bkps1
    bkps2_with_0 = [0] + bkps2

    disagreement = 0
    beginj = 0  # avoids unnecessary computations
    for (start1, end1) in pairwise(bkps1_with_0):
        for (start2, end2) in pairwise(bkps2_with_0[beginj:]):
            nij = max(min(end1, end2)-max(start1, start2), 0)
            disagreement += nij * abs(end1 - end2)

            # we can skip the rest of the iteration, nij will be 0
            if end1 < end2:
                break
            else:
                beginj += 1

    disagreement /= n_samples * (n_samples - 1) / 2
    return 1.0 - disagreement
