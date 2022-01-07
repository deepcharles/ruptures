r"""Rand index (`randindex`)"""
from ruptures.metrics.sanity_check import sanity_check


def randindex(bkps1, bkps2):
    """Computes the Rand index (between 0 and 1) between two segmentations.

    The Rand index (RI) measures the similarity between two segmentations and
    is equal to the proportion of aggreement between two partitions.

    RI is between 0 (total disagreement) and 1 (total agreement).
    This function uses the efficient implementation of [1].

    [1] Prates, L. (2021). A more efficient algorithm to compute the Rand Index for
    change-point problems. ArXiv:2112.03738.

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
    n_bkps1 = len(bkps1)
    n_bkps2 = len(bkps2)

    disagreement = 0
    beginj: int = 0  # avoids unnecessary computations
    for index_bkps1 in range(n_bkps1):
        start1: int = bkps1_with_0[index_bkps1]
        end1: int = bkps1_with_0[index_bkps1 + 1]
        for index_bkps2 in range(beginj, n_bkps2):
            start2: int = bkps2_with_0[index_bkps2]
            end2: int = bkps2_with_0[index_bkps2 + 1]
            nij = max(min(end1, end2) - max(start1, start2), 0)
            disagreement += nij * abs(end1 - end2)

            # we can skip the rest of the iteration, nij will be 0
            if end1 < end2:
                break
            else:
                beginj = index_bkps2 + 1

    disagreement /= n_samples * (n_samples - 1) / 2
    return 1.0 - disagreement
