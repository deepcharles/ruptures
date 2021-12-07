r"""Rand index (`randindex`)"""
from ruptures.metrics import hamming
from ruptures.metrics.sanity_check import sanity_check


def randindex(bkps1, bkps2):
    """Rand index for two partitions. The result is scaled to be within 0 and
    1.

    Args:
        bkps1 (list): list of the last index of each regime.
        bkps2 (list): list of the last index of each regime.

    Returns:
        float: Rand index
    """
    return 1 - hamming(bkps1, bkps2)


def randindex_cpd(bkps1, bkps2):
    """Computes efficiently the Rand index for change-point detection given two
    sorted partitions.

    Args:
        bkps1 (list): sorted list of the last index of each regime.
        bkps2 (list): sorted list of the last index of each regime.

    Returns:
        float: Rand index
    """
    sanity_check(bkps1, bkps2)
    n_samples = max(bkps1)
    disagreement = 0
    bkps1.insert(0, 0)
    bkps2.insert(0, 0)

    beginj = 0  # avoids unnecessary computations
    for i in range(len(bkps1) - 1):
        for j in range(beginj, len(bkps2) - 1):
            nij = min(bkps1[i + 1], bkps2[j + 1])
            nij -= max(bkps1[i], bkps2[j])
            nij = max(nij, 0)
            disagreement += nij * abs(bkps1[i + 1] - bkps2[j + 1])

            # we can skip the rest of the iteration, nij will be 0
            if bkps1[i + 1] < bkps2[j + 1]:
                break

            else:
                beginj = j + 1

    bkps1.pop(0)
    bkps2.pop(0)
    disagreement /= n_samples * (n_samples - 1) / 2
    return 1.0 - disagreement
