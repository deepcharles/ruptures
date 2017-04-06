"""Cost functions for piecewise constant fonctions."""
import numpy as np
from ruptures.costs import NotEnoughPoints


def constantl2(signal):
    """Return the L2 cost.

    L2 norm of the residuals when approximating with a constant.

    Args:
        signal (array): signal (n_samples, n_features)

    Returns:
        float: L2 cost

    Raises:
        NotEnoughPoints: if there less than 2 samples
    """
    n_samples, _ = signal.shape
    if n_samples < 2:
        raise NotEnoughPoints
    cost = signal.var(axis=0).sum() * n_samples
    return cost


def constantl1(signal):
    """Return the L1 cost.

    L1 norm of the residuals when approximating with a constant.

    Args:
        signal (array): signal (n_samples, n_features)

    Returns:
        float: L1 cost

    Raises:
        NotEnoughPoints: if there less than 2 samples
    """
    n_samples, _ = signal.shape
    if n_samples < 2:
        raise NotEnoughPoints
    med = np.median(signal, axis=0)
    cost = abs(signal - med).sum()
    return cost
