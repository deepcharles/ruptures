r"""Kernelized mean change"""
import numpy as np
from scipy.spatial.distance import pdist, squareform

from ruptures.exceptions import NotEnoughPoints
from ruptures.base import BaseCost


class CostRbf(BaseCost):

    r"""Kernel cost function (rbf kernel)."""

    model = "rbf"

    def __init__(self):
        """Initialize the object."""
        self.gram = None
        self.min_size = 2

    def fit(self, signal) -> "CostRbf":
        """Sets parameters of the instance.

        Args:
            signal (array): signal. Shape (n_samples,) or (n_samples, n_features)

        Returns:
            self
        """
        if signal.ndim == 1:
            K = pdist(signal.reshape(-1, 1), metric="sqeuclidean")
        else:
            K = pdist(signal, metric="sqeuclidean")
        K_median = np.median(K)
        if K_median != 0:
            K /= K_median
        np.clip(K, 1e-2, 1e2, K)
        self.gram = np.exp(squareform(-K))
        return self

    def error(self, start, end) -> float:
        """Return the approximation cost on the segment [start:end].

        Args:
            start (int): start of the segment
            end (int): end of the segment

        Returns:
            segment cost

        Raises:
            NotEnoughPoints: when the segment is too short (less than `min_size` samples).
        """
        if end - start < self.min_size:
            raise NotEnoughPoints
        sub_gram = self.gram[start:end, start:end]
        val = np.diagonal(sub_gram).sum()
        val -= sub_gram.sum() / (end - start)
        return val
