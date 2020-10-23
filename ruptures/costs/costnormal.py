r"""Gaussian process changes (CostNormal)"""
import numpy as np
from numpy.linalg import slogdet

from ruptures.base import BaseCost
from ruptures.costs import NotEnoughPoints


class CostNormal(BaseCost):

    """Gaussian process change."""

    model = "normal"

    def __init__(self):
        """Initialize the object."""
        self.signal = None
        self.min_size = 2

    def fit(self, signal) -> "CostNormal":
        """Set parameters of the instance.

        Args:
            signal (array): signal. Shape (n_samples,) or (n_samples, n_features)

        Returns:
            self
        """
        if signal.ndim == 1:
            self.signal = signal.reshape(-1, 1)
        else:
            self.signal = signal

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
        sub = self.signal[start:end]

        if self.signal.shape[1] > 1:
            cov = np.cov(sub.T)
        else:
            cov = np.array([[sub.var()]])
        _, val = slogdet(cov)
        return val * (end - start)
