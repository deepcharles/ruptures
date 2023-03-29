r"""Gaussian process changes (CostNormal)"""
import warnings

import numpy as np
from numpy.linalg import slogdet
from ruptures.base import BaseCost
from ruptures.costs import NotEnoughPoints


class CostNormal(BaseCost):
    """Gaussian process change."""

    model = "normal"

    def __init__(self, add_small_diag=True):
        """Initialize the object.

        Args:
            add_small_diag (bool, optional): For signals with truly constant
                segments, the covariance matrix is badly conditioned, so we add
                a small diagonal matrix. Defaults to True.
        """
        self.signal = None
        self.min_size = 2
        self.add_small_diag = add_small_diag
        if add_small_diag:
            warnings.warn(
                "New behaviour in v1.1.5: "
                "a small bias is added to the covariance matrix to cope with truly "
                "constant segments (see PR#198).",
                UserWarning,
            )

    def fit(self, signal) -> "CostNormal":
        """Set parameters of the instance.

        Args:
            signal (array): signal of shape (n_samples,) or
                (n_samples, n_features)

        Returns:
            self
        """
        if signal.ndim == 1:
            self.signal = signal.reshape(-1, 1)
        else:
            self.signal = signal
        self.n_samples, self.n_dims = self.signal.shape
        return self

    def error(self, start, end) -> float:
        """Return the approximation cost on the segment [start:end].

        Args:
            start (int): start of the segment
            end (int): end of the segment

        Returns:
            segment cost

        Raises:
            NotEnoughPoints: when the segment is too short (less than `min_size`
                samples).
        """
        if end - start < self.min_size:
            raise NotEnoughPoints
        sub = self.signal[start:end]

        if self.signal.shape[1] > 1:
            cov = np.cov(sub.T)
        else:
            cov = np.array([[sub.var()]])
        if self.add_small_diag:  # adding small bias
            cov += 1e-6 * np.eye(self.n_dims)
        _, val = slogdet(cov)
        return val * (end - start)
