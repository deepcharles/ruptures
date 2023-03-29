r"""Kernelized mean change."""
import numpy as np
from scipy.spatial.distance import pdist, squareform

from ruptures.exceptions import NotEnoughPoints
from ruptures.base import BaseCost


class CostRbf(BaseCost):
    r"""Kernel cost function (rbf kernel)."""

    model = "rbf"

    def __init__(self, gamma=None):
        """Initialize the object."""
        self.min_size = 1
        self.gamma = gamma
        self._gram = None

    @property
    def gram(self):
        """Generate the gram matrix (lazy loading).

        Only access this function after a `.fit()` (otherwise
        `self.signal` is not defined).
        """
        if self._gram is None:
            K = pdist(self.signal, metric="sqeuclidean")
            if self.gamma is None:
                self.gamma = 1.0
                # median heuristics
                K_median = np.median(K)
                if K_median != 0:
                    # K /= K_median
                    self.gamma = 1 / K_median
            K *= self.gamma
            np.clip(K, 1e-2, 1e2, K)  # clipping to avoid exponential under/overflow
            self._gram = np.exp(squareform(-K))
        return self._gram

    def fit(self, signal) -> "CostRbf":
        """Sets parameters of the instance.

        Args:
            signal (array): signal. Shape (n_samples,) or (n_samples, n_features)

        Returns:
            self
        """
        if signal.ndim == 1:
            self.signal = signal.reshape(-1, 1)
        else:
            self.signal = signal

        # If gamma is none, set it using the median heuristic.
        # This heuristic involves computing the gram matrix which is lazy loaded
        # so we simply access the `.gram` property
        if self.gamma is None:
            self.gram

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
