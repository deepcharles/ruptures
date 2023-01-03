r"""CostCosine (kernel change point detection with the cosine similarity)"""
import numpy as np
from ruptures.base import BaseCost
from ruptures.costs import NotEnoughPoints
from scipy.spatial.distance import pdist, squareform


class CostCosine(BaseCost):
    r"""Kernel change point detection with the cosine similarity."""

    model = "cosine"

    def __init__(self):
        """Initialize the object."""
        self.signal = None
        self.min_size = 1
        self._gram = None

    @property
    def gram(self):
        """Generate the gram matrix (lazy loading).

        Only access this function after a `.fit()` (otherwise
        `self.signal` is not defined).
        """
        if self._gram is None:
            self._gram = squareform(1 - pdist(self.signal, metric="cosine"))
        return self._gram

    def fit(self, signal) -> "CostCosine":
        """Set parameters of the instance.

        Args:
            signal (array): array of shape (n_samples,) or (n_samples, n_features)

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
        sub_gram = self.gram[start:end, start:end]
        val = np.diagonal(sub_gram).sum()
        val -= sub_gram.sum() / (end - start)
        return val
