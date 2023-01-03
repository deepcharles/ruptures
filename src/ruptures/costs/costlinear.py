r"""Linear model change."""
from numpy.linalg import lstsq

from ruptures.base import BaseCost
from ruptures.costs import NotEnoughPoints


class CostLinear(BaseCost):
    r"""Least-square estimate for linear changes."""

    model = "linear"

    def __init__(self):
        """Initialize the object."""
        self.signal = None
        self.covar = None
        self.min_size = 2

    def fit(self, signal) -> "CostLinear":
        """Set parameters of the instance. The first column contains the
        observed variable. The other columns contains the covariates.

        Args:
            signal (array): signal of shape (n_samples, n_regressors+1)

        Returns:
            self
        """
        assert signal.ndim > 1, "Not enough dimensions"

        self.signal = signal[:, 0].reshape(-1, 1)
        self.covar = signal[:, 1:]
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
        y, X = self.signal[start:end], self.covar[start:end]
        _, residual, _, _ = lstsq(X, y, rcond=None)
        return residual.sum()
