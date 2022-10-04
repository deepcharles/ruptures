r"""Rank-based cost function (CostRank)"""
import numpy as np
from numpy.linalg import pinv, LinAlgError
from scipy.stats.mstats import rankdata

from ruptures.base import BaseCost
from ruptures.costs import NotEnoughPoints


class CostRank(BaseCost):
    r"""Rank-based cost function."""

    model = "rank"

    def __init__(self):
        """Initialize the object."""
        self.inv_cov = None
        self.ranks = None
        self.min_size = 2

    def fit(self, signal) -> "CostRank":
        """Set parameters of the instance.

        Args:
            signal (array): signal. Shape (n_samples,) or (n_samples, n_features)

        Returns:
            self
        """
        if signal.ndim == 1:
            signal = signal.reshape(-1, 1)

        obs, vars = signal.shape

        # Convert signal data into ranks in the range [1, n]
        ranks = rankdata(signal, axis=0)
        # Center the ranks into the range [-(n+1)/2, (n+1)/2]
        centered_ranks = ranks - ((obs + 1) / 2)
        # Sigma is the covariance of these ranks.
        # If it's a scalar, reshape it into a 1x1 matrix
        cov = np.cov(centered_ranks, rowvar=False, bias=True).reshape(vars, vars)

        # Use the pseudoinverse to handle linear dependencies
        # see Lung-Yut-Fong, A., Lévy-Leduc, C., & Cappé, O. (2015)
        try:
            self.inv_cov = pinv(cov)
        except LinAlgError as e:
            raise LinAlgError(
                "The covariance matrix of the rank signal is not invertible and the "
                "pseudo-inverse computation did not converge."
            ) from e
        self.ranks = centered_ranks
        self.signal = signal

        return self

    def error(self, start, end):
        """Return the approximation cost on the segment [start:end].

        Args:
            start (int): start of the segment
            end (int): end of the segment

        Returns:
            float: segment cost

        Raises:
            NotEnoughPoints: when the segment is too short (less than `min_size` samples).
        """
        if end - start < self.min_size:
            raise NotEnoughPoints

        mean = np.reshape(np.mean(self.ranks[start:end], axis=0), (-1, 1))

        return -(end - start) * mean.T @ self.inv_cov @ mean
