"""Cost functions for piecewise constant functions."""
import numpy as np
from scipy.spatial.distance import pdist, squareform
from ruptures.costs import NotEnoughPoints

from ruptures.base import BaseCost


class CostL2(BaseCost):

    r"""Computes the approximation error when the signal is assumed to be piecewise constant.
    Formally, for a signal :math:`\{y_t\}_t` on an interval :math:`I`,

    .. math:: c(y_{}) = \sum_{t\in I} \|y_t - \bar{y}\|^2_2

    where :math:`\bar{y}=\frac{1}{|I|} \sum\limits_{t\in I} y_t`.
    """

    def __init__(self):
        self.signal = None
        self.min_size = 2

    def fit(self, signal):
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

        # self.cumsum = np.cumsum(self.signal, axis=0)
        # self.cumsum_norm = np.cumsum((self.signal**2).sum(axis=1), axis=0)

        return self

    def error(self, start, end):
        """Returns the approximation cost on the segment [start:end].

        Args:
            start (int): start of the segment
            end (int): end of the segment

        Returns:
            float: segment cost

        Raises:
            NotEnoughPoints: when the segment is too short (less than ``'min_size'`` samples).
        """
        if end - start < self.min_size:
            raise NotEnoughPoints

        return self.signal[start:end].var(axis=0).sum()


class Cost(BaseCost):

    """Compute error (in different norms) when approximating a signal with a constant value."""

    def __init__(self, model="constantl2"):
        assert model in [
            "constantl1", "constantl2", "rbf"], "Choose different model."
        self.model = model
        if self.model in ["constantl1", "constantl2", "rbf"]:
            self.min_size = 2

        self.signal = None
        self.gram = None

    def fit(self, signal):
        """Update the parameters of the instance to fit the signal.

        Detailled description

        Args:
            arg1 (array): signal of shape (n_samples, n_features) of (n_samples,)

        Returns:
            self:
        """
        if signal.ndim == 1:
            self.signal = signal.reshape(-1, 1)
        else:
            self.signal = signal

        if self.model == "rbf":
            pairwise_dists = pdist(self.signal, 'sqeuclidean')
            pairwise_dists /= np.median(pairwise_dists)  # scaling
            self.gram = squareform(np.exp(-pairwise_dists))
            np.fill_diagonal(self.gram, 1)
        elif self.model == "constantl2":
            self.gram = self.signal.dot(self.signal.T)

        return self

    def error(self, start, end):
        """Return squared error on the interval start:end

        Detailled description

        Args:
            start (int): start index (inclusive)
            end (int): end index (exclusive)

        Returns:
            float: error

        Raises:
            NotEnoughPoints: when not enough points
        """
        if end - start < self.min_size:
            raise NotEnoughPoints
        if self.model in ["constantl2", "rbf"]:
            sub_gram = self.gram[start:end, start:end]
            cost = np.diagonal(sub_gram).sum()
            cost -= sub_gram.sum() / (end - start)
        elif self.model == "constantl1":
            med = np.median(self.signal[start:end], axis=0)
            cost = abs(self.signal[start:end] - med).sum()
        return cost
