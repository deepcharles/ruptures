r"""
Cost function for kernel change point detection
====================================================================================================

Cost function for piecewise constant signal in a Hilbert space.

"""
import numpy as np
from scipy.spatial.distance import pdist, squareform

from ruptures.exceptions import NotEnoughPoints
from ruptures.base import BaseCost


class CostRbf(BaseCost):

    r"""Computes the approximation error when the signal mapped onto a Hilbert space
    :math:`\mathcal{H}` through the Rbf kernel.
    Formally, for a signal :math:`\{y_t\}_t` on an interval :math:`I`,

    .. math:: c(y_{I}) = \min_{\mu\in\mathcal{H}} \sum_{t\in I} \|\Phi(y_t) - \mu \|_H^2

    where :math:`\Phi` is the mapping defined by :math:`k(x, y) = \exp(-\gamma\|x-y\|^2)` and
    :math:`\gamma` is the so-called bandwith parameter (chosen following the median heuristics).
    """

    model = "rbf"

    def __init__(self):
        self.gram = None
        self.min_size = 2

    def fit(self, signal):
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
        K /= -np.median(K)
        self.gram = np.exp(squareform(K))
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
        sub_gram = self.gram[start:end, start:end]
        val = np.diagonal(sub_gram).sum()
        val -= sub_gram.sum() / (end - start)
        return val
