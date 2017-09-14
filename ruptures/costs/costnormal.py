r"""
Cost function for piecewise Gaussian signals
====================================================================================================

Cost function for piecewise constant functions.

"""
import numpy as np
from numpy.linalg import det

from ruptures.base import BaseCost
from ruptures.costs import NotEnoughPoints


class CostNormal(BaseCost):

    r"""Computes the approximation error when the signal is assumed to be piecewise i.i.d with
    Gaussian density.
    Formally, for a signal :math:`\{y_t\}_t` on an interval :math:`I`,

    .. math:: c(y_{I}) = |I| \log\det\widehat{\Sigma}_I

    where :math:`\widehat{\Sigma}_I` is the empirical covariance matrix of the sub-signal :math:`\{y_t\}_{t\in I}`.
    """

    model = "normal"

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
        sub = self.signal[start:end]

        if self.signal.shape[1] > 1:
            cov = det(np.cov(sub.T))
        else:
            cov = sub.var()

        return np.log(cov) * (end - start)
