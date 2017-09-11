"""Cost function for piecewise constant functions."""
import numpy as np

from ruptures.base import BaseCost
from ruptures.costs import NotEnoughPoints


class CostL1(BaseCost):

    r"""Computes the approximation error when the signal is assumed to be piecewise constant.
    Formally, for a signal :math:`\{y_t\}_t` on an interval :math:`I`,

    .. math:: c(y_{I}) = \sum_{t\in I} \|y_t - \bar{y}\|_1

    where :math:`\bar{y}` is the componentwise median of :math:`\{y_t\}_{t\in I}`.
    """

    model = "l1"

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
        med = np.median(sub, axis=0)

        return abs(sub - med).sum()
