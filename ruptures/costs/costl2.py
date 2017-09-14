r"""
Mean squared deviation
====================================================================================================

Cost functions for piecewise constant functions.

"""
from ruptures.costs import NotEnoughPoints

from ruptures.base import BaseCost


class CostL2(BaseCost):

    r"""Computes the approximation error when the signal is assumed to be piecewise constant.
    Formally, for a signal :math:`\{y_t\}_t` on an interval :math:`I`,

    .. math:: c(y_{}) = \sum_{t\in I} \|y_t - \bar{y}\|^2_2

    where :math:`\bar{y}=\frac{1}{|I|} \sum\limits_{t\in I} y_t`.
    """

    model = "l2"

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

        return self.signal[start:end].var(axis=0).sum() * (end - start)
