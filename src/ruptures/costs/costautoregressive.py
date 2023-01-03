import numpy as np
from numpy.lib.stride_tricks import as_strided
from numpy.linalg import lstsq
from copy import deepcopy

from ruptures.base import BaseCost
from ruptures.costs import NotEnoughPoints


class CostAR(BaseCost):
    r"""Least-squares estimate for changes in autoregressive coefficients."""

    model = "ar"

    def __init__(self, order=4):
        """Initialize the object.

        Args:
            order (int): autoregressive order
        """
        self.signal = None
        self.covar = None
        self.min_size = max(5, order + 1)
        self.order = order

    def fit(self, signal):
        """Set parameters of the instance. The signal must be 1D.

        Args:
            signal (array): 1d signal. Shape (n_samples, 1) or (n_samples,).

        Returns:
            self: the current object
        """
        self.signal = deepcopy(signal)
        if signal.ndim == 1:
            self.signal = self.signal.reshape(-1, 1)

        # lagged covariates
        n_samples, _ = self.signal.shape
        strides = (self.signal.itemsize, self.signal.itemsize)
        shape = (n_samples - self.order, self.order)
        lagged = as_strided(self.signal, shape=shape, strides=strides)
        # pad the first columns
        lagged_after_padding = np.pad(lagged, ((self.order, 0), (0, 0)), mode="edge")
        # add intercept
        self.covar = np.c_[lagged_after_padding, np.ones(n_samples)]
        # pad signal on the edges
        self.signal[: self.order] = self.signal[self.order]
        return self

    def error(self, start, end):
        """Return the approximation cost on the segment [start:end].

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
        y, X = self.signal[start:end], self.covar[start:end]
        _, residual, _, _ = lstsq(X, y, rcond=None)
        return residual.sum()
