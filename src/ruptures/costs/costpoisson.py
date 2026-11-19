"""Cost for changes in the rate of independent Poisson counts."""

import numpy as np
from numpy.typing import NDArray
from scipy.special import xlogy
from typing_extensions import Self

from ruptures.base import BaseCost
from ruptures.exceptions import NotEnoughPoints


class CostPoisson(BaseCost):
    """Poisson negative log likelihood, up to a segmentation-independent
    constant."""

    model = "poisson"
    min_size = 1

    def fit(self, signal: NDArray[np.number]) -> Self:
        """Fit a univariate or multivariate count signal sampled at equal
        exposures."""
        signal = np.asarray(signal)
        if signal.ndim not in (1, 2) or signal.dtype.kind not in "uif":
            raise ValueError("Expected a one- or two-dimensional array of counts.")
        if (
            not np.isfinite(signal).all()
            or (signal < 0).any()
            or (signal != np.floor(signal)).any()
        ):
            raise ValueError("Poisson counts must be finite nonnegative integers.")
        self.signal = signal.reshape(-1, 1) if signal.ndim == 1 else signal
        self._prefix = np.zeros((len(signal) + 1, self.signal.shape[1]))
        np.cumsum(self.signal, axis=0, dtype=float, out=self._prefix[1:])
        return self

    def error(self, start: int, end: int) -> float:
        """Return the cost of fitting one rate per feature on [start:end]."""
        if end - start < self.min_size:
            raise NotEnoughPoints
        totals = self._prefix[end] - self._prefix[start]
        return -float(xlogy(totals, totals / (end - start)).sum())
