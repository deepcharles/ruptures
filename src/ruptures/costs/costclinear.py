r"""Continuous linear change."""
import numpy as np

from ruptures.base import BaseCost
from ruptures.costs import NotEnoughPoints


class CostCLinear(BaseCost):
    r"""Piecewise linear approximation with a continuity constraint."""

    model = "clinear"

    def __init__(self):
        """Initialize the object."""
        self.signal = None
        self.min_size = 3

    def fit(self, signal) -> "CostCLinear":
        """Set parameters of the instance.

        Args:
            signal (array): signal of shape (n_samples, n_dims) or (n_samples,)

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
            segment cost (float)

        Raises:
            NotEnoughPoints: when the segment is too short (less than `min_size`
                samples).
        """
        if end - start < self.min_size:
            raise NotEnoughPoints

        if start == 0:
            start = 1

        sub = self.signal[start:end]
        slope = (self.signal[end - 1] - self.signal[start - 1]) / (end - start)
        intercept = self.signal[start - 1]
        approx = slope.reshape(-1, 1) * np.arange(
            1, end - start + 1
        ) + intercept.reshape(-1, 1)
        return np.sum((sub - approx.transpose()) ** 2)
