r"""Continuous linear change."""

from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing_extensions import Self

from ruptures.base import BaseCost
from ruptures.costs import NotEnoughPoints


class CostCLinear(BaseCost):
    r"""Piecewise linear approximation with a continuity constraint."""

    model = "clinear"

    def __init__(self, timestamps: Optional[ArrayLike] = None) -> None:
        """Use numeric sample times for interpolation, or row positions by
        default."""
        self.signal = None
        self.min_size = 3
        self._timestamps = None
        if timestamps is not None:
            times = np.asarray(timestamps)
            if (
                times.ndim != 1
                or times.dtype.kind not in "iuf"
                or not np.all(np.isfinite(times))
                or not np.all(times[1:] > times[:-1])
            ):
                raise ValueError(
                    "timestamps must be finite, numeric and strictly increasing"
                )
            # Subtract integer timestamps before rounding to floating point (e.g. nanoseconds).
            dtype = (
                np.uint64
                if times.dtype.kind in "iu"
                else np.result_type(times, np.float64)
            )
            self._timestamps = times.astype(dtype, copy=False)

    def fit(self, signal: NDArray[np.number]) -> Self:
        """Set parameters of the instance.

        Args:
            signal (array): signal of shape (n_samples, n_dims) or (n_samples,)

        Returns:
            self
        """
        if self._timestamps is not None and len(self._timestamps) != len(signal):
            raise ValueError("timestamps must have one entry per sample")
        if signal.ndim == 1:
            self.signal = signal.reshape(-1, 1)
        else:
            self.signal = signal

        return self

    def error(self, start: int, end: int) -> float:
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
        if self._timestamps is None:
            elapsed = np.arange(1, end - start + 1)
        else:
            elapsed = self._timestamps[start:end] - self._timestamps[start - 1]
        slope = (self.signal[end - 1] - self.signal[start - 1]) / elapsed[-1]
        intercept = self.signal[start - 1]
        approx = slope.reshape(-1, 1) * elapsed + intercept.reshape(-1, 1)
        return np.sum((sub - approx.transpose()) ** 2)
