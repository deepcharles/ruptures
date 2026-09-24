r"""CostL2 (least squared deviation)"""

from numpy.typing import NDArray
import numpy as np
from typing_extensions import Self


from ruptures.costs import NotEnoughPoints

from ruptures.base import BaseCost


class CostL2(BaseCost):
    r"""Least squared deviation."""

    model = "l2"

    def __init__(self) -> None:
        """Initialize the object."""
        self.signal = None
        self._cumsum = None
        self._cumsum_sq = None
        self.min_size = 1

    def fit(self, signal: NDArray[np.number]) -> Self:
        """Set parameters of the instance.

        Args:
            signal (array): array of shape (n_samples,) or (n_samples, n_features)

        Returns:
            self
        """
        if signal.ndim == 1:
            self.signal = signal.reshape(-1, 1)
        else:
            self.signal = signal

        # Centering is not required by the prefix-sum formula, but it reduces
        # cancellation when the signal has a large offset. Accumulating in at
        # least double precision also preserves the accuracy of integer and
        # low-precision floating-point inputs.
        dtype = np.result_type(self.signal.dtype, np.float64)
        centered_signal = self.signal - self.signal.mean(axis=0, dtype=dtype)
        squared_signal = np.real(centered_signal * centered_signal.conj())

        prefix_shape = (self.signal.shape[0] + 1, self.signal.shape[1])
        self._cumsum = np.empty(prefix_shape, dtype=dtype)
        self._cumsum[0] = 0
        np.cumsum(centered_signal, axis=0, dtype=dtype, out=self._cumsum[1:])

        self._cumsum_sq = np.empty(prefix_shape, dtype=squared_signal.dtype)
        self._cumsum_sq[0] = 0
        np.cumsum(
            squared_signal,
            axis=0,
            dtype=squared_signal.dtype,
            out=self._cumsum_sq[1:],
        )

        return self

    def error(self, start: int, end: int) -> float:
        """Return the approximation cost on the segment [start:end].

        Args:
            start (int): start of the segment
            end (int): end of the segment

        Returns:
            segment cost

        Raises:
            NotEnoughPoints: when the segment is too short (less than `min_size` samples).
        """
        if end - start < self.min_size:
            raise NotEnoughPoints

        n_samples = end - start
        if n_samples == 1:
            return 0.0

        segment_sum = self._cumsum[end] - self._cumsum[start]
        segment_sum_sq = self._cumsum_sq[end] - self._cumsum_sq[start]
        segment_cost = (
            segment_sum_sq - np.real(segment_sum * segment_sum.conj()) / n_samples
        )

        # Round-off can make a theoretically non-negative cost slightly negative.
        return float(np.maximum(segment_cost.sum(), 0.0))
