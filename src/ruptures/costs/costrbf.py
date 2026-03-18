r"""Kernelized mean change."""

import typing
import functools

import numpy as np

from ruptures.exceptions import NotEnoughPoints
from ruptures.base import BaseCost


class _ImplicitSquareDistanceMatrix:
    _signal: np.typing.NDArray[np.number]

    def __init__(self, signal: np.typing.NDArray[np.number]) -> None:
        self._signal = signal

    @functools.cached_property
    def _centered_signal(self) -> np.ndarray:
        return self._signal - self._signal.mean(axis=0)[None, :]

    @functools.cached_property
    def _sqnorms(self) -> np.ndarray:
        return (self._centered_signal**2).sum(axis=1)

    @functools.cached_property
    def shape(self) -> tuple[int, int]:
        return (self._signal.shape[0],) * 2

    def __getitem__(
        self,
        idx: tuple[int | slice | np.ndarray, int | slice | np.ndarray],
    ) -> np.typing.NDArray[np.number]:
        idxr, idxc = idx
        sqnorms_row = self._sqnorms[idxr]
        sqnorms_row_1d = np.atleast_1d(sqnorms_row)
        sqnorms_col = self._sqnorms[idxc]
        sqnorms_col_1d = np.atleast_1d(sqnorms_col)
        mat = (-2 * np.atleast_2d(self._centered_signal[idxr])) @ np.atleast_2d(
            self._centered_signal[idxc]
        ).T
        mat += sqnorms_row_1d[:, None]
        mat += sqnorms_col_1d[None, :]
        np.maximum(mat, 0, out=mat)
        if sqnorms_col.ndim < 1:
            mat = mat[:, 0]
        if sqnorms_row.ndim < 1:
            mat = mat[0]
        return mat


def _explicit_square_distance_matrix(signal: np.typing.NDArray[np.number]):
    centered_signal = signal - signal.mean(axis=0)[None, :]
    sqnorms = (centered_signal**2).sum(axis=1)
    mat = (-2 * centered_signal) @ centered_signal.T
    mat += sqnorms[:, None]
    mat += sqnorms[None, :]

    np.maximum(mat, 0, out=mat)
    return mat


class _ImplicitGramMatrix:
    _square_distance_matrix: _ImplicitSquareDistanceMatrix
    _gamma: float

    def __init__(
        self, square_distance_matrix: _ImplicitSquareDistanceMatrix, gamma: float
    ):
        self._square_distance_matrix = square_distance_matrix
        self._gamma = gamma

    @functools.cached_property
    def shape(self):
        return self._square_distance_matrix.shape

    def __getitem__(
        self,
        idx: tuple[int | slice | np.ndarray, int | slice | np.ndarray],
    ) -> np.typing.NDArray[np.number]:
        mat = self._square_distance_matrix[idx]
        mat *= -self._gamma
        np.exp(mat, out=mat)
        return mat


class CostRbf(BaseCost):
    r"""Kernel cost function (rbf kernel)."""

    model = "rbf"

    min_size: int
    gamma: typing.Optional[float]
    signal: typing.Optional[np.typing.NDArray[np.number]]
    _gram: typing.Optional[np.typing.NDArray[np.number]]
    quadratic_precompute: typing.Optional[bool]

    def __init__(
        self,
        gamma: typing.Optional[float] = None,
        quadratic_precompute: typing.Optional[bool] = True,
    ):
        """Initialize the object."""
        self.min_size = 1
        self.gamma = gamma
        self._gram = None
        self.signal = None
        self.quadratic_precompute = quadratic_precompute

    @functools.cached_property
    def gram(self) -> typing.Union[np.typing.NDArray[np.number], _ImplicitGramMatrix]:
        """Generate the gram matrix (lazy loading).

        Only access this function after a `.fit()` (otherwise
        `self.signal` is not defined).
        """

        if self.signal is None:
            raise ValueError(".fit() must be used before accessing this matrix")

        if (
            self.quadratic_precompute
            or self.quadratic_precompute is None
            and self.signal.shape[0] <= 4096
        ):
            mat = _explicit_square_distance_matrix(self.signal)
            if self.gamma is None:
                # median heuristic (on whole matrix if n<=4096, on subsample
                # else)
                subsample_factor = int(np.ceil(self.signal.shape[0] / 4096))
                med: float = np.median(
                    mat[::subsample_factor, (subsample_factor // 2) :: subsample_factor]
                )
                self.gamma = 1.0
                if med > 0:
                    self.gamma = 1 / med
            mat *= -self.gamma
            np.exp(mat, out=mat)
            return mat

        square_distance_matrix = _ImplicitSquareDistanceMatrix(self.signal)
        gamma = self.gamma
        if gamma is None:
            # median heuristic on a subsample
            subsample_factor = int(np.ceil(self.signal.shape[0] / 4096))
            med: float = np.median(
                square_distance_matrix[
                    ::subsample_factor, (subsample_factor // 2) :: subsample_factor
                ]
            )
            gamma = 1.0
            if med > 0:
                gamma = 1 / med
            self.gamma = gamma
        return _ImplicitGramMatrix(square_distance_matrix, gamma)

    def fit(self, signal: np.typing.ArrayLike) -> "CostRbf":
        """Sets parameters of the instance.

        Args:
            signal (array): signal. Shape (n_samples,) or (n_samples, n_features)

        Returns:
            self
        """
        signal_ndarray: np.typing.NDArray[np.number] = np.asarray(signal)
        if signal_ndarray.ndim == 1:
            self.signal = signal_ndarray.reshape(-1, 1)
        else:
            self.signal = signal_ndarray

        # If gamma is none, set it using the median heuristic.
        # This heuristic involves computing the gram matrix which is lazy loaded
        # so we simply access the `.gram` property
        if self.gamma is None:
            self.gram

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
        sub_gram = self.gram[start:end, start:end]
        val = np.diagonal(sub_gram).sum()
        val -= sub_gram.sum() / (end - start)
        return val
