r"""E-Divisive change point detection (Matteson & James, 2014)."""

from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist
from typing_extensions import Self

from ruptures.base import BaseEstimator
from ruptures.exceptions import BadSegmentationParameters


class EDivisive(BaseEstimator):
    r"""E-Divisive: energy-statistic-based change point detection.

    Detects multiple change points in multivariate time series by
    recursively splitting segments that maximise the empirical energy
    divergence.  No distributional assumption is required.

    At each iteration the algorithm selects the split point
    :math:`\tau^*` and segment :math:`[a, b)` that maximise the
    scaled energy statistic

    .. math::

        Q(\tau; a, b) = \frac{mn}{m+n}
        \left(
          \frac{2}{mn}\sum_{i=a}^{\tau-1}\sum_{j=\tau}^{b-1}
              \|\mathbf{x}_i - \mathbf{x}_j\|^\alpha
          - \frac{1}{m^2}\sum_{i,i'=a}^{\tau-1}
              \|\mathbf{x}_i - \mathbf{x}_{i'}\|^\alpha
          - \frac{1}{n^2}\sum_{j,j'=\tau}^{b-1}
              \|\mathbf{x}_j - \mathbf{x}_{j'}\|^\alpha
        \right),

    where :math:`m = \tau - a` and :math:`n = b - \tau`.

    A permutation test decides whether the best split is significant.
    Alternatively, pass ``n_bkps`` to fix the number of breakpoints and
    skip significance testing.

    References:
        Matteson, D. S. & James, N. A. (2014). A Nonparametric Approach
        for Multiple Change Point Analysis of Multivariate Data.
        *Journal of the American Statistical Association*, 109(505), 334-345.

    Example:
        >>> import numpy as np
        >>> from ruptures import EDivisive
        >>> signal = np.concatenate([np.zeros(50), np.ones(50)])
        >>> EDivisive(n_perms=0).fit_predict(signal, n_bkps=1)
        [50, 100]
    """

    def __init__(
        self,
        alpha: float = 1.0,
        min_size: int = 2,
        n_perms: int = 200,
        sig_level: float = 0.05,
    ) -> None:
        """Initialise EDivisive.

        Args:
            alpha: exponent applied to pairwise Euclidean distances,
                must satisfy ``0 < alpha <= 2``. The default ``alpha=1``
                corresponds to standard Euclidean distances.
            min_size: minimum number of samples in any segment.
            n_perms: number of permutations used in the significance
                test.  Set to ``0`` to disable the test (greedy
                splitting until ``n_bkps`` is reached).
            sig_level: significance level for the permutation test.
                Ignored when ``n_bkps`` is supplied to
                :meth:`predict`.
        """
        if not (0.0 < alpha <= 2.0):
            raise ValueError("alpha must satisfy 0 < alpha <= 2.")
        self.alpha = alpha
        self.min_size = min_size
        self.n_perms = n_perms
        self.sig_level = sig_level

        self.signal: Optional[NDArray[np.floating]] = None
        self.n_samples: Optional[int] = None
        self._dist: Optional[NDArray[np.floating]] = None
        self._rng = np.random.default_rng()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _q_stat(self, idx_a: NDArray[np.intp], idx_b: NDArray[np.intp]) -> float:
        """Return the scaled energy statistic between two index groups."""
        m, n = idx_a.shape[0], idx_b.shape[0]
        assert self._dist is not None  # guaranteed after fit()
        cross = float(self._dist[np.ix_(idx_a, idx_b)].sum())
        w_a = float(self._dist[np.ix_(idx_a, idx_a)].sum())
        w_b = float(self._dist[np.ix_(idx_b, idx_b)].sum())
        e = 2.0 * cross / (m * n) - w_a / (m * m) - w_b / (n * n)
        return e * m * n / (m + n)

    def _best_split(self, start: int, end: int) -> tuple[Optional[int], float]:
        """Return ``(t*, Q*)`` — the optimal split and its Q value."""
        best_t: Optional[int] = None
        best_q = -np.inf
        seg = np.arange(start, end, dtype=np.intp)
        for t in range(start + self.min_size, end - self.min_size + 1):
            m = t - start
            q = self._q_stat(seg[:m], seg[m:])
            if q > best_q:
                best_q, best_t = q, t
        return best_t, best_q

    def _perm_test(
        self,
        start: int,
        t_star: int,
        end: int,
        q_star: float,
    ) -> bool:
        """Return ``True`` if the split at ``t_star`` is significant.

        Uses a fixed-split permutation test: the indices in ``[start, end)``
        are shuffled and the energy statistic is recomputed at the same
        split size as ``t_star``, yielding the null distribution.
        """
        if self.n_perms == 0:
            return True
        m = t_star - start
        indices = np.arange(start, end, dtype=np.intp)
        exceed = 0
        for _ in range(self.n_perms):
            perm = self._rng.permutation(indices)
            if self._q_stat(perm[:m], perm[m:]) >= q_star:
                exceed += 1
        return (exceed / self.n_perms) < self.sig_level

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, signal: NDArray[np.number]) -> Self:
        """Pre-compute the pairwise distance matrix.

        Args:
            signal: array of shape ``(n_samples,)`` or
                ``(n_samples, n_features)``.

        Returns:
            self
        """
        if signal.ndim == 1:
            self.signal = signal.reshape(-1, 1).astype(float)
        else:
            self.signal = signal.astype(float)
        self.n_samples = self.signal.shape[0]
        self._dist = cdist(self.signal, self.signal, metric="euclidean") ** self.alpha
        self._rng = np.random.default_rng()
        return self

    def predict(self, n_bkps: Optional[int] = None) -> list[int]:
        """Return the list of change points.

        Must be called after :meth:`fit`.

        Args:
            n_bkps: target number of change points.  When supplied,
                splitting stops as soon as ``n_bkps`` breakpoints have
                been found and no permutation test is performed.  When
                ``None``, the permutation test (controlled by
                ``n_perms`` and ``sig_level``) governs when to stop.

        Returns:
            Sorted list of breakpoint indices, by convention the last
            element equals ``n_samples``.

        Raises:
            BadSegmentationParameters: if neither ``n_bkps`` nor the
                permutation test can produce a valid segmentation.
        """
        if self.signal is None:
            raise BadSegmentationParameters(
                "Call fit() before predict()."
            )

        assert self.n_samples is not None

        if n_bkps is not None and n_bkps < 0:
            raise BadSegmentationParameters("n_bkps must be a non-negative integer.")

        bkps: set[int] = set()
        # Segments still eligible for splitting: list of (start, end) pairs.
        pending: list[tuple[int, int]] = [(0, self.n_samples)]

        while pending:
            # Find the globally best split across all pending segments.
            best_candidate: Optional[tuple[float, int, int, int]] = None
            for start, end in pending:
                if end - start < 2 * self.min_size:
                    continue
                t, q = self._best_split(start, end)
                if t is not None:
                    if best_candidate is None or q > best_candidate[0]:
                        best_candidate = (q, t, start, end)

            if best_candidate is None:
                break

            q_best, t_best, s_best, e_best = best_candidate

            # Stopping criterion.
            if n_bkps is not None:
                if len(bkps) >= n_bkps:
                    break
            else:
                if not self._perm_test(s_best, t_best, e_best, q_best):
                    break

            # Accept the split.
            bkps.add(t_best)
            pending.remove((s_best, e_best))
            for child in ((s_best, t_best), (t_best, e_best)):
                if child[1] - child[0] >= 2 * self.min_size:
                    pending.append(child)

        bkps.add(self.n_samples)
        return sorted(bkps)

    def fit_predict(
        self,
        signal: NDArray[np.number],
        n_bkps: Optional[int] = None,
    ) -> list[int]:
        """Fit and predict in one step.

        Args:
            signal: array of shape ``(n_samples,)`` or
                ``(n_samples, n_features)``.
            n_bkps: see :meth:`predict`.

        Returns:
            see :meth:`predict`.
        """
        self.fit(signal)
        return self.predict(n_bkps=n_bkps)
