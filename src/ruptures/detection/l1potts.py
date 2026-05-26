r"""L1Potts.

Implementation prepared with assistance from Claude (Anthropic) acting
as a coding agent: rewrite of the original L1Potts contribution to
faithfully match Algorithm 1 of Storath, Weinmann & Unser (2017), plus
parent-pointer backtracking, input hardening, and the test suite under
tests/test_l1potts.py. The algorithm itself is the authors'; the agent's
role was implementation, hardening, and testing.
"""

import numpy as np

from ruptures.base import BaseEstimator
from ruptures.exceptions import BadSegmentationParameters


class L1Potts(BaseEstimator):
    """Penalized change point detection for piecewise constant 1D signals with
    L1 data fidelity.

    Solves the L1 Potts model

        min_u  sum_i w_i * |f_i - u_i|  +  gamma * #{ i : u_i != u_{i+1} }

    where f is the input signal, w are non-negative weights, and gamma > 0
    is the jump penalty.

    Implements Algorithm 1 of:

        Storath, Weinmann, Unser. Jump-penalized least absolute values
        estimation of scalar or circle-valued signals. Information and
        Inference, 2017.

    The algorithm reduces the search space to V = unique(signal), justified
    by Theorem 2 of the paper (a global minimizer takes values in V because
    the weighted L1 median always lies among the data values), then runs
    a Viterbi-type dynamic program over (level, sample) pairs. The Potts
    penalty is handled in O(K) per column via the Felzenszwalb-Huttenlocher
    trick (the global minimum of the previous column plus gamma is shared
    across all jump targets). Time complexity is O(K * N) where K is the
    number of distinct values in the signal.

    The forward pass matches the paper exactly. For backtracking we store
    explicit parent pointers (one byte per cell) instead of subtracting
    gamma in place from the float64 cost table; this is mathematically
    equivalent to the paper's procedure but reduces memory from
    8 * K * N bytes to K * N + O(K + N) bytes.

    Only 1D signals are supported.
    """

    def __init__(self):
        """Initialize a L1Potts instance."""
        self.jump = 1
        self.min_size = 1
        self.n_samples = None
        self.signal = None
        self._levels = None
        self._weights = None

    def fit(self, signal, weights=None) -> "L1Potts":
        """Set params.

        Args:
            signal (array): signal to segment. Shape (n_samples,) or
                (n_samples, 1). Must be a real-valued numeric array with
                only finite entries.
            weights (array, optional): per-sample non-negative finite
                weights. Shape (n_samples,). Defaults to uniform weights.

        Returns:
            self

        Raises:
            BadSegmentationParameters: if ``signal`` is not 1D (after
                squeezing a trailing length-1 axis) or is empty.
            ValueError: if ``signal`` or ``weights`` has a non-numeric dtype,
                contains NaN/Inf, or ``weights`` has the wrong shape or
                negative entries.
        """
        signal = np.asarray(signal)
        if signal.dtype.kind not in ("i", "u", "f"):
            raise ValueError(
                "L1Potts requires a real-valued numeric signal, got dtype "
                "{}.".format(signal.dtype)
            )
        signal = np.atleast_1d(signal.astype(float, copy=False).squeeze())
        if signal.ndim != 1:
            raise BadSegmentationParameters("L1Potts only accepts 1D signals.")
        if signal.size == 0:
            raise BadSegmentationParameters("L1Potts requires a non-empty signal.")
        if not np.all(np.isfinite(signal)):
            raise ValueError("signal must contain only finite values (no NaN or Inf).")
        n_samples = signal.shape[0]

        if weights is None:
            weights = np.ones(n_samples, dtype=float)
        else:
            weights = np.asarray(weights)
            if weights.dtype.kind not in ("i", "u", "f"):
                raise ValueError(
                    "weights must be real-valued numeric, got dtype "
                    "{}.".format(weights.dtype)
                )
            weights = weights.astype(float, copy=False)
            if weights.shape != (n_samples,):
                raise ValueError(
                    "weights must have shape ({},), got {}.".format(
                        n_samples, weights.shape
                    )
                )
            if not np.all(np.isfinite(weights)):
                raise ValueError(
                    "weights must contain only finite values (no NaN or Inf)."
                )
            if np.any(weights < 0):
                raise ValueError("weights must be non-negative.")

        # Defensive copies: shield the solver from later mutations of the
        # caller's arrays. Cheap (16 * n_samples bytes total).
        self.signal = signal.copy()
        self.n_samples = n_samples
        self._weights = weights.copy()
        self._levels = np.unique(self.signal)
        return self

    def predict(self, pen):
        """Return the optimal breakpoints.

        Must be called after the fit method. The breakpoints are associated
        with the signal passed to ``fit``.

        Args:
            pen (float): jump penalty (>0). May be ``+inf`` to disallow jumps.

        Raises:
            RuntimeError: if called before ``fit``.
            ValueError: if ``pen`` is not a positive number (or NaN).

        Returns:
            list: sorted list of breakpoints (last entry is ``n_samples``).
        """
        if self.signal is None:
            raise RuntimeError("predict() called before fit(); call fit(signal) first.")
        try:
            pen = float(pen)
        except (TypeError, ValueError):
            raise ValueError(
                "pen must be a positive real number, got {!r}.".format(pen)
            )
        if not pen > 0:
            raise ValueError("pen must be positive, got {!r}.".format(pen))

        signal = self.signal
        weights = self._weights
        levels = self._levels
        n_samples = self.n_samples
        n_levels = levels.shape[0]

        # Forward DP. dp[k] is the running cost of segmenting signal[:n+1]
        # ending at level levels[k]. Only the previous column is kept.
        # parents[k, n] == 1 means the optimum entering (levels[k], n) was
        # reached via a jump (best previous level + pen). parents[:, 0] is
        # unused.
        # prev_argmin[n] is argmin_k dp[k] right after step n; used during
        # backtracking to identify the level taken when a jump is made.
        parents = np.zeros((n_levels, n_samples), dtype=np.uint8)
        prev_argmin = np.empty(n_samples, dtype=np.intp)

        dp = np.abs(levels - signal[0]) * weights[0]
        prev_argmin[0] = int(np.argmin(dp))
        for n in range(1, n_samples):
            jump_cost = dp.min() + pen
            take_jump = jump_cost < dp
            parents[:, n] = take_jump
            dp = np.abs(levels - signal[n]) * weights[n] + np.where(
                take_jump, jump_cost, dp
            )
            prev_argmin[n] = int(np.argmin(dp))

        # Backtracking. Walk backwards, recording a breakpoint each time the
        # level changes between consecutive samples.
        level = int(prev_argmin[n_samples - 1])
        bkps = [n_samples]
        for n in range(n_samples - 1, 0, -1):
            new_level = int(prev_argmin[n - 1]) if parents[level, n] else level
            if new_level != level:
                bkps.append(n)
            level = new_level
        bkps.reverse()
        return bkps

    def fit_predict(self, signal, pen, weights=None):
        """Fit to the signal and return the optimal breakpoints.

        Helper method to call fit and predict once.

        Args:
            signal (array): signal. Shape (n_samples,) or (n_samples, 1).
            pen (float): jump penalty (>0)
            weights (array, optional): per-sample non-negative weights.

        Returns:
            list: sorted list of breakpoints
        """
        self.fit(signal, weights=weights)
        return self.predict(pen)

    def _compute_functional_value(self, bkps, pen):
        """Compute the L1 Potts functional value for a given segmentation.

        Uses the (weighted) median of each segment, which is the optimal
        constant level for the L1 fit. Helper for comparing solvers; not
        part of the public API.
        """
        functional_value = pen * (len(bkps) - 1)
        weights = self._weights
        bkps = [0] + bkps
        for i in range(len(bkps) - 1):
            seg = self.signal[bkps[i] : bkps[i + 1]]
            seg_w = weights[bkps[i] : bkps[i + 1]]
            level = _weighted_median(seg, seg_w)
            functional_value += float(np.sum(seg_w * np.abs(seg - level)))
        return functional_value


def _weighted_median(values, weights):
    """Lower weighted median of ``values`` with non-negative ``weights``.

    Returns the smallest value v such that the cumulative weight of the
    samples at or below v is at least half of the total weight. Falls
    back to the unweighted middle element if the total weight is zero.
    """
    order = np.argsort(values, kind="stable")
    sorted_vals = values[order]
    sorted_w = weights[order]
    csum = np.cumsum(sorted_w)
    total = csum[-1]
    if total <= 0:
        return float(sorted_vals[len(sorted_vals) // 2])
    idx = int(np.searchsorted(csum, total / 2.0, side="left"))
    if idx >= sorted_vals.shape[0]:
        idx = sorted_vals.shape[0] - 1
    return float(sorted_vals[idx])
