r"""Wild Binary Segmentation (WBS).

Reference: Fryzlewicz, P. (2014). Wild binary segmentation for multiple
change-point detection. *The Annals of Statistics*, 42(6), 2243–2281.
https://doi.org/10.1214/14-AOS1245
"""

from functools import lru_cache
from typing import Any, Optional, Union
from typing_extensions import Self

import numpy as np
from numpy.typing import NDArray
from ruptures.base import BaseCost, BaseEstimator
from ruptures.costs import cost_factory
from ruptures.exceptions import BadSegmentationParameters
from ruptures.utils import pairwise, sanity_check


class WBS(BaseEstimator):
    """Wild Binary Segmentation (WBS) change-point detection.

    Implements the WBS algorithm of Fryzlewicz (2014). Instead of always
    splitting the whole signal (as in Binary Segmentation), WBS draws
    ``n_draws`` random sub-intervals at fit time, finds the best split
    point within each, and greedily accepts the candidate with the highest
    gain — then recurses on both sub-segments.  This robustness to
    closely-spaced change points is the key advantage over standard Binseg.

    The stopping rule depends on the parameter passed to :meth:`predict`:
    ``n_bkps``, ``pen``, or ``epsilon``.

    Example usage::

        import numpy as np
        from ruptures.datasets import pw_constant
        from ruptures.detection import WBS

        signal, true_bkps = pw_constant(n_samples=200, n_bkps=3, noise_std=2)
        algo = WBS(model="l2").fit(signal)
        predicted = algo.predict(n_bkps=3)
    """

    def __init__(
        self,
        model: str = "l2",
        custom_cost: Optional[BaseCost] = None,
        min_size: int = 2,
        jump: int = 5,
        n_draws: int = 5000,
        seed: Optional[int] = None,
        params: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize a WBS instance.

        Args:
            model (str, optional): segment model, ["l1", "l2", "rbf", ...].
                Not used if ``custom_cost`` is not None.
            custom_cost (BaseCost, optional): custom cost function. Defaults to None.
            min_size (int, optional): minimum segment length. Defaults to 2.
            jump (int, optional): subsample (one every *jump* points). Defaults to 5.
            n_draws (int, optional): number of random sub-intervals to draw.
                Fryzlewicz (2014) recommends ≥ 5000 for reliable detection.
                Defaults to 5000.
            seed (int, optional): random seed for reproducibility. Defaults to None.
            params (dict, optional): parameters forwarded to the cost function.
        """
        if custom_cost is not None and isinstance(custom_cost, BaseCost):
            self.cost = custom_cost
        else:
            if params is None:
                self.cost = cost_factory(model=model)
            else:
                self.cost = cost_factory(model=model, **params)
        self.min_size = max(min_size, self.cost.min_size)
        self.jump = jump
        self.n_draws = n_draws
        self.seed = seed
        self.n_samples = None
        self.signal = None
        self._sub_intervals: list[tuple[int, int]] = []

    def _draw_intervals(self, rng: np.random.Generator) -> list[tuple[int, int]]:
        """Draw n_draws random sub-intervals [s, e) of [0, n_samples)."""
        starts = rng.integers(0, self.n_samples - 1, size=self.n_draws)
        ends = rng.integers(1, self.n_samples + 1, size=self.n_draws)
        # Ensure s < e and interval is long enough to contain at least one split
        intervals = []
        for s, e in zip(starts, ends):
            if s >= e:
                s, e = e, s
            if e - s >= 2 * self.min_size:
                intervals.append((int(s), int(e)))
        # Always include the full signal interval as a fallback
        if not intervals:
            intervals = [(0, self.n_samples)]
        return intervals

    @lru_cache(maxsize=None)
    def single_bkp(self, start: int, end: int) -> tuple[Union[int, None], float]:
        """Return the best breakpoint in ``[start, end)`` and its gain.

        Identical to :class:`~ruptures.detection.Binseg`'s
        implementation so that the per-segment optimum is computed and
        cached in the same way.
        """
        segment_cost = self.cost.error(start, end)
        if np.isinf(segment_cost) and segment_cost < 0:
            return None, 0
        gain_list = []
        for bkp in range(start, end, self.jump):
            if bkp - start >= self.min_size and end - bkp >= self.min_size:
                gain = (
                    segment_cost
                    - self.cost.error(start, bkp)
                    - self.cost.error(bkp, end)
                )
                gain_list.append((gain, bkp))
        if not gain_list:
            return None, 0
        gain, bkp = max(gain_list)
        return bkp, gain

    def _best_candidate(self, start: int, end: int) -> tuple[Union[int, None], float]:
        """Find the best split candidate across all sub-intervals within
        [start, end).

        For each pre-drawn random interval that lies entirely within
        ``[start, end)``, compute the optimal split point; return the
        one with the highest gain.  Falls back to ``single_bkp(start,
        end)`` when no drawn interval is contained in the current
        segment.
        """
        best_bkp, best_gain = None, 0.0
        for s, e in self._sub_intervals:
            if s >= start and e <= end:
                bkp, gain = self.single_bkp(s, e)
                if bkp is not None and gain > best_gain:
                    best_bkp, best_gain = bkp, gain
        # Fallback: no random interval fits inside [start, end)
        if best_bkp is None:
            best_bkp, best_gain = self.single_bkp(start, end)
        return best_bkp, best_gain

    def _seg(
        self,
        n_bkps: Optional[int] = None,
        pen: Optional[float] = None,
        epsilon: Optional[float] = None,
    ) -> dict[tuple[int, int], float]:
        """Run Wild Binary Segmentation with the given stopping rule.

        Args:
            n_bkps (int): number of breakpoints to find before stopping.
            pen (float): penalty value (> 0).
            epsilon (float): reconstruction budget (> 0).

        Returns:
            dict: partition ``{(start, end): cost_value, ...}``
        """
        bkps = [self.n_samples]
        stop = False
        while not stop:
            stop = True
            new_bkps = [
                self._best_candidate(start, end) for start, end in pairwise([0] + bkps)
            ]
            bkp, gain = max(new_bkps, key=lambda x: x[1])

            if bkp is None:
                break

            if n_bkps is not None:
                if len(bkps) - 1 < n_bkps:
                    stop = False
            elif pen is not None:
                if gain > pen:
                    stop = False
            elif epsilon is not None:
                error = self.cost.sum_of_costs(bkps)
                if error > epsilon:
                    stop = False

            if not stop:
                bkps.append(bkp)
                bkps.sort()

        return {
            (start, end): self.cost.error(start, end)
            for start, end in pairwise([0] + bkps)
        }

    def fit(self, signal: NDArray[np.number]) -> Self:
        """Fit the WBS model to ``signal``.

        Draws random sub-intervals and pre-computes the cost matrix.

        Args:
            signal (array): signal to segment.
                Shape ``(n_samples, n_features)`` or ``(n_samples,)``.

        Returns:
            self
        """
        if signal.ndim == 1:
            self.signal = signal.reshape(-1, 1)
        else:
            self.signal = signal
        self.n_samples, _ = self.signal.shape
        self.cost.fit(signal)
        self.single_bkp.cache_clear()

        rng = np.random.default_rng(self.seed)
        self._sub_intervals = self._draw_intervals(rng)

        return self

    def predict(
        self,
        n_bkps: Optional[int] = None,
        pen: Optional[float] = None,
        epsilon: Optional[float] = None,
    ) -> list[int]:
        """Return the optimal breakpoints.

        Must be called after :meth:`fit`.  The stopping rule depends on
        the parameter passed.

        Args:
            n_bkps (int): number of breakpoints to find before stopping.
            pen (float): penalty value (> 0).
            epsilon (float): reconstruction budget (> 0).

        Raises:
            AssertionError: if none of ``n_bkps``, ``pen``, ``epsilon`` is set.
            BadSegmentationParameters: in case of impossible segmentation
                configuration.

        Returns:
            list: sorted list of breakpoints (last element is ``n_samples``).
        """
        msg = "Give a parameter."
        assert any(param is not None for param in (n_bkps, pen, epsilon)), msg

        if not sanity_check(
            n_samples=self.cost.signal.shape[0],
            n_bkps=0 if n_bkps is None else n_bkps,
            jump=self.jump,
            min_size=self.min_size,
        ):
            raise BadSegmentationParameters

        partition = self._seg(n_bkps=n_bkps, pen=pen, epsilon=epsilon)
        bkps = sorted(e for s, e in partition.keys())
        return bkps

    def fit_predict(
        self,
        signal: NDArray[np.number],
        n_bkps: Optional[int] = None,
        pen: Optional[float] = None,
        epsilon: Optional[float] = None,
    ) -> list[int]:
        """Fit to the signal and return the optimal breakpoints.

        Args:
            signal (array): signal. Shape ``(n_samples, n_features)``
                or ``(n_samples,)``.
            n_bkps (int): number of breakpoints.
            pen (float): penalty value (> 0).
            epsilon (float): reconstruction budget (> 0).

        Returns:
            list: sorted list of breakpoints.
        """
        self.fit(signal)
        return self.predict(n_bkps=n_bkps, pen=pen, epsilon=epsilon)
