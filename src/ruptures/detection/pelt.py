r"""Pelt."""

from typing import Any, Optional
from typing_extensions import Self

import numpy as np
from numpy.typing import NDArray
from ruptures.costs import cost_factory
from ruptures.base import BaseCost, BaseEstimator
from ruptures.exceptions import BadSegmentationParameters
from ruptures.utils import sanity_check


class Pelt(BaseEstimator):
    """Penalized change point detection.

    For a given model and penalty level, computes the segmentation which
    minimizes the constrained sum of approximation errors.
    """

    def __init__(
        self,
        model="l2",
        custom_cost: Optional[BaseCost] = None,
        min_size: int = 2,
        jump: int = 5,
        params: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize a Pelt instance.

        Args:
            model (str, optional): segment model, ["l1", "l2", "rbf"]. Not used if ``'custom_cost'`` is not None.
            custom_cost (BaseCost, optional): custom cost function. Defaults to None.
            min_size (int, optional): minimum segment length.
            jump (int, optional): subsample (one every *jump* points).
            params (dict, optional): a dictionary of parameters for the cost instance.
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
        self.n_samples = None

    def _seg(self, pen: float) -> dict[tuple[int, int], float]:
        """Compute the optimal penalized partition with delayed pruning.

        A start dominated at ``s`` stays eligible until ``s + min_size``.
        Only then can ``s`` legally start the replacement segment.
        As in the usual PELT rule, this requires the cost inequality
        ``C(r, u) >= C(r, s) + C(s, u)`` on legal segments.

        Args:
            pen (float): Penalty per segment. This differs from a penalty per
                change point by the same constant for every partition.

        Returns:
            dict: Mapping from segment bounds to segment cost plus penalty.
        """
        partitions = {0: {(0, 0): 0}}
        admissible = []
        prune_at = {}

        endpoints = [
            k for k in range(0, self.n_samples, self.jump) if k >= self.min_size
        ]
        endpoints.append(self.n_samples)

        # Each possible start enters once, only after a legal final segment
        # can follow it. Exclude n: it cannot start a nonempty segment.
        pending = iter([0] + endpoints[:-1])
        next_start = next(pending, None)

        for bkp in endpoints:
            while next_start is not None and next_start <= bkp - self.min_size:
                admissible.append(next_start)
                next_start = next(pending, None)

            # A witness s can replace a start only at endpoints u >= s+m.
            # Compare sample coordinates, not the number of grid iterations;
            # the final endpoint can lie off the jump grid.
            admissible = [
                t for t in admissible if t not in prune_at or bkp < prune_at[t]
            ]

            candidates = []
            for t in admissible:
                partition = partitions[t].copy()
                partition[(t, bkp)] = self.cost.error(t, bkp) + pen
                candidates.append((t, partition, sum(partition.values())))

            _, best_partition, best_value = min(candidates, key=lambda item: item[2])
            partitions[bkp] = best_partition

            if bkp != self.n_samples:
                for t, _, value in candidates:
                    if value > best_value + pen:
                        # Endpoints increase, so the first witness gives the
                        # earliest safe expiry. Later witnesses cannot extend it.
                        prune_at.setdefault(t, bkp + self.min_size)

        best_partition = partitions[self.n_samples]
        del best_partition[(0, 0)]
        return best_partition

    def fit(self, signal: NDArray[np.number]) -> Self:
        """Set params.

        Args:
            signal (array): signal to segment. Shape (n_samples, n_features) or (n_samples,).

        Returns:
            self
        """
        # update params
        self.cost.fit(signal)
        if signal.ndim == 1:
            (n_samples,) = signal.shape
        else:
            n_samples, _ = signal.shape
        self.n_samples = n_samples
        return self

    def predict(self, pen: float) -> list[int]:
        """Return the optimal breakpoints.

        Must be called after the fit method. The breakpoints are associated with the signal passed
        to [`fit()`][ruptures.detection.pelt.Pelt.fit].

        Args:
            pen (float): penalty value (>0)

        Raises:
            BadSegmentationParameters: in case of impossible segmentation
                configuration

        Returns:
            list: sorted list of breakpoints
        """
        # raise an exception in case of impossible segmentation configuration
        if not sanity_check(
            n_samples=self.cost.signal.shape[0],
            n_bkps=0,
            jump=self.jump,
            min_size=self.min_size,
        ):
            raise BadSegmentationParameters

        partition = self._seg(pen)
        bkps = sorted(e for s, e in partition.keys())
        return bkps

    def fit_predict(self, signal: NDArray[np.number], pen: float) -> list[int]:
        """Fit to the signal and return the optimal breakpoints.

        Helper method to call fit and predict once

        Args:
            signal (array): signal. Shape (n_samples, n_features) or (n_samples,).
            pen (float): penalty value (>0)

        Returns:
            list: sorted list of breakpoints
        """
        self.fit(signal)
        return self.predict(pen)
