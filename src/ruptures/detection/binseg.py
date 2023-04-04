r"""Binary segmentation."""
from functools import lru_cache

import numpy as np
from ruptures.base import BaseCost, BaseEstimator
from ruptures.costs import cost_factory
from ruptures.exceptions import BadSegmentationParameters
from ruptures.utils import pairwise, sanity_check


class Binseg(BaseEstimator):
    """Binary segmentation."""

    def __init__(self, model="l2", custom_cost=None, min_size=2, jump=5, params=None):
        """Initialize a Binseg instance.

        Args:
            model (str, optional): segment model, ["l1", "l2", "rbf",...]. Not used if ``'custom_cost'`` is not None.
            custom_cost (BaseCost, optional): custom cost function. Defaults to None.
            min_size (int, optional): minimum segment length. Defaults to 2 samples.
            jump (int, optional): subsample (one every *jump* points). Defaults to 5 samples.
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
        self.signal = None

    def _seg(self, n_bkps=None, pen=None, epsilon=None):
        """Computes the binary segmentation.

        The stopping rule depends on the parameter passed to the function.

        Args:
            n_bkps (int): number of breakpoints to find before stopping.
            penalty (float): penalty value (>0)
            epsilon (float): reconstruction budget (>0)

        Returns:
            dict: partition dict {(start, end): cost value,...}
        """
        # initialization
        bkps = [self.n_samples]
        stop = False
        while not stop:
            stop = True
            new_bkps = [
                self.single_bkp(start, end) for start, end in pairwise([0] + bkps)
            ]
            bkp, gain = max(new_bkps, key=lambda x: x[1])

            if bkp is None:  # all possible configuration have been explored.
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
        partition = {
            (start, end): self.cost.error(start, end)
            for start, end in pairwise([0] + bkps)
        }
        return partition

    @lru_cache(maxsize=None)
    def single_bkp(self, start, end):
        """Return the optimal breakpoint of [start:end] (if it exists)."""
        segment_cost = self.cost.error(start, end)
        if np.isinf(segment_cost) and segment_cost < 0:  # if cost is -inf
            return None, 0
        gain_list = list()
        for bkp in range(start, end, self.jump):
            if bkp - start >= self.min_size and end - bkp >= self.min_size:
                gain = (
                    segment_cost
                    - self.cost.error(start, bkp)
                    - self.cost.error(bkp, end)
                )
                gain_list.append((gain, bkp))
        try:
            gain, bkp = max(gain_list)
        except ValueError:  # if empty sub_sampling
            return None, 0
        return bkp, gain

    def fit(self, signal) -> "Binseg":
        """Compute params to segment signal.

        Args:
            signal (array): signal to segment. Shape (n_samples, n_features) or (n_samples,).

        Returns:
            self
        """
        # update some params
        if signal.ndim == 1:
            self.signal = signal.reshape(-1, 1)
        else:
            self.signal = signal
        self.n_samples, _ = self.signal.shape
        self.cost.fit(signal)
        self.single_bkp.cache_clear()

        return self

    def predict(self, n_bkps=None, pen=None, epsilon=None):
        """Return the optimal breakpoints.

        Must be called after the fit method. The breakpoints are associated with the
        signal passed to [`fit()`][ruptures.detection.binseg.Binseg.fit].
        The stopping rule depends on the parameter passed to the function.

        Args:
            n_bkps (int): number of breakpoints to find before stopping.
            pen (float): penalty value (>0)
            epsilon (float): reconstruction budget (>0)

        Raises:
            AssertionError: if none of `n_bkps`, `pen`, `epsilon` is set.
            BadSegmentationParameters: in case of impossible segmentation
                configuration

        Returns:
            list: sorted list of breakpoints
        """
        msg = "Give a parameter."
        assert any(param is not None for param in (n_bkps, pen, epsilon)), msg

        # raise an exception in case of impossible segmentation configuration
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

    def fit_predict(self, signal, n_bkps=None, pen=None, epsilon=None):
        """Fit to the signal and return the optimal breakpoints.

        Helper method to call fit and predict once

        Args:
            signal (array): signal. Shape (n_samples, n_features) or (n_samples,).
            n_bkps (int): number of breakpoints.
            pen (float): penalty value (>0)
            epsilon (float): reconstruction budget (>0)

        Returns:
            list: sorted list of breakpoints
        """
        self.fit(signal)
        return self.predict(n_bkps=n_bkps, pen=pen, epsilon=epsilon)
