r"""Window-based change point detection."""


import numpy as np
from scipy.signal import argrelmax

from ruptures.base import BaseCost, BaseEstimator
from ruptures.costs import cost_factory
from ruptures.utils import unzip, sanity_check
from ruptures.exceptions import BadSegmentationParameters


class Window(BaseEstimator):
    """Window sliding method."""

    def __init__(
        self, width=100, model="l2", custom_cost=None, min_size=2, jump=5, params=None
    ):
        """Instanciate with window length.

        Args:
            width (int, optional): window length. Defaults to 100 samples.
            model (str, optional): segment model, ["l1", "l2", "rbf"]. Not used if `custom_cost` is not None.
            custom_cost (BaseCost, optional): custom cost function. Defaults to None.
            min_size (int, optional): minimum segment length.
            jump (int, optional): subsample (one every *jump* points).
            params (dict, optional): a dictionary of parameters for the cost instance.`
        """
        self.min_size = min_size
        self.jump = jump
        self.width = 2 * (width // 2)
        self.n_samples = None
        self.signal = None
        self.inds = None
        if custom_cost is not None and isinstance(custom_cost, BaseCost):
            self.cost = custom_cost
        else:
            if params is None:
                self.cost = cost_factory(model=model)
            else:
                self.cost = cost_factory(model=model, **params)
        self.score = list()

    def _seg(self, n_bkps=None, pen=None, epsilon=None):
        """Sequential peak search.

        The stopping rule depends on the parameter passed to the function.

        Args:
            n_bkps (int): number of breakpoints to find before stopping.
            penalty (float): penalty value (>0)
            epsilon (float): reconstruction budget (>0)

        Returns:
            list: breakpoint index list
        """
        # initialization
        bkps = [self.n_samples]
        stop = False
        error = self.cost.sum_of_costs(bkps)
        # peak search
        # forcing order to be above one in case jump is too large (issue #16)
        order = max(max(self.width, 2 * self.min_size) // (2 * self.jump), 1)
        peak_inds_shifted = argrelmax(self.score, order=order, mode="wrap")[0]

        if peak_inds_shifted.size == 0:  # no peaks if the score is constant
            return bkps
        gains = np.take(self.score, peak_inds_shifted)
        peak_inds_arr = np.take(self.inds, peak_inds_shifted)
        # sort according to score value
        _, peak_inds = unzip(sorted(zip(gains, peak_inds_arr)))
        peak_inds = list(peak_inds)

        while not stop:
            stop = True
            # _, bkp = max((v, k) for k, v in enumerate(self.score, start=1)
            # if not any(abs(k - b) < self.width // 2 for b in bkps[:-1]))

            try:
                # index with maximum score
                bkp = peak_inds.pop()
            except IndexError:  # peak_inds is empty
                break

            if n_bkps is not None:
                if len(bkps) - 1 < n_bkps:
                    stop = False
            elif pen is not None:
                gain = error - self.cost.sum_of_costs(sorted([bkp] + bkps))
                if gain > pen:
                    stop = False
            elif epsilon is not None:
                if error > epsilon:
                    stop = False

            if not stop:
                bkps.append(bkp)
                bkps.sort()
                error = self.cost.sum_of_costs(bkps)

        return bkps

    def fit(self, signal) -> "Window":
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
        # indexes
        self.inds = np.arange(self.n_samples, step=self.jump)
        # delete borders
        keep = (self.inds >= self.width // 2) & (
            self.inds < self.n_samples - self.width // 2
        )
        self.inds = self.inds[keep]
        self.cost.fit(signal)
        # compute score
        score = list()
        for k in self.inds:
            start, end = k - self.width // 2, k + self.width // 2
            gain = self.cost.error(start, end)
            if np.isinf(gain) and gain < 0:
                # segment is constant and no improvment possible on start .. end
                score.append(0)
                continue
            gain -= self.cost.error(start, k) + self.cost.error(k, end)
            score.append(gain)
        self.score = np.array(score)
        return self

    def predict(self, n_bkps=None, pen=None, epsilon=None):
        """Return the optimal breakpoints.

        Must be called after the fit method. The breakpoints are associated with the signal passed
        to [`fit()`][ruptures.detection.window.Window.fit].
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
        # raise an exception in case of impossible segmentation configuration
        if not sanity_check(
            n_samples=self.cost.signal.shape[0],
            n_bkps=0 if n_bkps is None else n_bkps,
            jump=self.jump,
            min_size=self.min_size,
        ):
            raise BadSegmentationParameters

        msg = "Give a parameter."
        assert any(param is not None for param in (n_bkps, pen, epsilon)), msg

        bkps = self._seg(n_bkps=n_bkps, pen=pen, epsilon=epsilon)
        return bkps

    def fit_predict(self, signal, n_bkps=None, pen=None, epsilon=None):
        """Helper method to call fit and predict once."""
        self.fit(signal)
        return self.predict(n_bkps=n_bkps, pen=pen, epsilon=epsilon)
