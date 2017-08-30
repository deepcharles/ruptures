"""

Greedy change point detection for piecewise linear signals.
Sequential algorithm.

"""
import numpy as np
from numpy.linalg import lstsq
from ruptures.utils import pairwise


class GreedyLinear:

    """Greedy change point detection for piecewise linear processes."""

    def __init__(self, jump=10):
        """

        Args:
            jump (int, optional): only consider change points multiple of *jump*. Defaults to 10.

        Returns:
            self
        """
        self.jump = jump
        self.n_samples = None
        self.signal = None
        self.covariates = None
        self.dim = None

    def seg(self, n_bkps=None, pen=None, epsilon=None):
        """Computes the greedy segmentation.

        The stopping rule depends on the parameter passed to the function.

        Args:
            n_bkps (int): number of breakpoints to find before stopping.
            penalty (float): penalty value (>0)
            epsilon (float): reconstruction budget

        Returns:
            list: list of breakpoint indexes
        """
        stop = False
        bkps = [self.n_samples]
        inds = np.arange(self.jump, self.n_samples - self.jump, self.jump)
        residual = self.signal
        res_norm = residual.var() * self.n_samples

        while not stop:
            # greedy search
            res_list = list()
            for ind in inds:  # greedy search
                res_tmp = 0
                y_left, y_right = residual[:ind], residual[ind:]
                x_left, x_right = self.covariates[:ind], self.covariates[ind:]
                for x, y in zip((x_left, x_right), (y_left, y_right)):
                    # linear fit
                    _, res_sub, _, _ = lstsq(x, y)
                    # error on sub-signal
                    res_tmp += res_sub
                res_list.append(res_tmp)
            # find best index
            _, bkp_opt = min(zip(res_list, inds))

            # orthogonal projection
            proj = np.zeros(self.signal.shape)
            for start, end in pairwise(sorted([0, bkp_opt] + bkps)):
                y = self.signal[start:end]
                x = self.covariates[start:end]
                coef, _, _, _ = lstsq(x, y)
                proj[start:end] = x.dot(coef).reshape(-1, 1)
            residual = self.signal - proj

            # stopping criterion
            stop = True
            if n_bkps is not None:
                if len(bkps) - 1 < n_bkps:
                    stop = False
            elif pen is not None:
                if res_norm - residual.var() * self.n_samples > pen:
                    stop = False
            elif epsilon is not None:
                if residual.var() * self.n_samples > epsilon:
                    stop = False
            # update
            if not stop:
                res_norm = residual.var() * self.n_samples
                bkps.append(bkp_opt)

            bkps.sort()
        return bkps

    def fit(self, signal, covariates):
        """Compute params to segment signal.

        Args:
            signal (array): univariate signal to segment. Shape (n_samples, 1) or (n_samples,).
            covariates (array): covariates. Shape (n_samples, n_features).

        Returns:
            self
        """
        # update some params
        if signal.ndim == 1:
            self.signal = signal.reshape(-1, 1) - signal.mean()
        else:
            self.signal = signal - signal.mean()
        self.n_samples, dim = self.signal.shape
        assert dim == 1, "Signal must be 1D."

        self.covariates = covariates
        _, self.dim = self.covariates.shape
        assert covariates.ndim == 2, "Reshape the covariates."
        assert covariates.shape[0] == self.n_samples, "Check size."

        return self

    def predict(self, n_bkps=None, pen=None, epsilon=None):
        """Return the optimal breakpoints.

        Must be called after the fit method. The breakpoints are associated with the signal passed
        to fit().
        The stopping rule depends on the parameter passed to the function.

        Args:
            n_bkps (int): number of breakpoints to find before stopping.
            penalty (float): penalty value (>0)
            penalty (float): penalty value

        Returns:
            list: sorted list of breakpoints
        """
        msg = "Give a parameter."
        assert any(param is not None for param in (n_bkps, pen, epsilon)), msg

        bkps = self.seg(n_bkps=n_bkps, pen=pen, epsilon=epsilon)
        return bkps

    def fit_predict(self, signal, covariates, n_bkps=None, pen=None, epsilon=None):
        """Helper method to call fit and predict once."""
        self.fit(signal, covariates)
        return self.predict(n_bkps=n_bkps, pen=pen, epsilon=epsilon)
