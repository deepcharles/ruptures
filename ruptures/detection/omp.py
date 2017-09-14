"""

Orthogonal matching pursuit for changepoint detection.

Fast but approximate.

Euclidean norm.

"""
import numpy as np
from numpy.linalg import norm

from ruptures.utils import pairwise
from ruptures.base import BaseEstimator


class Omp(BaseEstimator):

    """Contient l'algorithme de parcours des partitions."""

    def __init__(self, min_size=2, jump=5):
        """Initialize an Omp instance

        Args:
            min_size (int, optional): minimum segment length. Defaults to 2 samples.
            jump (int, optional): subsample (one every "jump" points). Defaults to 5 samples.

        Returns:
            self
        """
        self.min_size = min_size  # not used
        self.jump = jump  # not used
        self.n_samples = None
        self.signal = None

    def _seg(self, n_bkps=None, pen=None, epsilon=None):
        """Computes the binary segmentation.

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
        residual = self.signal
        inds = np.arange(1, self.n_samples)
        correction = 1 / inds + 1 / inds[::-1]

        while not stop:
            res_norm = norm(residual)
            # greedy search
            raw_corr = np.sum(residual.cumsum(axis=0)**2, axis=1)
            correlation = raw_corr[:-1].flatten() * correction
            bkp_opt, _ = max(
                enumerate(correlation, start=1), key=lambda x: x[1])

            # orthogonal projection
            proj = np.zeros(self.signal.shape)
            for (start, end) in pairwise(sorted([0, bkp_opt] + bkps)):
                proj[start:end] = self.signal[start:end].mean(axis=0)
            residual = self.signal - proj

            # stopping criterion
            stop = True
            if n_bkps is not None:
                if len(bkps) - 1 < n_bkps:
                    stop = False
            elif pen is not None:
                if res_norm - norm(residual) > pen:
                    stop = False
            elif epsilon is not None:
                if norm(residual) > epsilon:
                    stop = False
            # update
            if not stop:
                res_norm = norm(residual)
                bkps.append(bkp_opt)

        bkps.sort()
        return bkps

    def fit(self, signal):
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
        self.signal = self.signal - self.signal.mean(axis=0)
        self.n_samples, _ = self.signal.shape

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

        bkps = self._seg(n_bkps=n_bkps, pen=pen, epsilon=epsilon)
        return bkps

    def fit_predict(self, signal, n_bkps=None, pen=None, epsilon=None):
        """Helper method to call fit and predict once."""
        self.fit(signal)
        return self.predict(n_bkps=n_bkps, pen=pen, epsilon=epsilon)
