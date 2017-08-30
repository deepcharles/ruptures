"""

Sliding-window changepoint detection with likelihood ratio for multivariate gaussian.

"""
import numpy as np
from numpy import cov
from numpy.linalg import det


class WinLr:

    """Window sliding method for the Box's M test (for two multivariate gaussian variables)."""

    @staticmethod
    def stat(left, right):
        """Computes likelihood ratio."""
        cleft, cright = cov(left), cov(right)
        return det((cleft + cright) / 2)**2 / det(cleft) / det(cright)

    def __init__(self, width=100):
        """Instanciate with window length.

        Args:
            width (int): window lenght.

        Returns:
            self
        """
        self.width = 2 * (width // 2)
        self.n_samples = None
        self.signal = None
        self.score = None

    def seg(self, n_bkps=None, pen=None, epsilon=None):
        """Computes the binary segmentation.

        The stopping rule depends on the parameter passed to the function.

        Args:
            n_bkps (int): number of breakpoints to find before stopping.
            penalty (float): penalty value (>0) (not used)
            epsilon (float): error budget (not used)

        Returns:
            list: breakpoint index list
        """

        # initialization
        bkps = [self.n_samples]
        stop = False

        while not stop:
            stop = True
            _, bkp = max((v, k) for k, v in enumerate(self.score, start=self.width // 2)
                         if not any(abs(k - b) < self.width // 2 for b in bkps[:-1]))

            if n_bkps is not None:
                if len(bkps) - 1 < n_bkps:
                    stop = False
            elif pen is not None:
                pass
            elif epsilon is not None:
                pass

            if not stop:
                bkps.append(bkp)
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
        self.n_samples, dim = self.signal.shape
        # score
        shape = (self.n_samples - self.width + 1, dim, self.width)
        strides = (self.signal.strides[
                   0], self.signal.strides[1], self.signal.strides[0])
        traj = np.lib.stride_tricks.as_strided(
            self.signal, shape=shape, strides=strides)
        self.score = [self.stat(x[:, :self.width // 2], x[:, self.width // 2:])
                      for x in traj]
        return self

    def predict(self, n_bkps=None, pen=None, epsilon=None):
        """Return the optimal breakpoints.

        Must be called after the fit method. The breakpoints are associated with the signal passed
        to fit().
        The stopping rule depends on the parameter passed to the function.

        Args:
            n_bkps (int): number of breakpoints to find before stopping.
            penalty (float): penalty value (>0) (not used)
            epsilon (float): error budget (not used)

        Returns:
            list: sorted list of breakpoints
        """
        msg = "Give a parameter."
        assert any(param is not None for param in (n_bkps, pen, epsilon)), msg

        bkps = self.seg(n_bkps=n_bkps, pen=pen, epsilon=epsilon)
        return bkps

    def fit_predict(self, signal, n_bkps=None, pen=None, epsilon=None):
        """Helper method to call fit and predict once."""
        self.fit(signal)
        return self.predict(n_bkps=n_bkps, pen=pen, epsilon=epsilon)
