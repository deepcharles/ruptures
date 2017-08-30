"""

Sliding-window changepoint detection with Mmd test.

Fast and approximate.

"""
import numpy as np
from scipy.signal import fftconvolve

from ruptures.utils import pairwise


class WinMmd:

    """Window sliding method for the Mmd test."""

    def __init__(self, width=100):
        """Instanciate with window length.

        Args:
            width (int): window lenght.

        Returns:
            self
        """
        self.width = 2 * (width // 2)
        self.n_samples = None
        self.gram = None
        self.score = None
        # window
        halfw = self.width // 2
        self.win = np.ones((self.width, self.width))
        self.win[halfw:, :halfw] = -1
        self.win[:halfw, halfw:] = -1
        self.win /= halfw**2

    def seg(self, n_bkps=None, pen=None, epsilon=None):
        """Computes the binary segmentation.

        The stopping rule depends on the parameter passed to the function.

        Args:
            n_bkps (int): number of breakpoints to find before stopping.
            penalty (float): penalty value (>0)
            penalty (float): penalty value

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

    def fit(self, gram):
        """Compute params to segment signal.

        Args:
            gram (array): gram matrix to segment. Shape (n_samples, n_samples).

        Returns:
            self
        """
        assert gram.shape[0] == gram.shape[1], "Not a square matrix."
        # update some params
        self.gram = gram
        self.n_samples, _ = self.gram.shape
        # convolution
        self.score = np.diag(fftconvolve(self.gram, self.win, mode='valid'))
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

    def fit_predict(self, gram, n_bkps=None, pen=None, epsilon=None):
        """Helper method to call fit and predict once."""
        self.fit(gram)
        return self.predict(n_bkps=n_bkps, pen=pen, epsilon=epsilon)
