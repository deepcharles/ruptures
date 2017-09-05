"""

Sliding-window changepoint detection.

Fast and approximate.

"""
import numpy as np
from scipy.ndimage import convolve as sconv

from ruptures.costs import Cost
from ruptures.utils import pairwise


class Window:

    """Window sliding method."""

    def __init__(self, width=100):
        """Instanciate with window length.

        Args:
            width (int): window lenght.

        Returns:
            self
        """
        self.width = 2 * (width // 2)
        # window
        self.win = np.sign(np.linspace(-1, 1, self.width)).reshape(-1, 1)
        self.n_samples = None
        self.signal = None
        self.cost = Cost(model="constantl2")
        self.score = None

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
        error = self.cost.error(0, self.n_samples)

        while not stop:
            stop = True
            _, bkp = max((v, k) for k, v in enumerate(self.score, start=1)
                         if not any(abs(k - b) < self.width // 2 for b in bkps[:-1]))

            if n_bkps is not None:
                if len(bkps) - 1 < n_bkps:
                    stop = False
            elif pen is not None:
                new_error = sum(self.cost.error(start, end)
                                for start, end in pairwise(sorted([0, bkp] + bkps)))
                gain = error - new_error
                if gain > pen:
                    stop = False
                    error = sum(self.cost.error(start, end)
                                for start, end in pairwise([0] + bkps))
            elif epsilon is not None:
                if error > epsilon:
                    stop = False
                    error = sum(self.cost.error(start, end)
                                for start, end in pairwise([0] + bkps))

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
        self.n_samples, _ = self.signal.shape

        self.cost.fit(signal)
        # compute score
        convolution = sconv(self.signal, self.win, mode='mirror')
        self.score = np.sum(convolution**2, axis=1)
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

    def fit_predict(self, signal, n_bkps=None, pen=None, epsilon=None):
        """Helper method to call fit and predict once."""
        self.fit(signal)
        return self.predict(n_bkps=n_bkps, pen=pen, epsilon=epsilon)
