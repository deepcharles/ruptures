import numpy as np
from ruptures.costs.exceptions import NotEnoughPoints
from np.fft import rfftfreq


def sinus_mse(signal):
    """
    Returns the cost function.
    Here:
    Mean squared error when doing a regression against waves at the fourier
    frequencies (i / n for n in range(n / 2))
    Associated K (see Pelt.__init__): 0

    signal: array of shape (n_points,) or (n_points, 1)
    """
    s = signal
    if s.ndim == 1:
        s = s.reshape(-1, 1)

    def error_func(start, end):
        assert 0 <= start <= end

        if end - start < 20:  # we need at least 20 points
            raise NotEnoughPoints

        # we only consider signals with an even number of points.
        n = ((end - start) // 2) * 2
        sig = s[start:(start + n)]

        # frequencies at which we do the regression
        freqs = np.fft.rfftfreq(n)
        temps = np.arange(n) * np.pi * 2

        regressors = [np.column_stack((np.cos(w * temps), np.sin(w * temps)))
                      for w in freqs[1:-1]]

        X = np.column_stack([np.ones(n)] + regressors +
                            [np.cos(freqs[-1] * temps)])

        hat_matrix = np.diag([1] + [2] * (n - 2) + [1]) / n
        beta = hat_matrix.dot(X.T).dot(sig)
        residuals = sig.reshape(-1, 1) - X.dot(beta)

        return residuals.mean()

    return error_func
