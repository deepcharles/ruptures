import numpy as np
from ruptures.utils.memoizedict import MemoizeDict


class NotEnoughPoints(Exception):
    """Raise this exception when there is not enough point to calculate a
    cost function """
    pass


def gaussmean(signal):
    """
    Returns the cost function used in Pelt.
    Here: - max log likelihood of a univariate gaussian random variable.
    Associated K (see Pelt.__init__): 0

    signal: array of shape (n_points,) or (n_points, 1)
    """
    s = signal
    if s.ndim == 1:
        s = s.reshape(-1, 1)

    @MemoizeDict
    def error_func(start, end):
        n = s.shape[0]
        assert 0 <= start <= end < n

        if end - start <= 0:  # we need at least 2 points
            raise NotEnoughPoints
        ind = np.arange(start, end + 1)
        sig = s[ind]
        m, v = sig.mean(), sig.var()
        if v == 0:
            return np.inf
        res = np.sum((sig - m) ** 2)
        res /= 2 * v
        res += (end - start + 1) / 2 * np.log(v)
        res += (end - start + 1) / 2 * np.log(2 * np.pi)
        return res

    return error_func
