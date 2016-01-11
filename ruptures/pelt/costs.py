import numpy as np
from numpy.linalg import lstsq


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

    def error_func(start, end):
        assert 0 <= start <= end

        if end - start < 2:  # we need at least 2 points
            raise NotEnoughPoints

        sig = s[start:end]
        m, v = sig.mean(), sig.var()
        if v == 0:
            return np.inf
        res = np.sum((sig - m) ** 2)
        res /= 2 * v
        res += (end - start + 1) / 2 * np.log(v)
        res += (end - start + 1) / 2 * np.log(2 * np.pi)
        return res

    return error_func


def linear_mse(signal):
    """
    Returns the cost function used in Pelt.
    Here: mean squared error when approximating with a linear function.
    Associated K (see Pelt.__init__): 0

    signal: array of shape (n_points,) or (n_points, 1) or (n_points, d)
    """
    s = signal
    if s.ndim == 1:
        s = s.reshape(-1, 1)

    def error_func(start, end):
        assert 0 <= start <= end

        if end - start < 2:  # we need at least 2 points
            raise NotEnoughPoints

        sig = s[start:end]

        # we regress over the time variable
        tt = np.arange(start, end)
        a = np.column_stack((tt, np.ones(tt.shape)))
        assert a.shape[0] == sig.shape[0]
        # doing the regression
        res_lstsq = lstsq(a, sig)
        assert res_lstsq[1].shape[0] == sig.shape[1]

        # mean squared error
        res = np.sum(res_lstsq[1]) / sig.shape[0]

        return res

    return error_func
