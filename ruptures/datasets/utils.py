import numpy as np


def uniform_with_constant_sum(n, s):
    """
    Returns n random non negative integers which sum to s.
    """
    v = np.random.rand(n - 1)
    u = - np.log(v)
    u /= np.sum(u)
    res = np.array(u * s, dtype=int)
    res = np.append(res, s - res.sum())
    assert res.sum() == s
    np.random.shuffle(res)
    return res
