import numpy as np
import ruptures.pelt.costs as costs
from ruptures.utils.memoizedict import MemoizeDict


def test_gaussmean():
    n_samples, n_features = 100, 1
    s = np.random.randn(n_samples, n_features)
    func = costs.gaussmean(s)
    assert func(0, n_samples - 1) > 0
    assert isinstance(func, MemoizeDict)

    func = costs.gaussmean(np.ones((n_samples, n_features)))
    assert func(0, n_samples - 1) == np.inf
