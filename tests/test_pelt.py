import numpy as np
import ruptures.pelt.Pelt as p
import ruptures.pelt.costs as costs
import sklearn.datasets.samples_generator as s_generator

#
# class signal(object):
#     """dummy class for testing"""
#
#     def __init__(self, s):
#         if s.ndim == 1:
#             self.s = s.reshape(-1, 1)
#         else:
#             self.s = s
#         self.n = self.s.shape[0]
#
#     def error_func(self, start, end):
#         """
#         - max log likelihood, univariate gaussian
#         """
#         assert 0 <= start <= end < self.n
#         ind = np.arange(start, end + 1)
#         sig = self.s[ind]
#         m, v = 0, (sig ** 2).mean()
#         if v == 0:
#             return np.inf
#         res = np.sum((sig - m) ** 2)
#         res /= 2 * v
#         res += (end - start + 1) / 2 * np.log(v)
#         res += (end - start + 1) / 2 * np.log(2 * np.pi)
#         return res


def test_ruptures1D():
    n_ruptures = 5
    n_samples, n_features = 500, 1
    stds = np.linspace(1, 20, n_ruptures)
    sig, y = s_generator.make_blobs(n_samples=n_samples, centers=n_ruptures,
                                    n_features=n_features,
                                    cluster_std=stds,
                                    shuffle=False)
    # vraies ruptures
    # vraies_ruptures = [
    # k + 1 for (k, (v, w)) in enumerate(zip(y[:-1], y[1:])) if v != w]

    # import matplotlib.pyplot as plt
    # plt.plot(sig)
    # plt.show()
    c = costs.gaussmean(sig)
    for pen in np.logspace(0.1, 100, 20):
        pe = p.Pelt(c, penalty=pen, n=sig.shape[0], K=0)
        pe.fit()
