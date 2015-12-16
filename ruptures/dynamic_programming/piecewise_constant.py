from ruptures.base import BaseEstimator
import numpy as np
import ruptures.dynamic_programming.dynamic_programming as d_prog


class ConstantError:

    def __init__(self, signal):
        s = np.array(signal)
        if s.ndim == 1:
            self.s = s.reshape(-1, 1)
        else:
            self.s = s
        self.n = self.s.shape[0]
        self.gram = np.dot(self.s, self.s.T)
        self.norm = np.diag(self.gram).cumsum()
        self.integral_m = self.gram.cumsum(axis=0).cumsum(axis=1)
        self.partition = dict()

    def error_func(self, start, end):
        """
        This computes the error when the segment [start, end] is approached by
        a constant (its mean):
        Let m denote the integer segment [start, end]
        error = sum_{i in m} norm(X_i)^2  - 1/(end - start + 1)
            sum_{i,j in m} <X_i|X_j>
        :param start: the first index of the segment
        :param end: the last index of the signal
        :return: float. The approximation error
        """
        if start == 0:
            return (- self.integral_m[end, end] / (end - start + 1) +
                    self.norm[end])
        else:
            res = self.integral_m[start - 1, start - 1]
            res += self.integral_m[end, end]
            res -= self.integral_m[start - 1, end]
            res -= self.integral_m[end, start - 1]
            res /= end - start + 1
            res *= - 1.
            res += self.norm[end] - self.norm[start - 1]
            return res


class Constant(BaseEstimator):

    def __init__(self, signal):
        self.error = ConstantError(signal)
        self.n = self.error.n

    def fit(self, d, jump=1, min_size=1):
        self.partition = d_prog.dynamic_prog(
            self.error.error_func, d, 0, self.n - 1, jump, min_size)
        return self.partition

# Not run
# if __name__ == '__main__':
#     import numpy as np
#     import matplotlib.pyplot as plt
#
#     n_samples = 200
#     time = np.linspace(0, 12, n_samples)
#     # 2 ruptures
#     sig = np.sign(np.sin(0.7 * time))
#     sig += 0.2 * np.random.normal(size=sig.shape)
#
#     c = Constant(sig)
#     res = c.fit(3, 1, 2)
#     print(res)
#
#     ruptures = [s for (s, e) in res.keys() if s != 0]
#     plt.plot(sig)
#     plt.vlines(ruptures, ymin=np.min(sig), ymax=np.max(sig))
#     plt.show()
