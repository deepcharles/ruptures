"""Implementation of the lasso regression"""
from copy import deepcopy

import numpy as np
from numpy.linalg import norm


def soft_thresholding_operator(vect, thresh):
    """Soft thresholding"""
    thresh = abs(thresh)
    zer = np.zeros(vect.shape)
    res = np.where(vect > thresh, vect - thresh, zer)
    res = np.where(vect < thresh, vect + thresh, res)
    return res


class Lasso:

    """Lasso regression with coordinate gradient descent.

    minimize  ||Y - XW||^Fro_2 + alpha * ||W||_21
    over W
    where |W||_21 = sum_i sqrt{sum_j w_{ij}^2}
    """

    def __init__(self, alpha=1.0, max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.coef_ = None

    def fit(self, X, y):
        """Perform the coordinate gradient descent.

        Args:
            X (array): (n_samples, n_features), should be normalized.
            y (array): (n_samples, n_target), should have zero mean.

        Returns:
            self: description
        """
        # check if normalized
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        _, n_features = X.shape
        _, n_target = y.shape
        msg = "Matrix X has not been normalized."
        assert np.allclose(X.mean(axis=0), np.zeros(n_features)), msg
        assert np.allclose(norm(X, axis=0), np.ones(n_features)), msg
        msg = "Signal y has not been de-meaned"
        assert np.allclose(y.mean(axis=0), np.zeros(n_target)), msg

        beta = np.zeros((n_features, n_target))

        for _ in range(self.max_iter):
            for j in range(n_features):
                tmp_beta = deepcopy(beta)
                tmp_beta[j, :] = 0.
                r_j = y - X.dot(tmp_beta)
                arg1 = np.dot(X[:, j], r_j)  # shape (n_target,)
                arg2 = self.alpha  # penalty

                beta[j] = soft_thresholding_operator(arg1, arg2)
                beta[j] /= (X[:, j]**2).sum()

        self.coef_ = beta
        return self

    def predict(self, X):
        """Predict the response to a regressor matrix X."""
        return np.dot(X, self.coef_)
