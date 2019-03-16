import warnings

import numpy as np

from ruptures.base import BaseCost
from sklearn.mixture import GaussianMixture


class GaussianMixtureCost(BaseCost):
    model = ""
    min_size = 2

    def __init__(self, n_components):
        self.n_components = n_components
        self.model = GaussianMixture(n_components=n_components)
        self.signal = []

    def fit(self, signal):
        """Set the internal parameter."""
        self.signal = signal
        return self

    def error(self, start, end):
        """Return the approximation cost on the segment [start:end].

        Args:
            start (int): start of the segment
            end (int): end of the segment

        Returns:
            float: segment cost
        """
        sub = self.signal[start:end]
        if self.fitted_models[start][end - 1] is not None:
            fitted_model = self.fitted_models[start][end - 1]
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted_model = self.model.fit(sub)
                self.fitted_models[start][end - 1] = fitted_model
        return - fitted_model.score(sub) * len(sub)
