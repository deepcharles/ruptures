import warnings

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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted_model = self.model.fit(sub)
        return - fitted_model.score(sub) * len(sub)
