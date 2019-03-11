import numpy as np

from ruptures.base import BaseCost


class CombinatorialCost(BaseCost):
    model = ""
    min_size = 2

    def __init__(self, costs):
        self.signal = []
        self.costs = costs

    def fit(self, signal):
        """Set the internal parameter."""
        self.signal = signal
        for i in range(len(self.costs)):
            ith_row = self.get_ith_row(self.signal, i)
            self.costs[i].fit(ith_row)
        return self

    def get_ith_row(self, signal, i):
        ith_row = signal[:, i]
        return ith_row[:, np.newaxis]

    def error(self, start, end):
        """Return the approximation cost on the segment [start:end].

        Args:
            start (int): start of the segment
            end (int): end of the segment

        Returns:
            float: segment cost
        """

        total_error = 0
        for i in range(len(self.costs)):
            total_error += self.costs[i].error(start, end)
        return total_error
