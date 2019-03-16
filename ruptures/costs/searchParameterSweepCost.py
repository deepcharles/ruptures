import sys

from ruptures.base import BaseCost

import numpy as np


class GaussianMixtureCost(BaseCost):
    model = ""
    min_size = 2

    def __init__(self, n_components, cost):
        self.n_components = n_components
        self.cost = cost
        self.signal = []
        self.segment_costs = [[]]

    def fit(self, signal):
        """Set the internal parameter."""
        self.signal = signal
        self.cost.fit(signal)
        signal_length = len(signal)
        self.segment_costs = np.empty((signal_length, signal_length), dtype=float)
        for i in range(signal_length):
            for j in range(signal_length):
                self.segment_costs[i][j] = sys.float_info.max
        return self

    def error(self, start, end):
        """Return the approximation cost on the segment [start:end].

        Args:
            start (int): start of the segment
            end (int): end of the segment

        Returns:
            float: segment cost
        """
        if self.segment_costs[start][end - 1] == sys.float_info.max:
            segment_cost = self.cost.error(start, end)
            self.segment_costs[start][end - 1] = segment_cost
        else:
            segment_cost = self.segment_costs[start][end - 1]

        return segment_cost
