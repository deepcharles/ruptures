from ruptures.search_methods import changepoint
from ruptures.costs import NotEnoughPoints
import numpy as np


@changepoint
class ConstantL2:

    def error(self, start, end):
        """
        Cost: Squared error when approximating with a constant value (its
        mean).
        Associated K (see Pelt): 0

        Args:
            start (int): first index of the segment (index included)
            end (int): last index of the segment (index excluded)

        Returns:
            float: L2 cost on the segment

        Raises:
            NotEnoughPoints: if there are not enough points to compute the cost
            for the specified segment (here, at least 2 points).
        """

        if end - start < 2:  # we need at least 2 points
            raise NotEnoughPoints

        sig = self.signal[start:end]
        v = sig.var(axis=0).sum()
        return v * (end - start)

    def set_params(self):
        pass

    @property
    def K(self):
        return 0


@changepoint
class GaussMLE:

    def error(self, start, end):
        """Associated K (see Pelt.__init__): 0

        Args:
            start (int): first index of the segment (index included)
            end (int): last index of the segment (index excluded)

        Raises:
            NotEnoughPoints: if there are not enough points to compute the cost
            for the specified segment (here, at least 2 points).

        Returns:
            function: cost function. Here: - max log likelihood of a univariate
            gaussian random variable.

        """
        assert 0 <= start <= end
        if end - start < 2:  # we need at least 2 points
            raise NotEnoughPoints

        sig = self.signal[start:end]
        v = sig.var(axis=0)
        if np.sum(np.abs(v)) == 0:
            return np.inf
        res = 1 + np.log(2 * np.pi * v)
        res = res.sum()
        res *= (end - start) / 2
        return res

    def set_params(self):
        pass

    @property
    def K(self):
        return 0


@changepoint
class ConstantL1:

    def error(self, start, end):
        """Cost: minimum L1 error when approximating with a constant value.
        Associated K (see Pelt): 0

        Args:
            start (int): first index of the segment (index included)
            end (int): last index of the segment (index excluded)

        Returns:
            float: L1 cost on the segment

        Raises:
            NotEnoughPoints: if there are not enough points to compute the cost
        """
        if end - start < 2:  # we need at least 2 points
            raise NotEnoughPoints

        sig = self.signal[start:end]
        v = np.abs(sig - np.median(sig, axis=0)).sum()
        return v

    def set_params(self):
        pass

    @property
    def K(self):
        return 0
