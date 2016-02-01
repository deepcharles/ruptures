import numpy as np
from numpy.linalg import lstsq
from ruptures.search_methods import changepoint
from ruptures.costs import NotEnoughPoints


@changepoint
class LinearMSE:

    def error(self, start, end):
        """
        Cost: Squared error when approximating with the best linear fit.
        Associated K (see Pelt): 0

        Args:
            start (int): first index of the segment (index included)
            end (int): last index of the segment (index excluded)

        Raises:
            NotEnoughPoints: if there are not enough points to compute the cost
            for the specified segment (here, at least 3 points).

        Returns:
            float: cost on the given segment.
        """

        if end - start < 3:  # we need at least 2 points
            raise NotEnoughPoints

        sig = self.signal[start:end]
        # we regress over the time variable
        tt = np.arange(start, end)
        a = np.column_stack((tt, np.ones(tt.shape)))
        assert a.shape[0] == sig.shape[0]
        # doing the regression
        res_lstsq = lstsq(a, sig)
        assert res_lstsq[1].shape[0] == sig.shape[1]

        # mean squared error
        res = np.sum(res_lstsq[1]) / sig.shape[0]

        return res

    def set_params(self):
        pass

    @property
    def K(self):
        return 0
