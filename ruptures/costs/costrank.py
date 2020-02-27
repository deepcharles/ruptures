r"""
.. _sec-costrank:

Rank-based cost function
====================================================================================================

Description
----------------------------------------------------------------------------------------------------

This cost function detects general distribution changes in multivariate signals, using a rank transformation :cite:`rank-Lung-Yut-Fong2015`.
Formally, for a signal :math:`\{y_t\}_t` on an interval :math:`[a, b)`,

    .. math:: c_{rank}(a, b) = -(b - a) \bar{r}_{a..b}' \hat{\Sigma}_r^{-1} \bar{r}_{a..b}

where :math:`\bar{r}_{a..b}` is the empirical mean of the sub-signal
:math:`\{r_t\}_{t=a+1}^b`, and :math:`\hat{\Sigma}_r` is the covariance matrix of the
complete rank signal :math:`r`.

Usage
----------------------------------------------------------------------------------------------------

Start with the usual imports and create a signal.

.. code-block:: python

    import numpy as np
    import matplotlib.pylab as plt
    import ruptures as rpt
    # creation of data
    n, dim = 500, 3  # number of samples, dimension
    n_bkps, sigma = 3, 5  # number of change points, noise standard deviation
    signal, bkps = rpt.pw_constant(n, dim, n_bkps, noise_std=sigma)

Then create a :class:`CostRank` instance and print the cost of the sub-signal :code:`signal[50:150]`.

.. code-block:: python

    c = rpt.costs.CostRank().fit(signal)
    print(c.error(50, 150))


You can also compute the sum of costs for a given list of change points.

.. code-block:: python

    print(c.sum_of_costs(bkps))
    print(c.sum_of_costs([10, 100, 200, 250, n]))


In order to use this cost class in a change point detection algorithm (inheriting from :class:`BaseEstimator`), either pass a :class:`CostRank` instance (through the argument ``'custom_cost'``) or set :code:`model="rank"`.

.. code-block:: python

    c = rpt.costs.CostRank(); algo = rpt.Dynp(custom_cost=c)
    # is equivalent to
    algo = rpt.Dynp(model="rank")


Code explanation
----------------------------------------------------------------------------------------------------

.. autoclass:: ruptures.costs.CostRank
    :members:
    :special-members: __init__


.. rubric:: References

.. bibliography:: ../biblio.bib
    :style: alpha
    :cited:
    :labelprefix: RA
    :keyprefix: rank-
"""
import numpy as np
from numpy.linalg import pinv, LinAlgError
from scipy.stats.mstats import rankdata

from ruptures.base import BaseCost
from ruptures.costs import NotEnoughPoints


class CostRank(BaseCost):
    r"""
    Rank-based cost function
    """

    model = "rank"

    def __init__(self):
        self.inv_cov = None
        self.ranks = None
        self.min_size = 2

    def fit(self, signal):
        """Set parameters of the instance.

        Args:
            signal (array): signal. Shape (n_samples,) or (n_samples, n_features)

        Returns:
            self
        """
        if signal.ndim == 1:
            signal = signal.reshape(-1, 1)

        obs, vars = signal.shape

        # Convert signal data into ranks in the range [1, n]
        ranks = rankdata(signal, axis=0)
        # Center the ranks into the range [-(n+1)/2, (n+1)/2]
        centered_ranks = (ranks - ((obs + 1) / 2))
        # Sigma is the covariance of these ranks.
        # If it's a scalar, reshape it into a 1x1 matrix
        cov = np.cov(centered_ranks, rowvar=False,
                     bias=True).reshape(vars, vars)

        # Use the pseudoinverse to handle linear dependencies
        # see Lung-Yut-Fong, A., Lévy-Leduc, C., & Cappé, O. (2015)
        try:
            self.inv_cov = pinv(cov)
        except LinAlgError as e:
            raise LinAlgError(
                "The covariance matrix of the rank signal is not invertible and the "
                "pseudo-inverse computation did not converge."
            ) from e
        self.ranks = centered_ranks

        return self

    def error(self, start, end):
        """Return the approximation cost on the segment [start:end].

        Args:
            start (int): start of the segment
            end (int): end of the segment

        Returns:
            float: segment cost

        Raises:
            NotEnoughPoints: when the segment is too short (less than ``'min_size'`` samples).
        """
        if end - start < self.min_size:
            raise NotEnoughPoints

        mean = np.reshape(np.mean(self.ranks[start:end], axis=0), (-1, 1))

        return -(end - start) * mean.T @ self.inv_cov @ mean
