r"""
.. _sec-costl1:

Least absolute deviation
====================================================================================================

Description
----------------------------------------------------------------------------------------------------

This cost function detects changes in the median of a signal.
Overall, it is a robust estimator of a shift in the central point (mean, median, mode) of a distribution :cite:`c1-Bai1995`.
Formally, for a signal :math:`\{y_t\}_t` on an interval :math:`I`,

    .. math:: c(y_{I}) = \sum_{t\in I} \|y_t - \bar{y}\|_1

where :math:`\bar{y}` is the componentwise median of :math:`\{y_t\}_{t\in I}`.

Usage
----------------------------------------------------------------------------------------------------

Start with the usual imports and create a signal.

.. code-block:: python

    import numpy as np
    import matplotlib.pylab as plt
    import ruptures as rpt
    # creation of data
    n, dim = 500, 3  # number of samples, dimension
    n_bkps, sigma = 3, 5  # number of change points, noise standart deviation
    signal, bkps = rpt.pw_constant(n, dim, n_bkps, noise_std=sigma)

Then create a :class:`CostL1` instance and print the cost of the sub-signal :code:`signal[50:150]`.

.. code-block:: python

    c = rpt.costs.CostL1().fit(signal)
    print(c.error(50, 150))


You can also compute the sum of costs for a given list of change points.

.. code-block:: python

    print(c.sum_of_costs(bkps))
    print(c.sum_of_costs([10, 100, 200, 250, n]))


In order to use this cost class in a change point detection algorithm (inheriting from :class:`BaseEstimator`), either pass a :class:`CostL1` instance (through the argument ``'custom_cost'``) or set :code:`model="l1"`.

.. code-block:: python

    c = rpt.costs.CostL1(); algo = rpt.Dynp(custom_cost=c)
    # is equivalent to
    algo = rpt.Dynp(model="l1")


Code explanation
----------------------------------------------------------------------------------------------------

.. autoclass:: ruptures.costs.CostL1
    :members:
    :special-members: __init__


.. rubric:: References

.. bibliography:: ../biblio.bib
    :style: alpha
    :cited:
    :labelprefix: C1
    :keyprefix: c1-

"""
import numpy as np
# from scipy.stats import rankdata
from scipy.stats.mstats import rankdata
from numpy.linalg import inv

from ruptures.base import BaseCost
from ruptures.costs import NotEnoughPoints


class CostRank(BaseCost):
    r"""
    Least absolute deviation.
    """

    model = "rank"

    def __init__(self):
        self.sigma = None
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

        def func(row):
            row = np.reshape((row + 0.5), (-1, 1))
            return row @ row.T

        ranks = np.argsort(signal, axis=0)
        centered_ranks = (ranks - len(signal) / 2).astype(int)

        sigma = np.apply_along_axis(
            func,
            1,
            centered_ranks
        )
        sigma = np.sum(sigma, axis=0)

        self.sigma = sigma
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
        return -(end - start) * mean.T @ inv(self.sigma) @ mean / len(self.ranks)
