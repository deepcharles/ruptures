r"""
.. _sec-costtsquared:

Hotelling's T2 statistic based cost function
====================================================================================================

Description
----------------------------------------------------------------------------------------------------

This cost function detects drift of the center of the signal :cite:`tsquared-Banko2011`.
Formally, for a signal :math:`\{y_t\}_t` on an interval :math:`I`, the cost function is equal to

    .. math:: c(y_{I}) = \sum_{t\in I} \|y_t - \bar{y_t}\|_2

where :math:`\bar{y_t}` is the deviation from the center of the data on lower dimensional representation of :math:`\{y_t\}_{t\in I}`,

    .. math:: \bar{y_t} = {U_p}^T y_t {y_t}^T {U_p}

where :math:`U_p` is singular vectors belong to the most important :math:`p` singular values of :math:`\{y_t\}_{t\in I}`.

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

Then create a :class:`CostTSquared` instance and print the cost of the sub-signal :code:`signal[50:150]`.

.. code-block:: python

    c = rpt.costs.CostTSquared(n_components=2).fit(signal)
    print(c.error(50, 150))


You can also compute the sum of costs for a given list of change points.

.. code-block:: python

    print(c.sum_of_costs(bkps))
    print(c.sum_of_costs([10, 100, 200, 250, n]))


In order to use this cost class in a change point detection algorithm (inheriting from :class:`BaseEstimator`), either pass a :class:`CostTSquared` instance (through the argument ``'custom_cost'``) or set :code:`model="rank"`.

.. code-block:: python

    c = rpt.costs.CostTSquared(); algo = rpt.Dynp(custom_cost=c)
    # is equivalent to
    algo = rpt.Dynp(model="t2")


Code explanation
----------------------------------------------------------------------------------------------------

.. autoclass:: ruptures.costs.CostTSquared
    :members:
    :special-members: __init__

.. rubric:: References

.. bibliography:: ../biblio.bib
    :style: alpha
    :cited:
    :labelprefix: RA
    :keyprefix: tsquared-
"""


import numpy as np
from sklearn.decomposition import PCA

from ruptures.base import BaseCost
from ruptures.costs import NotEnoughPoints


class CostTSquared(BaseCost):

    """Hotelling's T-Squared."""

    model = "t2"

    def __init__(self, n_components=2):
        """Create a new instance.

        Args:
            n_components (int, optional):  number of components to keep in signal in each segment.

        Returns:
            self
        """
        self.min_size = 2
        self.signal = None
        self.n_components = n_components

    def fit(self, signal):
        """Set parameters of the instance.

        Args:
            signal (array): signal. Shape (n_samples,) or (n_samples, n_features)

        Returns:
            self
        """
        if signal.ndim == 1 or (signal.ndim == 2 and signal.shape[1] == 1):
            raise ValueError("The signal must be multivariate.")
        else:
            self.signal = signal

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
        sub = self.signal[start:end]

        pca = PCA(self.n_components)
        sub_tr = pca.fit_transform(sub)

        return np.linalg.norm(sub_tr)**2
