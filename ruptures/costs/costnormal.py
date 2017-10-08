r"""
.. _sec-normal:

Gaussian process change
====================================================================================================

Description
----------------------------------------------------------------------------------------------------

This cost function detects changes in the mean and scale of a Gaussian time series.
Formally, for a signal :math:`\{y_t\}_t` on an interval :math:`I`,

    .. math:: c(y_{I}) = |I| \log\det\widehat{\Sigma}_I

where :math:`\widehat{\Sigma}_I` is the empirical covariance matrix of the sub-signal :math:`\{y_t\}_{t\in I}`.


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

Then create a :class:`CostNormal` instance and print the cost of the sub-signal :code:`signal[50:150]`.

.. code-block:: python

    c = rpt.costs.CostNormal().fit(signal)
    print(c.error(50, 150))

You can also compute the sum of costs for a given list of change points.

.. code-block:: python

    print(c.sum_of_costs(bkps))
    print(c.sum_of_costs([10, 100, 200, 250, n]))


In order to use this cost class in a change point detection algorithm (inheriting from :class:`BaseEstimator`), either pass a :class:`CostNormal` instance (through the argument ``'custom_cost'``) or set :code:`model="normal"`.

.. code-block:: python

    c = rpt.costs.CostNormal(); algo = rpt.Dynp(custom_cost=c)
    # is equivalent to
    algo = rpt.Dynp(model="normal")


Code explanation
----------------------------------------------------------------------------------------------------

.. autoclass:: ruptures.costs.CostNormal
    :members:
    :special-members: __init__

"""
import numpy as np
from numpy.linalg import slogdet

from ruptures.base import BaseCost
from ruptures.costs import NotEnoughPoints


class CostNormal(BaseCost):

    """Maximum Gaussian likelihood."""

    model = "normal"

    def __init__(self):
        self.signal = None
        self.min_size = 2

    def fit(self, signal):
        """Set parameters of the instance.

        Args:
            signal (array): signal. Shape (n_samples,) or (n_samples, n_features)

        Returns:
            self
        """
        if signal.ndim == 1:
            self.signal = signal.reshape(-1, 1)
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

        if self.signal.shape[1] > 1:
            cov = np.cov(sub.T)
        else:
            cov = np.array([[sub.var()]])
        _, val = slogdet(cov)
        return val * (end - start)
