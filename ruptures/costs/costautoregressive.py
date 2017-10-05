r"""

.. _sec-autoregressive:

Change in a autoregressive model
====================================================================================================

Description
----------------------------------------------------------------------------------------------------


Usage
----------------------------------------------------------------------------------------------------

Start with the usual imports and create a signal with piecewise linear trends.

.. code-block:: python

    from itertools import cycle
    import numpy as np
    import matplotlib.pylab as plt
    import ruptures as rpt
    # creation of data
    n = 2000
    n_bkps, sigma = 4, 0.5  # number of change points, noise standart deviation
    bkps = [400, 1000, 1300, 1800, n]
    f1 = np.array([0.075, 0.1])
    f2 = np.array([0.1, 0.125])
    freqs = np.zeros((n, 2))
    for sub, val in zip(np.split(freqs, bkps[:-1]), cycle([f1, f2])):
        sub += val
    tt = np.arange(n)
    signal = np.sum((np.sin(2*np.pi*tt*f) for f in freqs.T))
    signal += np.random.normal(scale=sigma, size=signal.shape)
    # display signal
    rpt.show.display(signal, bkps, figsize=(10, 6))
    plt.show()

Then create a :class:`CostAR` instance and print the cost of the sub-signal
:code:`signal[50:150]`.
The autoregressive order can be specified through the keyword ``'order'``.

.. code-block:: python

    c = rpt.costs.CostAR(order=10).fit(signal)
    print(c.error(50, 150))


You can also compute the sum of costs for a given list of change points.

.. code-block:: python

    print(c.sum_of_costs(bkps))
    print(c.sum_of_costs([10, 100, 200, 250, n]))


In order to use this cost class in a change point detection algorithm (inheriting from
:class:`BaseEstimator`), either pass a :class:`CostAR` instance (through the argument
``'custom_cost'``) or set :code:`model="ar"`.
Additional parameters can be passed to the cost instance through the keyword ``'params'``.

.. code-block:: python

    c = rpt.costs.CostAR(order=10); algo = rpt.Dynp(custom_cost=c)
    # is equivalent to
    algo = rpt.Dynp(model="ar", params={"order": 10})


Code explanation
----------------------------------------------------------------------------------------------------

.. autoclass:: ruptures.costs.CostAR
    :members:
    :special-members: __init__


.. rubric:: References

.. bibliography:: ../biblio.bib
    :style: alpha
    :cited:
    :labelprefix: AR
    :keyprefix: ar-

"""
import numpy as np
from numpy.lib.stride_tricks import as_strided
from numpy.linalg import lstsq

from ruptures.base import BaseCost
from ruptures.costs import NotEnoughPoints


class CostAR(BaseCost):

    r"""
    Least-squares estimate for changes in autoregressive coefficients.
    """

    model = "ar"

    def __init__(self, order=4):
        self.signal = None
        self.covar = None
        self.min_size = max(5, order + 1)
        self.order = order

    def fit(self, signal):
        """Set parameters of the instance.
        The first column contains the observed variable.
        The other columns contains the covariates.

        Args:
            signal (array): 1d signal. Shape (n_samples, 1) or (n_samples,).

        Returns:
            self
        """
        if signal.ndim == 1:
            self.signal = signal.reshape(-1, 1)
        else:
            self.signal = signal

        # lagged covariates
        n_samples, _ = self.signal.shape
        strides = (self.signal.itemsize, self.signal.itemsize)
        shape = (n_samples - self.order, self.order)
        lagged = as_strided(self.signal, shape=shape, strides=strides)
        # pad the first columns
        lagged_after_padding = np.pad(lagged,
                                      ((self.order, 0), (0, 0)),
                                      mode="edge")
        # add intercept
        self.covar = np.c_[lagged_after_padding, np.ones(n_samples)]
        # pad signal on the edges
        self.signal[:self.order] = self.signal[self.order]
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
        y, X = self.signal[start:end], self.covar[start:end]
        _, residual, _, _ = lstsq(X, y)
        return residual.sum()
