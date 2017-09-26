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

    import numpy as np
    import matplotlib.pylab as plt
    import ruptures as rpt
    # creation of data
    n, n_reg = 2000, 3  # number of samples, number of regressors (including intercept)
    n_bkps, sigma = 3, 5  # number of change points, noise standart deviation
    # regressors
    tt = np.linspace(0, 10*np.pi, n)
    X = np.vstack((np.sin(tt), np.sin(5*tt), np.ones(n))).T
    # parameter vectors
    deltas, bkps = rpt.pw_constant(n, n_reg, n_bkps, noisy=False, delta=(1, 3))
    # observed signal
    y = np.sum(X*deltas, axis=1)
    y += np.random.normal(size=signal.shape)
    # display signal
    rpt.show.display(y, bkps, figsize=(10, 6))
    plt.show()

Then create a :class:`CostAR` instance and print the cost of the sub-signal
:code:`signal[50:150]`.
The autoregressive order can be specified through the keyword ``'order'``.

.. code-block:: python

    # stack observed signal and regressors.
    # first dimension is the observed signal.
    signal = np.column_stack((y.reshape(-1, 1), X))
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


# class Cost(BaseCost):

#     """Compute error (in different norms) when approximating a signal with a constant value."""

#     def __init__(self, model="l2"):
#         assert model in [
#             "l1", "l2", "rbf"], "Choose different model."
#         self.model = model
#         if self.model in ["l1", "l2", "rbf"]:
#             self.min_size = 2

#         self.signal = None
#         self.gram = None

#     def fit(self, signal):
#         """Update the parameters of the instance to fit the signal.

#         Detailled description

#         Args:
# signal (array): signal of shape (n_samples, n_features) of (n_samples,)

#         Returns:
#             self:
#         """
#         if signal.ndim == 1:
#             self.signal = signal.reshape(-1, 1)
#         else:
#             self.signal = signal

#         if self.model == "rbf":
#             pairwise_dists = pdist(self.signal, 'sqeuclidean')
# pairwise_dists /= np.median(pairwise_dists)  # scaling
#             self.gram = squareform(np.exp(-pairwise_dists))
#             np.fill_diagonal(self.gram, 1)
#         elif self.model == "l2":
#             self.gram = self.signal.dot(self.signal.T)

#         return self

#     def error(self, start, end):
#         """Return squared error on the interval start:end

#         Detailled description

#         Args:
#             start (int): start index (inclusive)
#             end (int): end index (exclusive)

#         Returns:
#             float: error

#         Raises:
#             NotEnoughPoints: when not enough points
#         """
#         if end - start < self.min_size:
#             raise NotEnoughPoints
#         if self.model in ["l2", "rbf"]:
#             sub_gram = self.gram[start:end, start:end]
#             cost = np.diagonal(sub_gram).sum()
#             cost -= sub_gram.sum() / (end - start)
#         elif self.model == "l1":
#             med = np.median(self.signal[start:end], axis=0)
#             cost = abs(self.signal[start:end] - med).sum()
#         return cost
