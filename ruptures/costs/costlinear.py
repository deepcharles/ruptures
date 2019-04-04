r"""

.. _sec-linear:

Linear model change
====================================================================================================

Description
----------------------------------------------------------------------------------------------------

Let :math:`0<t_1<t_2<\dots<n` be unknown change points indexes.
Consider the following multiple linear regression model

.. math::

    y_t = z_t' \delta_j + \varepsilon_t, \quad \forall t=t_j,\dots,t_{j+1}-1

for :math:`j>1`.
Here, the observed dependant variable is :math:`y_t\in\mathbb{R}`, the covariate vector is
:math:`x_t \in\mathbb{R}^p`, the disturbance is :math:`\varepsilon_t\in\mathbb{R}`.
The vectors :math:`\delta_j\in\mathbb{R}^p` are the paramater vectors (or regression coefficients).

The least-squares estimates of the break dates is obtained by minimiming the sum of squared
residuals :cite:`cl-Bai2003`.
Formally, the associated cost function on an interval :math:`I` is

    .. math:: c(y_{I}) = \min_{\delta\in\mathbb{R}^p} \sum_{t\in I} \|y_t - \delta' z_t \|_2^2


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
    deltas, bkps = rpt.pw_constant(n, n_reg, n_bkps, noise_std=None, delta=(1, 3))
    # observed signal
    y = np.sum(X*deltas, axis=1)
    y += np.random.normal(size=y.shape)
    # display signal
    rpt.show.display(y, bkps, figsize=(10, 6))
    plt.show()

Then create a :class:`CostLinear` instance and print the cost of the sub-signal
:code:`signal[50:150]`.

.. code-block:: python

    # stack observed signal and regressors.
    # first dimension is the observed signal.
    signal = np.column_stack((y.reshape(-1, 1), X))
    c = rpt.costs.CostLinear().fit(signal)
    print(c.error(50, 150))


You can also compute the sum of costs for a given list of change points.

.. code-block:: python

    print(c.sum_of_costs(bkps))
    print(c.sum_of_costs([10, 100, 200, 250, n]))


In order to use this cost class in a change point detection algorithm (inheriting from
:class:`BaseEstimator`), either pass a :class:`CostLinear` instance (through the argument
``'custom_cost'``) or set :code:`model="linear"`.

.. code-block:: python

    c = rpt.costs.CostLinear(); algo = rpt.Dynp(custom_cost=c)
    # is equivalent to
    algo = rpt.Dynp(model="linear")


Code explanation
----------------------------------------------------------------------------------------------------

.. autoclass:: ruptures.costs.CostLinear
    :members:
    :special-members: __init__


.. rubric:: References

.. bibliography:: ../biblio.bib
    :style: alpha
    :cited:
    :labelprefix: CL
    :keyprefix: cl-

"""
from numpy.linalg import lstsq

from ruptures.base import BaseCost
from ruptures.costs import NotEnoughPoints


class CostLinear(BaseCost):

    r"""
    Least-squares estimate for linear changes.
    """

    model = "linear"

    def __init__(self):
        self.signal = None
        self.covar = None
        self.min_size = 2

    def fit(self, signal):
        """Set parameters of the instance.
        The first column contains the observed variable.
        The other columns contains the covariates.

        Args:
            signal (array): signal. Shape (n_samples, n_regressors+1)

        Returns:
            self
        """
        assert signal.ndim > 1, "Not enough dimensions"

        self.signal = signal[:, 0].reshape(-1, 1)
        self.covar = signal[:, 1:]
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
        _, residual, _, _ = lstsq(X, y, rcond=None)
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
