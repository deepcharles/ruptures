r"""
.. _sec-metric:

Mahalanobis-type metric
====================================================================================================

Description
----------------------------------------------------------------------------------------------------

Given a positive semi-definite matrix :math:`M\in\mathbb{R}^{d\times d}`,
this cost function detects changes in the mean of the embedded signal defined by the pseudo-metric

    .. math:: \|x-y\|_M^2 = (x-y)^t M (x-y)

Formally, for a signal :math:`\{y_t\}_t` on an interval :math:`I`, the cost function is equal to

    .. math:: c(y_{I}) = \sum_{t\in I} \|y_t - \bar{\mu}\|_{M}^2

where :math:`\bar{\mu}` is the empirical mean of the sub-signal :math:`\{y_t\}_{t\in I}`.
The matrix :math:`M` can for instance be the result of a similarity learning algorithm :cite:`ml-Xing2003` or the inverse of the empirical covariance matrix (yielding the Mahalanobis distance).

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

Then create a :class:`CostMl` instance and print the cost of the sub-signal :code:`signal[50:150]`.

.. code-block:: python

    M = np.eye(dim)
    c = rpt.costs.CostMl(metric=M).fit(signal)
    print(c.error(50, 150))

You can also compute the sum of costs for a given list of change points.

.. code-block:: python

    print(c.sum_of_costs(bkps))
    print(c.sum_of_costs([10, 100, 200, 250, n]))


In order to use this cost class in a change point detection algorithm (inheriting from :class:`BaseEstimator`), either pass a :class:`CostMl` instance (through the argument ``'custom_cost'``) or set :code:`model="mahalanobis"`.

.. code-block:: python

    c = rpt.costs.CostMl(metric=M); algo = rpt.Dynp(custom_cost=c)
    # is equivalent to
    algo = rpt.Dynp(model="mahalanobis", params={"metric": M})


Code explanation
----------------------------------------------------------------------------------------------------

.. autoclass:: ruptures.costs.CostMl
    :members:
    :special-members: __init__

.. rubric:: References

.. bibliography:: ../biblio.bib
    :style: alpha
    :cited:
    :labelprefix: ML
    :keyprefix: ml-


"""
import numpy as np
from numpy.linalg import inv

from ruptures.base import BaseCost
from ruptures.exceptions import NotEnoughPoints


class CostMl(BaseCost):

    r"""
    Mahalanobis-type cost function.
    """

    model = "mahalanobis"

    def __init__(self, metric=None):
        """Create a new instance.

        Args:
            metric (ndarray, optional): PSD matrix that defines a Mahalanobis-type pseudo distance. If None, defaults to the Mahalanobis matrix. Shape (n_features, n_features).

        Returns:
            self
        """
        self.metric = metric
        self.gram = None
        self.min_size = 2

    def fit(self, signal):
        """Sets parameters of the instance.

        Args:
            signal (array): signal. Shape (n_samples,) or (n_samples, n_features)

        Returns:
            self
        """

        s_ = signal.reshape(-1, 1) if signal.ndim == 1 else signal

        # Mahalanobis metric if self.metric is None
        if self.metric is None:
            covar = np.cov(s_.T)
            self.metric = inv(
                covar.reshape(1, 1) if covar.size == 1 else covar)

        self.gram = s_.dot(self.metric).dot(s_.T)
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
        sub_gram = self.gram[start:end, start:end]
        val = np.diagonal(sub_gram).sum()
        val -= sub_gram.sum() / (end - start)
        return val
