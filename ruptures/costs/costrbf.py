r"""
.. _sec-kernel:

Kernelized mean change
====================================================================================================

Description
----------------------------------------------------------------------------------------------------

Given a positive semi-definite kernel :math:`k(\cdot, \cdot) : \mathbb{R}^d\times \mathbb{R}^d \mapsto \mathbb{R}` and its associated feature map :math:`\Phi:\mathbb{R}^d \mapsto \mathcal{H}` (where :math:`\mathcal{H}` is an appropriate Hilbert space), this cost function detects changes in the mean of the embedded signal :math:`\{\Phi(y_t)\}_t` :cite:`ker-arlot2012kernel,ker-gretton2012kernel`.
Formally, for a signal :math:`\{y_t\}_t` on an interval :math:`I`,

    .. math:: c(y_{I}) = \sum_{t\in I} \|\Phi(y_t) - \bar{\mu}\|_{\mathcal{H}}^2

where :math:`\bar{\mu}` is the empirical mean of the embedded sub-signal :math:`\{\Phi(y_t)\}_{t\in I}`.
Here the kernel is the radial basis function (rbf):

    .. math:: k(x, y) = \exp(-\gamma \|x-y\|^2)

where :math:`\|\cdot\|` is the Euclidean norm and :math:`\gamma>0` is the so-called bandwidth parameter and is determined according to median heuristics (i.e. equal to the inverse of median of all pairwise distances).

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

Then create a :class:`CostRbf` instance and print the cost of the sub-signal :code:`signal[50:150]`.

.. code-block:: python

    c = rpt.costs.CostRbf().fit(signal)
    print(c.error(50, 150))

You can also compute the sum of costs for a given list of change points.

.. code-block:: python

    print(c.sum_of_costs(bkps))
    print(c.sum_of_costs([10, 100, 200, 250, n]))


In order to use this cost class in a change point detection algorithm (inheriting from :class:`BaseEstimator`), either pass a :class:`CostRbf` instance (through the argument ``'custom_cost'``) or set :code:`model="rbf"`.

.. code-block:: python

    c = rpt.costs.CostRbf(); algo = rpt.Dynp(custom_cost=c)
    # is equivalent to
    algo = rpt.Dynp(model="rbf")


Code explanation
----------------------------------------------------------------------------------------------------

.. autoclass:: ruptures.costs.CostRbf
    :members:
    :special-members: __init__

.. rubric:: References

.. bibliography:: ../biblio.bib
    :style: alpha
    :cited:
    :labelprefix: KER
    :keyprefix: ker-


"""
import numpy as np
from scipy.spatial.distance import pdist, squareform

from ruptures.exceptions import NotEnoughPoints
from ruptures.base import BaseCost


class CostRbf(BaseCost):

    r"""
    Kernel cost function (rbf kernel).
    """

    model = "rbf"

    def __init__(self):
        self.gram = None
        self.min_size = 2

    def fit(self, signal):
        """Sets parameters of the instance.

        Args:
            signal (array): signal. Shape (n_samples,) or (n_samples, n_features)

        Returns:
            self
        """
        if signal.ndim == 1:
            K = pdist(signal.reshape(-1, 1), metric="sqeuclidean")
        else:
            K = pdist(signal, metric="sqeuclidean")
        K_median = np.median(K) 
        if K_median != 0:
             K/= K_median
        np.clip(K, 1e-2, 1e2, K)
        self.gram = np.exp(squareform(-K))
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
