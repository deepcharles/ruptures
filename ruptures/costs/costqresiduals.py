r"""

.. _sec-costqresiduals:

Q reconstruction error based cost function
====================================================================================================

Description
----------------------------------------------------------------------------------------------------

This cost function detects change of the correlation between the variables with ignoring noisy demension :cite:`qresiduals-Banko2011`.
Formally, for a signal :math:`\{y_t\}_t` on an interval :math:`I`, the cost function is equal to

    .. math:: c(y_{I}) = \sum_{t\in I} \|y_t - \bar{y_t}\|_2

where :math:`\bar{y_t}` is the PCA based reconstructed value of :math:`\{y_t\}_{t\in I}` computed by

    .. math:: \bar{y_t} = {U_p}^T U_p y_t

where :math:`U_p` is singular vectors belong to the most important :math:`p` singular values of :math:`y_t`.

Usage
----------------------------------------------------------------------------------------------------

Start with the usual imports and create a signal.

.. code-block:: python

    import numpy as np
    import matplotlib.pylab as plt
    import ruptures as rpt
    # creation of data
    n = 200  # number of samples
    n_bkps = 3  # number of change points
    n_noisy_features = 3  # number of noisy features
    signal, bkps = rpt.pw_normal(n, n_bkps, n_noisy_features)

Then create a :class:`CostQResiduals` instance and print the cost of the sub-signal :code:`signal[50:150]`.

.. code-block:: python

    c = rpt.costs.CostQResiduals(n_components=1).fit(signal)
    print(c.error(50, 150))

You can also compute the sum of costs for a given list of change points.

.. code-block:: python

    print(c.sum_of_costs(bkps))
    print(c.sum_of_costs([10, 100, 200, 250, n]))


In order to use this cost class in a change point detection algorithm (inheriting from :class:`BaseEstimator`), either pass a :class:`CostNormal` instance (through the argument ``'custom_cost'``) or set :code:`model="qresid"`.

.. code-block:: python

    c = rpt.costs.CostQResiduals(); algo = rpt.Dynp(custom_cost=c)
    # is equivalent to
    algo = rpt.Dynp(model="qresid")


Code explanation
----------------------------------------------------------------------------------------------------

.. autoclass:: ruptures.costs.CostQResiduals
    :members:
    :special-members: __init__

.. rubric:: References

.. bibliography:: ../biblio.bib
    :style: alpha
    :cited:
    :labelprefix: RA
    :keyprefix: qresiduals-
"""


import numpy as np
from sklearn.decomposition import PCA

from ruptures.base import BaseCost
from ruptures.costs import NotEnoughPoints


class CostQResiduals(BaseCost):

    """Q residuals."""

    model = "qresid"

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

        pca = PCA(self.n_components).fit(sub)
        sub_reconstruct = pca.inverse_transform(pca.transform(sub))

        return np.sum((sub - sub_reconstruct)**2)
