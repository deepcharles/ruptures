r"""
.. _sec-binseg:

Binary segmentation
====================================================================================================

Description
----------------------------------------------------------------------------------------------------

Binary change point detection is used to perform fast signal segmentation and is implemented in
:class:`ruptures.detection.BinSeg`.
It is a sequential approach: first, one change point is detected in the complete input signal, then
series is split around this change point, then the operation is repeated on the two resulting
sub-signals. See for instance :cite:`bs-Bai1997` and :cite:`bs-fryzlewicz2014` for a theoretical and
algorithmic analysis of :class:`ruptures.detection.BinSeg`.
The benefits of binary segmentation includes low complexity (of the order of
:math:`\mathcal{O}(n\log n)`, where :math:`n` is the number of samples), the fact that it can extend
any single change point detection method to detect multiple changes points and that it can work
whether the number of regimes is known beforehand or not.

.. figure:: /images/schema_binseg.png
   :scale: 50 %
   :alt: Schematic view of the binary segmentation algorithm

   Schematic view of the binary segmentation algorithm.


Usage
----------------------------------------------------------------------------------------------------

Start with the usual imports and create a signal.

.. code-block:: python

    import numpy as np
    import matplotlib.pylab as plt
    import ruptures as rpt
    # creation of data
    n = 500  # number of samples
    n_bkps, sigma = 3, 5  # number of change points, noise standart deviation
    signal, bkps = rpt.pw_constant(n, dim, n_bkps, noise_std=sigma)

To perform a binary segmentation of a signal, initialize a :class:`ruptures.detection.BinSeg`
instance.

.. code-block:: python

    # change point detection
    model = "l2"  # "l1", "rbf", "linear", "normal", "ar"
    algo = rpt.Binseg(model=model).fit(signal)
    my_bkps = algo.predict(n_bkps=3)

    # show results
    rpt.show.display(signal, bkps, my_bkps, figsize=(10, 6))
    plt.show()

In the situation in which the number of change points is unknown, one can specify a penalty using
the ``'pen'`` parameter or a threshold on the residual norm using ``'epsilon'``.

.. code-block:: python

    my_bkps = algo.predict(pen=np.log(n)*dim*sigma**2)
    # or
    my_bkps = algo.predict(epsilon=3*n*sigma**2)

.. seealso:: :ref:`sec-general-formulation` for more information about stopping rules of sequential algorithms.

For faster predictions, one can modify the ``'jump'`` parameter during initialization.
The higher it is, the faster the prediction is achieved (at the expense of precision).

.. code-block:: python

    algo = rpt.Binseg(model=model, jump=10).fit(signal)


Code explanation
----------------------------------------------------------------------------------------------------

.. autoclass:: ruptures.detection.Binseg
    :members:
    :special-members: __init__


.. rubric:: References

.. bibliography:: ../biblio.bib
    :style: alpha
    :cited:
    :labelprefix: BS
    :keyprefix: bs-

"""
from functools import lru_cache
from ruptures.base import BaseCost, BaseEstimator
from ruptures.costs import cost_factory
from ruptures.utils import pairwise


class Binseg(BaseEstimator):

    """Binary segmentation."""

    def __init__(self, model="l2", custom_cost=None, min_size=2, jump=5, params=None):
        """Initialize a Binseg instance.

        Args:
            model (str, optional): segment model, ["l1", "l2", "rbf",...]. Not used if ``'custom_cost'`` is not None.
            custom_cost (BaseCost, optional): custom cost function. Defaults to None.
            min_size (int, optional): minimum segment length. Defaults to 2 samples.
            jump (int, optional): subsample (one every *jump* points). Defaults to 5 samples.
            params (dict, optional): a dictionary of parameters for the cost instance.


        Returns:
            self
        """

        if custom_cost is not None and isinstance(custom_cost, BaseCost):
            self.cost = custom_cost
        else:
            if params is None:
                self.cost = cost_factory(model=model)
            else:
                self.cost = cost_factory(model=model, **params)
        self.min_size = max(min_size, self.cost.min_size)
        self.jump = jump
        self.n_samples = None
        self.signal = None
        # cache for intermediate results
        self.single_bkp = lru_cache(maxsize=None)(self._single_bkp)

    def _seg(self, n_bkps=None, pen=None, epsilon=None):
        """Computes the binary segmentation.

        The stopping rule depends on the parameter passed to the function.

        Args:
            n_bkps (int): number of breakpoints to find before stopping.
            penalty (float): penalty value (>0)
            epsilon (float): reconstruction budget (>0)

        Returns:
            dict: partition dict {(start, end): cost value,...}
        """

        # initialization
        bkps = [self.n_samples]
        stop = False
        while not stop:
            stop = True
            new_bkps = [self.single_bkp(start, end)
                        for start, end in pairwise([0] + bkps)]
            bkp, gain = max(new_bkps, key=lambda x: x[1])

            if bkp is None:  # all possible configuration have been explored.
                break

            if n_bkps is not None:
                if len(bkps) - 1 < n_bkps:
                    stop = False
            elif pen is not None:
                if gain > pen:
                    stop = False
            elif epsilon is not None:
                error = self.cost.sum_of_costs(bkps)
                if error > epsilon:
                    stop = False

            if not stop:
                bkps.append(bkp)
                bkps.sort()
        partition = {(start, end): self.cost.error(start, end)
                     for start, end in pairwise([0] + bkps)}
        return partition

    def _single_bkp(self, start, end):
        """Return the optimal breakpoint of [start:end] (if it exists)."""
        segment_cost = self.cost.error(start, end)
        gain_list = list()
        for bkp in range(start, end, self.jump):
            if bkp - start > self.min_size and end - bkp > self.min_size:
                gain = segment_cost - \
                    self.cost.error(start, bkp) - self.cost.error(bkp, end)
                gain_list.append((gain, bkp))
        try:
            gain, bkp = max(gain_list)
        except ValueError:  # if empty sub_sampling
            return None, 0
        return bkp, gain

    def fit(self, signal):
        """Compute params to segment signal.

        Args:
            signal (array): signal to segment. Shape (n_samples, n_features) or (n_samples,).

        Returns:
            self
        """
        # update some params
        if signal.ndim == 1:
            self.signal = signal.reshape(-1, 1)
        else:
            self.signal = signal
        self.n_samples, _ = self.signal.shape
        self.cost.fit(signal)
        self.single_bkp.cache_clear()

        return self

    def predict(self, n_bkps=None, pen=None, epsilon=None):
        """Return the optimal breakpoints.

        Must be called after the fit method. The breakpoints are associated with the signal passed
        to fit().
        The stopping rule depends on the parameter passed to the function.

        Args:
            n_bkps (int): number of breakpoints to find before stopping.
            penalty (float): penalty value (>0)
            epsilon (float): reconstruction budget (>0)

        Returns:
            list: sorted list of breakpoints
        """
        msg = "Give a parameter."
        assert any(param is not None for param in (n_bkps, pen, epsilon)), msg

        partition = self._seg(n_bkps=n_bkps, pen=pen, epsilon=epsilon)
        bkps = sorted(e for s, e in partition.keys())
        return bkps

    def fit_predict(self, signal, n_bkps=None, pen=None, epsilon=None):
        """Fit to the signal and return the optimal breakpoints.

        Helper method to call fit and predict once

        Args:
            signal (array): signal. Shape (n_samples, n_features) or (n_samples,).
            n_bkps (int): number of breakpoints.
            penalty (float): penalty value (>0)
            epsilon (float): reconstruction budget (>0)

        Returns:
            list: sorted list of breakpoints
        """
        self.fit(signal)
        return self.predict(n_bkps=n_bkps, pen=pen, epsilon=epsilon)
