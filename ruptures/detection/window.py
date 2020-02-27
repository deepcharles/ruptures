r"""
.. _sec-window:

Window-based change point detection
====================================================================================================

Description
----------------------------------------------------------------------------------------------------

Window-based change point detection is used to perform fast signal segmentation and is implemented in
:class:`ruptures.detection.Window`.
The algorithm uses two windows which slide along the data stream.
The statistical properties of the signals within each window are compared with a discrepancy
measure.
For a given cost function :math:`c(\cdot)` (see :ref:`sec-costs`), a discrepancy measure is derived
:math:`d(\cdot,\cdot)` as follows:

.. math:: d(y_{u..v}, y_{v..w}) = c(y_{u..w}) - c(y_{u..v}) - c(y_{v..w})

where :math:`\{y_t\}_t` is the input signal and :math:`u<v<w` are indexes.
The discrepancy is the cost gain of splitting the sub-signal :math:`y_{u..w}` at the index
:math:`v`.
If the sliding windows :math:`u..v` and :math:`v..w` both fall into a segment, their statistical
properties are similar and the discrepancy between the first window and the second window is low.
If the sliding windows fall into two dissimilar segments, the discrepancy is significantly
higher, suggesting that the boundary between windows is a change point.
The discrepancy curve is the curve, defined for all indexes :math:`t` between :math:`w/2` and :math:`n-w/2`
(:math:`n` is the number of samples),

.. math:: \big(t, d(y_{t-w/2..t}, y_{t..t+w/2})\big)

where :math:`w` is the window length.
A sequential peak search is performed on the discrepancy curve in order to detect change points.

The benefits of window-based segmentation includes low complexity (of the order of
:math:`\mathcal{O}(n w)`, where :math:`n` is the number of samples), the fact that it can extend
any single change point detection method to detect multiple changes points and that it can work
whether the number of regimes is known beforehand or not.

.. figure:: /images/schema_fenetre.png
   :scale: 50 %
   :alt: Schematic view of the window sliding algorithm

   Schematic view of the window sliding algorithm.

.. seealso:: :ref:`sec-binseg`, :ref:`sec-bottup`.


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

To perform a binary segmentation of a signal, initialize a :class:`ruptures.detection.Window`
instance.

.. code-block:: python

    # change point detection
    model = "l2"  # "l1", "rbf", "linear", "normal", "ar"
    algo = rpt.Window(width=40, model=model).fit(signal)
    my_bkps = algo.predict(n_bkps=3)

    # show results
    rpt.show.display(signal, bkps, my_bkps, figsize=(10, 6))
    plt.show()

The window length (in number of samples) is modified through the argument ``'width'``.
Usual methods assume that the window length is smaller than the smallest regime length.

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

    algo = rpt.Window(model=model, jump=10).fit(signal)


Code explanation
----------------------------------------------------------------------------------------------------

.. autoclass:: ruptures.detection.Window
    :members:
    :special-members: __init__

"""


import numpy as np
from scipy.signal import argrelmax

from ruptures.base import BaseCost, BaseEstimator
from ruptures.costs import cost_factory
from ruptures.utils import unzip


class Window(BaseEstimator):

    """Window sliding method."""

    def __init__(self, width=100, model="l2", custom_cost=None, min_size=2, jump=5, params=None):
        """Instanciate with window length.

        Args:
            width (int, optional): window length. Defaults to 100 samples.
            model (str, optional): segment model, ["l1", "l2", "rbf"]. Not used if
            ``'custom_cost'`` is not None.
            custom_cost (BaseCost, optional): custom cost function. Defaults to None.
            min_size (int, optional): minimum segment length.
            jump (int, optional): subsample (one every *jump* points).
            params (dict, optional): a dictionary of parameters for the cost instance.

        Returns:
            self
        """
        self.min_size = min_size
        self.jump = jump
        self.width = 2 * (width // 2)
        self.n_samples = None
        self.signal = None
        self.inds = None
        if custom_cost is not None and isinstance(custom_cost, BaseCost):
            self.cost = custom_cost
        else:
            if params is None:
                self.cost = cost_factory(model=model)
            else:
                self.cost = cost_factory(model=model, **params)
        self.score = list()

    def _seg(self, n_bkps=None, pen=None, epsilon=None):
        """Sequential peak search.

        The stopping rule depends on the parameter passed to the function.

        Args:
            n_bkps (int): number of breakpoints to find before stopping.
            penalty (float): penalty value (>0)
            epsilon (float): reconstruction budget (>0)

        Returns:
            list: breakpoint index list
        """

        # initialization
        bkps = [self.n_samples]
        stop = False
        error = self.cost.sum_of_costs(bkps)
        # peak search
        # forcing order to be above one in case jump is too large (issue #16)
        order = max(max(self.width, 2*self.min_size) // (2 * self.jump), 1)
        peak_inds_shifted, = argrelmax(self.score,
                                       order=order,
                                       mode="wrap")

        if peak_inds_shifted.size == 0:  # no peaks if the score is constant
            return bkps
        gains = np.take(self.score, peak_inds_shifted)
        peak_inds_arr = np.take(self.inds, peak_inds_shifted)
        # sort according to score value
        _, peak_inds = unzip(sorted(zip(gains, peak_inds_arr)))
        peak_inds = list(peak_inds)

        while not stop:
            stop = True
            # _, bkp = max((v, k) for k, v in enumerate(self.score, start=1)
            # if not any(abs(k - b) < self.width // 2 for b in bkps[:-1]))

            try:
                # index with maximum score
                bkp = peak_inds.pop()
            except IndexError:  # peak_inds is empty
                break

            if n_bkps is not None:
                if len(bkps) - 1 < n_bkps:
                    stop = False
            elif pen is not None:
                gain = error - self.cost.sum_of_costs(sorted([bkp] + bkps))
                if gain > pen:
                    stop = False
            elif epsilon is not None:
                if error > epsilon:
                    stop = False

            if not stop:
                bkps.append(bkp)
                bkps.sort()
                error = self.cost.sum_of_costs(bkps)

        return bkps

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
        # indexes
        self.inds = np.arange(self.n_samples,
                              step=self.jump)
        # delete borders
        keep = (self.inds >= self.width // 2) & (
            self.inds < self.n_samples - self.width // 2)
        self.inds = self.inds[keep]
        self.cost.fit(signal)
        # compute score
        score = list()
        for k in self.inds:
            start, end = k - self.width // 2, k + self.width // 2
            gain = self.cost.error(start, end)
            gain -= self.cost.error(start, k) + self.cost.error(k, end)
            score.append(gain)
        self.score = np.array(score)
        return self

    def predict(self, n_bkps=None, pen=None, epsilon=None):
        """Return the optimal breakpoints.

        Must be called after the fit method. The breakpoints are associated with the signal passed
        to fit().
        The stopping rule depends on the parameter passed to the function.

        Args:
            n_bkps (int): number of breakpoints to find before stopping.
            penalty (float): penalty value (>0)
            penalty (float): penalty value

        Returns:
            list: sorted list of breakpoints
        """
        msg = "Give a parameter."
        assert any(param is not None for param in (n_bkps, pen, epsilon)), msg

        bkps = self._seg(n_bkps=n_bkps, pen=pen, epsilon=epsilon)
        return bkps

    def fit_predict(self, signal, n_bkps=None, pen=None, epsilon=None):
        """Helper method to call fit and predict once."""
        self.fit(signal)
        return self.predict(n_bkps=n_bkps, pen=pen, epsilon=epsilon)
