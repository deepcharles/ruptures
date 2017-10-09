r"""
Exact segmentation: Pelt
====================================================================================================

Description
----------------------------------------------------------------------------------------------------

The method is implemented in :class:`ruptures.detection.Pelt`.

Because the enumeration of all possible partitions impossible, the algorithm relies on a pruning
rule. Many indexes are discarded, greatly reducing the computational cost while retaining the
ability to find the optimal segmentation.
The implementation follows :cite:`b-Killick2012a`. In addition, under certain conditions on the change
point repartition, the computational complexity is linear on average.

When calling :meth:`ruptures.detection.Pelt.__init__`, the minimum distance between change points
can be set through the keyword ``'min_size'``; through the parameter ``'jump'``, only change
point indexes multiple of a particular value are considered.


Usage
----------------------------------------------------------------------------------------------------

.. code-block:: python

    import numpy as np
    import matplotlib.pylab as plt
    import ruptures as rpt

    # creation of data
    n, dim = 500, 3
    n_bkps, sigma = 3, 1
    signal, b = rpt.pw_constant(n, dim, n_bkps, noise_std=sigma)

    # change point detection
    model = "l1"  # "l2", "rbf"
    algo = rpt.Pelt(model=model, min_size=3, jump=5).fit(signal)
    my_bkps = algo.predict(pen=3)

    # show results
    fig, (ax,) = rpt.display(signal, bkps, my_bkps, figsize=(10, 6))
    plt.show()


Code explanation
----------------------------------------------------------------------------------------------------

.. autoclass:: ruptures.detection.Pelt
    :members:
    :special-members: __init__

.. rubric:: References

.. bibliography:: ../biblio.bib
    :style: alpha
    :cited:
    :labelprefix: B
    :keyprefix: b-
"""
from math import floor

from ruptures.costs import cost_factory
from ruptures.base import BaseCost, BaseEstimator


class Pelt(BaseEstimator):

    """Penalized change point detection.

    For a given model and penalty level, computes the segmentation which minimizes the constrained
    sum of approximation errors.

    """

    def __init__(self, model="l2", custom_cost=None, min_size=2, jump=5, params=None):
        """Initialize a Pelt instance.

        Args:
            model (str, optional): segment model, ["l1", "l2", "rbf"]. Not used if ``'custom_cost'`` is not None.
            custom_cost (BaseCost, optional): custom cost function. Defaults to None.
            min_size (int, optional): minimum segment length.
            jump (int, optional): subsample (one every *jump* points).
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

    def _seg(self, pen):
        """Computes the segmentation for a given penalty using PELT (or a list
        of penalties).

        Args:
            penalty (float): penalty value

        Returns:
            dict: partition dict {(start, end): cost value,...}
        """

        # initialization
        # partitions[t] contains the optimal partition of signal[0:t]
        partitions = dict()  # this dict will be recursively filled
        partitions[0] = {(0, 0): 0}
        admissible = []

        # Recursion
        ind = [
            k for k in range(0, self.n_samples, self.jump) if k >= self.min_size]
        ind += [self.n_samples]
        for bkp in ind:
            # adding a point to the admissible set from the previous loop.
            new_adm_pt = floor((bkp - self.min_size) / self.jump)
            new_adm_pt *= self.jump
            admissible.append(new_adm_pt)

            subproblems = list()
            for t in admissible:
                # left partition
                try:
                    tmp_partition = partitions[t].copy()
                except KeyError:  # no partition of 0:t exists
                    continue
                # we update with the right partition
                tmp_partition.update({(t, bkp): self.cost.error(t, bkp) + pen})
                subproblems.append(tmp_partition)

            # finding the optimal partition
            partitions[bkp] = min(
                subproblems, key=lambda d: sum(d.values()))
            # trimming the admissible set
            admissible = [t for t, partition in
                          zip(admissible, subproblems) if
                          sum(partition.values()) <=
                          sum(partitions[bkp].values()) + pen]

        best_partition = partitions[self.n_samples]
        del best_partition[(0, 0)]
        return best_partition

    def fit(self, signal):
        """Set params.

        Args:
            signal (array): signal to segment. Shape (n_samples, n_features) or (n_samples,).

        Returns:
            self
        """
        # update params
        self.cost.fit(signal)
        if signal.ndim == 1:
            n_samples, = signal.shape
        else:
            n_samples, _ = signal.shape
        self.n_samples = n_samples
        return self

    def predict(self, pen):
        """Return the optimal breakpoints.

        Must be called after the fit method. The breakpoints are associated with the signal passed
        to fit().

        Args:
            pen (float): penalty value (>0)

        Returns:
            list: sorted list of breakpoints
        """
        partition = self._seg(pen)
        bkps = sorted(e for s, e in partition.keys())
        return bkps

    def fit_predict(self, signal, pen):
        """Fit to the signal and return the optimal breakpoints.

        Helper method to call fit and predict once

        Args:
            signal (array): signal. Shape (n_samples, n_features) or (n_samples,).
            pen (float): penalty value (>0)

        Returns:
            list: sorted list of breakpoints
        """
        self.fit(signal)
        return self.predict(pen)
