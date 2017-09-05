"""
For a given model, the change point detection problem amounts to
minimize the penalized approximation error over all the potential breakpoint repartitions
The penalty is proportional to the number of change points.
The higher the penalty value, the less change points are predicted.

Formally,

.. math:: \widehat{\mathbf{p}}_{\\beta} = \\arg \min_{\mathbf{p}} \sum_{i=1}^{|\mathbf{p}|} c(y_{p_i})\quad + \\beta |\mathbf{p}|.

The method is implemented in :class:`ruptures.detection.Pelt`.

Available cost functions:
----------------------------------------------------------------------------------------------------

For the list of available costs functions, see :ref:`sec-costs`.


Examples
----------------------------------------------------------------------------------------------------
.. code-block:: python

    import numpy as np
    import matplotlib.pylab as plt
    import ruptures as rpt

    # creation of data
    n, dim = 500, 3
    n_bkps, sigma = 3, 1
    signal, b = rpt.pw_constant(n, dim, n_bkps, noisy=True, sigma=sigma)

    # change point detection
    model = "constantl1"  # "constantl2", "rbf"
    algo = rpt.Pelt(model="constantl1", min_size=3, jump=5).fit(signal)
    my_bkps = algo.predict(pen=3)

    # show results
    fig, (ax,) = rpt.display(signal, bkps, my_bkps, figsize=(10, 6))
    plt.show()

Code explanation
====================================================================================================

.. autoclass:: ruptures.detection.Pelt
    :members:
    :special-members: __init__

"""
from math import floor

from ruptures.costs import Cost


class Pelt:

    """Contient l'algorithme de parcours des partitions."""

    def __init__(self, model="constantl2", min_size=2, jump=1):
        """One line description

        Detailled description

        Args:
            model (str): constantl1|constantl2|rbf
            min_size (int, optional): minimum segment length
            jump (int, optional): subsample (one every *jump* points)

        Returns:
            self
        """
        self.model = model
        self.min_size = min_size
        self.jump = jump
        self.cost = Cost(model=self.model)
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
