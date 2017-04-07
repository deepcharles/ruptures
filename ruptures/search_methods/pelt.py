"""

Penalized changepoint detection.

Implementation follows PELT which has supposedly linear complexity in the number of samples.


"""
from math import floor
import numpy as np
from scipy.spatial.distance import pdist, squareform

from ruptures.costs import Cost


class Pelt:

    """Contient l'algorithme de parcours des partitions."""

    def __init__(self, model="constantl2", min_size=2, jump=1):
        """One line description

        Detailled description

        Args:
            model (str): constantl1|constantl2|rbf
            min_size (int, optional): minimum segment length
            jump (int, optional): subsample (one every "jump" points)

        Returns:
            self
        """
        self.model = model
        self.min_size = min_size
        self.jump = jump
        self.cost = Cost(model=self.model)
        self.n_samples = None

    def seg(self, pen):
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
                except KeyError:  # no partition of 0..t exists
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
        """Compute params to segment signal.

        Args:
            signal (array): signal to segment. Shape (n_samples, n_features) or (n_samples,).

        Returns:
            self
        """
        # update some params
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
            pen (float): penalty value

        Returns:
            list: sorted list of breakpoints
        """
        partition = self.seg(pen)
        bkps = sorted(e for s, e in partition.keys())
        return bkps

    def fit_predict(self, signal, pen):
        """Helper method to call fit and predict once."""
        self.fit(signal)
        return self.predict(pen)
