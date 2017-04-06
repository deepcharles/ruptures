"""

Penalized changepoint detection.

Implementation follows PELT which has supposedly linear complexity in the number of samples.


"""
from math import floor
import numpy as np
from scipy.spatial.distance import pdist, squareform

from ruptures.costs import constantl1, constantl2


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
        self.signal = None  # signal à segmenter

        if self.model == "rbf":
            self.gram = None

    def error(self, start, end):
        """Return the error value on segment [start, end]."""
        sig = self.signal[start:end]
        if self.model == "constantl2":
            cost = constantl2(sig)
        elif self.model == "constantl1":
            cost = constantl1(sig)
        elif self.model == "rbf":
            cost = -self.gram[start:end, start:end].sum() / (end - start)
        return cost

    def seg(self, pen):
        """Computes the segmentation for a given penalty using PELT (or a list
        of penalties).

        Args:
            signal (array): signal to segment. Shape (n_samples, n_features)
                or (n_samples,).
            penalty (float): penalty or list of penalties to use.
            jump (int, optional): number of points between two potential
                breakpoints.
            min_size (int, optional): regimes shorter than min_size are
                discarded.

        Returns:
            list: list of indexes (end of each regime).
        """
        n_samples, _ = self.signal.shape

        # initialization
        # partitions[t] contains the optimal partition of signal[0:t]
        partitions = dict()  # this dict will be recursively filled
        partitions[0] = {(0, 0): 0}
        admissible = []

        # Recursion
        ind = [k for k in range(0, n_samples, self.jump) if k >= self.min_size]
        ind += [n_samples]
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
                tmp_partition.update({(t, bkp): self.error(t, bkp) + pen})
                subproblems.append(tmp_partition)

            # finding the optimal partition
            partitions[bkp] = min(
                subproblems, key=lambda d: sum(d.values()))
            # trimming the admissible set
            admissible = [t for t, partition in
                          zip(admissible, subproblems) if
                          sum(partition.values()) <=
                          sum(partitions[bkp].values()) + pen]

        best_partition = partitions[n_samples]
        del best_partition[(0, 0)]
        return best_partition

    def fit(self, signal):
        """Compute params to segment signal.

        Args:
            signal (array): signal to segment. Shape (n_samples, n_features) or (n_samples,).

        Returns:
            self
        """
        if signal.ndim == 1:
            self.signal = signal.reshape(-1, 1)
        else:
            self.signal = signal
        # update some params
        if self.model == "rbf":
            pairwise_dists = pdist(self.signal, 'sqeuclidean')
            pairwise_dists /= np.median(pairwise_dists)  # scaling
            self.gram = squareform(np.exp(-pairwise_dists))
            np.fill_diagonal(self.gram, 1)
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
