import abc
from ruptures.search_methods import BaseClass
import collections
from math import floor

# container for the parameters of the cache method.
Params = collections.namedtuple("Params", ["start", "end"])


class Pelt(BaseClass, metaclass=abc.ABCMeta):

    """Contient l'algorithme de parcours des partitions."""

    def __init__(self):
        super(Pelt, self).__init__()
        self._partition = None  # to hold the current best partition

    def cache(self, params):
        start, end = params.start, params.end
        return self.error(start, end)

    def fit(self, penalty, signal=None, jump=1, min_size=2):
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
        if signal is None:
            if self.signal is None:
                assert False, "You must define a signal."
        else:
            self.signal = signal

        n_samples = self.signal.shape[0]

        # initialization
        # partitions[t] contains the optimal partition of signal[0:t]
        partitions = dict()  # this dict will be recursively filled
        partitions[0] = {(0, 0): 0}
        admissible = []

        # Recursion
        ind = [k for k in range(0, n_samples, jump) if k >= min_size]
        ind += [n_samples]
        for bkp in ind:
            # adding a point to the admissible set from the previous loop.
            new_adm_pt = floor((bkp - min_size) / jump)
            new_adm_pt *= jump
            admissible.append(new_adm_pt)

            subproblems = list()
            for t in admissible:
                p = Params(start=t, end=bkp)
                # left partition
                try:
                    tmp_partition = partitions[t].copy()
                except KeyError:  # no partition of 0..t exists
                    continue
                # we update with the right partition
                tmp_partition.update({(t, bkp): self.cache(p) + penalty})
                subproblems.append(tmp_partition)

            # finding the optimal partition
            partitions[bkp] = min(
                subproblems, key=lambda d: sum(d.values()))
            # trimming the admissible set
            admissible = [t for t, partition in
                          zip(admissible, subproblems) if
                          sum(partition.values()) <=
                          sum(partitions[bkp].values()) + penalty]

        best_partition = partitions[n_samples]
        del best_partition[(0, 0)]
        self._partition = best_partition
        return sorted(e for (s, e) in self._partition)
