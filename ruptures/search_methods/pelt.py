import numpy as np
import abc
from ruptures.search_methods import BaseClass
import collections


class Pelt(BaseClass, metaclass=abc.ABCMeta):
    """Contient l'algorithme de parcours des partitions."""

    def __init__(self):
        super(Pelt, self).__init__()

    def reset_params(self, penalty, jump=1, min_size=2):
        """To reset parameters of the search method. Not to be confused with the
        set_params method (which set the parameters of the error method)

        Args:
            penalty (float): penalty to use
            jump (int, optional): number of points between two potential
                changepoints. Defaults to 1
            min_size (int, optional): minimum size of a segment. Defaults to 2

        Returns:
            None:
        """
        assert min_size > 0
        if isinstance(penalty, collections.Iterable):
            assert all(p > self.K and p >= 0 for p in penalty)
        else:
            assert penalty >= 0
            assert penalty > self.K
        self.penalty = penalty
        self.jump = jump
        self.min_size = min_size
        # do not forget to reset the self.search_method if important attributes
        # have been changed.
        self.search_method.cache_clear()

    def search_method(self):
        """Search the partition space for segmentation which minimize the
        penalized cost.

        Returns:
            dict: {(start, end): segment cost + penalty}
        """
        bkps = np.arange(0, self.n, self.jump)
        bkps[-1] = self.n

        # initialisation
        self.F = dict()
        self.F[0] = {}

        self.R = dict()
        self.R[0] = [0]

        for bkp, previous_bkp in zip(bkps[1:], bkps[:-1]):

            # we get the potential last breakpoints of the segment [0:end]
            R_tmp = [t for t in self.R[previous_bkp]
                     if self.min_size <= bkp - t]
            R_tmp.extend(t for t in bkps if self.min_size <=
                         bkp - t < self.min_size + self.jump)

            if len(R_tmp) == 0:
                self.F[bkp] = self.F[previous_bkp]
                self.R[bkp] = self.R[previous_bkp]
                continue

            # calcul des quantités F(t) + error(t:bkp)
            error_list = list()
            costs_list = list()
            for t in R_tmp:

                tmp_error = self.error(t, bkp)
                tmp_cost = tmp_error + sum(self.F[t].values())
                error_list.append(tmp_error)
                costs_list.append(tmp_cost)

            min_cost_index, min_cost, min_error = min(
                zip(R_tmp, costs_list, error_list), key=lambda z: z[1])

            # on met self.F à jour
            self.F[bkp] = self.F[min_cost_index].copy()
            self.F[bkp].update(
                {(min_cost_index, bkp): min_error + self.penalty})

            # on met self.R à jour
            F_bkp = sum(self.F[bkp].values())
            self.R[bkp] = [t for t, v in zip(R_tmp, costs_list) if v +
                           self.K < F_bkp]

            if len(self.R[bkp]) == 0:
                self.R[bkp] = [0]

        return self.F[self.n]

    def fit(self, signal, penalty, jump=1, min_size=2):
        """Computes the segmentation for a given penalty using PELT (or a list
        of penalties).

        Args:
            signal (array): signal to segment. Shape (n_samples,  n_features)
                or (n_samples,).
            penalty (float or iterable): penalty or list of penalties to use.
            jump (int, optional): number of points between two potential
                breakpoints.
            min_size (int, optional): regimes shorter than min_size are
                discarded.

        Returns:
            list: list of list of indexes (end of each regime) or list of
                indexes (same depth as penalty).
        """
        self.signal = signal

        if not isinstance(penalty, collections.Iterable):
            penalty_list = [penalty]
        else:
            penalty_list = penalty

        bkps_list = list()

        for pen in penalty_list:
            self.reset_params(penalty, jump, min_size)
            self.partition = self.search_method()
            self.bkps = sorted(e for (s, e) in self.partition)
            bkps_list.append(self.bkps)

        if not isinstance(penalty, collections.Iterable):
            return self.bkps
        return bkps_list
