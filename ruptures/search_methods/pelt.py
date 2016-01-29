from ruptures.base import BaseClass
import abc
from ruptures.search_methods import MemoizeDict


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
        assert penalty >= 0
        assert min_size > 0
        self.penalty = penalty
        self.jump = jump
        self.min_size = min_size
        # do not forget to reset the self.search_method if important attributes
        # have been changed.
        self.search_method = MemoizeDict(self.search_method.func)

    def search_method(self):

        first_end = self.min_size
        last_end = self.n

        # we reset some attributes
        self.R = {first_end: [0]}  # will contain potential changepoints
        self.cp = {0: list()}  # will contain the changepoint indexes.
        self.F = {0: - self.penalty}

        for end in range(first_end, last_end + 1):

            # epoch 1
            # costs for the different partitions of [0:end]
            segment_costs = list()
            for last_start in self.R[end]:
                c = self.error(last_start, end)
                segment_costs.append(
                    (last_start,
                     self.F[last_start] + c + self.penalty))

            t_1, f = min(segment_costs, key=lambda x: x[1])

            # epoch 2
            assert end not in self.F
            self.F[end] = f

            # epoch 3
            self.cp[end] = self.cp[t_1] + [t_1]

            # epoch 4
            R_tmp = list()
            for last_start in self.R[end] + [end - self.min_size]:
                if end - last_start >= self.min_size:
                    if last_start in self.F:
                        if self.F[last_start] + self.error(
                                last_start, end) + self.K <= self.F[end]:
                            R_tmp.append(last_start)

            self.R[end + 1] = R_tmp
        self.chg = [temps for temps in self.cp[
            self.n - 1] if 0 <= temps < self.n]

        return self.cp[last_end]

    def fit(self, signal, penalty, jump, min_size):
        self.reset_params(penalty, jump, min_size)
        self.signal = signal
        self.partition = self.search_method()
        self.bkps = sorted(t for t in self.partition)[1:] + [self.n]

        return self.bkps
