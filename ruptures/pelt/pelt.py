from ruptures.utils.memoizedict import MemoizeDict


class Pelt(object):
    """Contient l'algorithme de parcours des partitions."""

    def __init__(self, error_func, penalty, n, K=0, min_size=2, jump=1):
        self.error_func = MemoizeDict(error_func)
        assert isinstance(n, int)
        assert n > 2  # at least three points
        self.n = n
        assert penalty >= 0
        self.penalty = penalty
        self.K = K
        assert min_size > 0
        self.min_size = min_size
        self.jump = jump
        self.R = {min_size: [0]}  # will contain potential changepoints
        self.cp = {0: list()}  # will contain the changepoint indexes.
        self.F = {0: - self.penalty}
        self.chg = list()

    def fit(self):
        # the actual changepoint detection algorithm.

        start = self.min_size
        end = self.n

        # we reset some attributes
        self.R = {start: [0]}  # will contain potential changepoints
        self.cp = {0: list()}  # will contain the changepoint indexes.
        self.F = {0: - self.penalty}

        for fin in range(start, end + 1):

            # epoch 1
            tmp = list()
            for dernier_debut in self.R[fin]:
                c = self.error_func(dernier_debut, fin)
                tmp.append(
                    (dernier_debut,
                     self.F[dernier_debut] + c + self.penalty))

            t_1, f = min(tmp, key=lambda x: x[1])

            # epoch 2
            assert fin not in self.F
            self.F[fin] = f

            # epoch 3
            self.cp[fin] = self.cp[t_1] + [t_1]

            # epoch 4
            R_tmp = list()
            for dernier_debut in self.R[fin] + [fin - self.min_size]:
                if fin - dernier_debut >= self.min_size:
                    if dernier_debut in self.F:
                        if self.F[dernier_debut] + self.error_func(
                                dernier_debut, fin) + self.K <= self.F[fin]:
                            R_tmp.append(dernier_debut)

            self.R[fin + 1] = R_tmp
        self.chg = [temps for temps in self.cp[
            self.n - 1] if 0 <= temps < self.n]

        return self.cp[end]
