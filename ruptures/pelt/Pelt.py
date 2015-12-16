from ruptures.pelt.costs import NotEnoughPoints


class Pelt(object):
    """Contient l'algorithme de parcours des partitions."""

    def __init__(self, error_func, penalty, n, K=0):
        self.error_func = error_func
        assert isinstance(n, int)
        assert n > 2  # at least three points
        self.n = n
        assert penalty >= 0
        self.penalty = penalty
        self.K = K
        self.R = {0: [-1]}  # will contain potential changepoints
        self.cp = {-1: list()}  # will contain the changepoint indexes.
        self.F = {-1: - self.penalty}
        self.chg = list()

    def fit(self):

        # we reset some attributes
        self.R = {0: [-1]}  # will contain potential changepoints
        self.cp = {-1: list()}  # will contain the changepoint indexes.
        self.F = {-1: - self.penalty}

        # the actual changepoint detection algorithm:
        for tau in range(self.n):
            tmp = list()
            for t in self.R[tau]:
                try:
                    c = self.error_func(t + 1, tau)
                    tmp.append((t, self.F[t] + c + self.penalty))
                except NotEnoughPoints:
                    pass

            # tmp = [(t, self.F[t] + self.error_func(t + 1, tau) + self.penalty)
            #        for t in self.R[tau]]
            t, f = min(tmp, key=lambda x: x[1])

            assert tau not in self.F
            self.F[tau] = f
            self.cp[tau] = self.cp[t] + [t]
            self.R[tau + 1] = [tt for (tt, ff) in tmp if ff -
                               self.penalty + self.K <= self.F[tau]] + [tau]

        self.chg = [temps for temps in self.cp[
            self.n - 1] if 0 <= temps < self.n]

        return self.chg
