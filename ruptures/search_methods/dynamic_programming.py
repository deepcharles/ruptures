from ruptures.search_methods import BaseClass
import abc
from ruptures.search_methods import sanity_check
import collections
# container for the parameters needed to segment.
Params = collections.namedtuple(
    "Params", ["n_regimes", "start", "end", "jump", "min_size"])


class Dynp(BaseClass, metaclass=abc.ABCMeta):

    """Optimisation using dynamic programming. Given a error function, it computes
        the best partition for which the sum of errors is minimum."""

    def __init__(self):
        super(Dynp, self).__init__()
        self._partition = None  # to hold the current best partition

    def reset_params(self, jump=1, min_size=2):
        pass

    def cache(self, params):
        """Cache for the intermediate subproblems. Results are computed once
        then cached.

        Args:
            params (namedtupe): container for the change point
                parameters.

        Returns:
            dict: {(start, end): cost value, ...}
        """
        n_regimes = params.n_regimes
        start, end = params.start, params.end
        jump, min_size = params.jump, params.min_size

        if n_regimes == 1:
            return {(start, end): self.error(start, end)}
        elif n_regimes > 1:
            # Let's fill the list of admissible last breakpoints
            multiple_of_jump = (k for k in range(start, end) if k % jump == 0)
            admissible_bkps = list()
            for bkp in multiple_of_jump:
                n_samples = bkp - start
                # first check if left subproblem is possible
                if sanity_check(n_samples, n_regimes - 1, jump, min_size):
                    # second check if the right subproblem has enough points
                    if end - bkp >= min_size:
                        admissible_bkps.append(bkp)

            assert len(
                admissible_bkps) > 0, "No admissible last breakpoints found.\
             Parameters: {}".format(params)

            # Compute the subproblems
            sub_problems = list()
            for bkp in admissible_bkps:
                left_params = Params(n_regimes - 1, start, bkp, jump, min_size)
                right_params = Params(1, bkp, end, jump, min_size)
                left_partition = self.cache(left_params)
                right_partition = self.cache(right_params)
                tmp_partition = dict(left_partition)
                tmp_partition[(bkp, end)] = right_partition[(bkp, end)]
                sub_problems.append(tmp_partition)

            # Find the optimal partition
            return min(sub_problems, key=lambda d: sum(d.values()))

    def fit(self, n_regimes, signal=None, jump=1, min_size=2):
        """Computes the segmentation of a signal. If a signal is not passed to
        the function it reuses the cache.

        Args:
            n_regimes (int or iterable): number of regimes. If iterable (of
                integers), computes the segmentation for all integers.
            signal (arrayn optional): signal to segment. Shape
                (n_samples, n_features) or (n_samples,). If None, self.signal
                is used instead.
            jump (int, optional): number of points between two potential
                breakpoints.
            min_size (int, optional): regimes shorter than min_size are
                discarded.

        Returns:
            list: indexes of each breakpoints.
        """
        if signal is None:
            if self.signal is None:
                assert False, "You must define a signal."
        else:
            self.signal = signal
        n_samples = self.signal.shape[0]

        assert sanity_check(
            n_samples,
            n_regimes,
            jump,
            min_size), "Change n_regimes, jump or min_size."

        params = Params(n_regimes=int(n_regimes),
                        start=0, end=self.signal.shape[0],
                        jump=int(jump), min_size=int(min_size))
        self._partition = self.cache(params)
        # we return the end of each segment of the partition
        bkps = sorted(e for (s, e) in self._partition)
        return bkps
