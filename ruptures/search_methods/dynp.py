"""Changepoint detection with dynamic programming.

"""
from functools import lru_cache

from ruptures.search_methods import sanity_check
from ruptures.costs import Cost


class Dynp:

    """ Find exact changepoints using dynamic programming.

    Given a error function, it computes the best partition for which the sum of errors is minimum.

    """

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
        self.seg = lru_cache(maxsize=None)(
            self._seg)  # dynamic programming dans cette fonction.
        self.cost = Cost(model=self.model)
        self.n_samples = None

    def _seg(self, start, end, n_bkps):
        """Reccurence to find best partition of signale[start:end].

        This method is to be memoize and then used.

        Args:
            start (int): start of the segment (inclusive)
            end (int): end of the segment (exclusive)
            n_bkps (int): number of breakpoints

        Returns:
            dict: {(start, end): cost value, ...}
        """
        jump, min_size = self.jump, self.min_size

        if n_bkps == 0:
            cost = self.cost.error(start, end)
            return {(start, end): cost}
        elif n_bkps > 0:
            # Let's fill the list of admissible last breakpoints
            multiple_of_jump = (k for k in range(start, end) if k % jump == 0)
            admissible_bkps = list()
            for bkp in multiple_of_jump:
                n_samples = bkp - start
                # first check if left subproblem is possible
                if sanity_check(n_samples, n_bkps, jump, min_size):
                    # second check if the right subproblem has enough points
                    if end - bkp >= min_size:
                        admissible_bkps.append(bkp)

            assert len(
                admissible_bkps) > 0, "No admissible last breakpoints found.\
             start, end: ({},{}), n_bkps: {}.".format(start, end, n_bkps)

            # Compute the subproblems
            sub_problems = list()
            for bkp in admissible_bkps:
                left_partition = self.seg(start, bkp, n_bkps - 1)
                right_partition = self.seg(bkp, end, 0)
                tmp_partition = dict(left_partition)
                tmp_partition[(bkp, end)] = right_partition[(bkp, end)]
                sub_problems.append(tmp_partition)

            # Find the optimal partition
            return min(sub_problems, key=lambda d: sum(d.values()))

    def fit(self, signal):
        """Create the cache associated with the signal.

        Dynamic programming is a recurrence. Therefore intermediate results are cached to speed up
        computations. This method sets up the cache.


        Args:
            signal (array): signal to segment. Shape (n_samples, n_features) or (n_samples,).

        Returns:
            self
        """
        # clear cache
        self.seg.cache_clear()
        # update some params
        self.cost.fit(signal)
        if signal.ndim == 1:
            n_samples, = signal.shape
        else:
            n_samples, _ = signal.shape
        self.n_samples = n_samples

        return self

    def predict(self, n_bkps):
        """Return the optimal breakpoints.

        Must be called after the fit method. The breakpoints are associated with the signal passed
        to fit().

        Args:
            n_bkps (int): number of breakpoints to look for.

        Returns:
            list: sorted list of breakpoints
        """
        partition = self.seg(0, self.n_samples, n_bkps)
        bkps = sorted(e for s, e in partition.keys())
        return bkps

    def fit_predict(self, signal, n_bkps):
        """Helper method to call fit and predict once."""
        self.fit(signal)
        return self.predict(n_bkps)
