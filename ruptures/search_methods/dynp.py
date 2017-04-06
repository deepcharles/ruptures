"""Changepoint detection with dynamic programming.

"""
from functools import lru_cache
import numpy as np
from scipy.spatial.distance import pdist, squareform

from ruptures.search_methods import sanity_check
from ruptures.costs import constantl1, constantl2


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
        self.signal = None  # signal à segmenter
        self.seg = lru_cache(maxsize=None)(
            self._seg)  # dynamic programming dans cette fonction.

        if self.model == "rbf":
            self.gram = None

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
            sig = self.signal[start:end]
            if self.model == "constantl2":
                cost = constantl2(sig)
            elif self.model == "constantl1":
                cost = constantl1(sig)
            elif self.model == "rbf":
                cost = -self.gram[start:end, start:end].sum() / (end - start)
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
        if signal.ndim == 1:
            self.signal = signal.reshape(-1, 1)
        else:
            self.signal = signal
        # clear cache
        self.seg.cache_clear()
        # update some params
        if self.model == "rbf":
            pairwise_dists = pdist(self.signal, 'sqeuclidean')
            pairwise_dists /= np.median(pairwise_dists)  # scaling
            self.gram = squareform(np.exp(-pairwise_dists))
            np.fill_diagonal(self.gram, 1)
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
        n_samples, _ = self.signal.shape
        partition = self.seg(0, n_samples, n_bkps)
        bkps = sorted(e for s, e in partition.keys())
        return bkps

    def fit_predict(self, signal, n_bkps):
        """Helper method to call fit and predict once."""
        self.fit(signal)
        return self.predict(n_bkps)
