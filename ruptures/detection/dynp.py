r"""
Exact segmentation: dynamic programming
====================================================================================================

Description
----------------------------------------------------------------------------------------------------

The method is implemented in :class:`ruptures.detection.Dynp`.

Roughly speaking, it computes the cost of all subsequences of a given signal.
The number of computed costs is of the order :math:`\mathcal{O}(Kn^2)`, where :math:`K` is the number
of change points and :math:`n` the number of samples.
This has to be multiplied by the computational cost of computing the approximation error on one
sub-sequence.
Consequently, piecewise constant models are significantly faster than linear or autoregressive
models.

Computational cost is drastically reduced when considering only a subsample of possible change
points.
When calling :meth:`ruptures.detection.Dynp.__init__`, the minimum distance between change points
can be set through the keyword ``'min_size'``; through the parameter ``'jump'``, only change
point indexes multiple of a particular value are considered.


Usage
----------------------------------------------------------------------------------------------------

.. code-block:: python

    import numpy as np
    import matplotlib.pylab as plt
    import ruptures as rpt

    # creation of data
    n, dim = 500, 3
    n_bkps, sigma = 3, 5
    signal, bkps = rpt.pw_constant(n, dim, n_bkps, noise_std=sigma)

    # change point detection
    model = "l1"  # "l2", "rbf"
    algo = rpt.Dynp(model=model, min_size=3, jump=5).fit(signal)
    my_bkps = algo.predict(n_bkps=3)

    # show results
    rpt.show.display(signal, bkps, my_bkps, figsize=(10, 6))
    plt.show()


Code explanation
----------------------------------------------------------------------------------------------------
.. autoclass:: ruptures.detection.Dynp
    :members:
    :special-members: __init__

"""
from functools import lru_cache

from ruptures.utils import sanity_check
from ruptures.costs import cost_factory
from ruptures.base import BaseCost, BaseEstimator


class Dynp(BaseEstimator):

    """ Find optimal change points using dynamic programming.

    Given a segment model, it computes the best partition for which the sum of errors is minimum.

    """

    def __init__(self, model="l2", custom_cost=None, min_size=2, jump=5, params=None):
        """Creates a Dynp instance.

        Args:
            model (str, optional): segment model, ["l1", "l2", "rbf"]. Not used if ``'custom_cost'`` is not None.
            custom_cost (BaseCost, optional): custom cost function. Defaults to None.
            min_size (int, optional): minimum segment length.
            jump (int, optional): subsample (one every *jump* points).
            params (dict, optional): a dictionary of parameters for the cost instance.

        Returns:
            self
        """
        self.seg = lru_cache(maxsize=None)(self._seg)  # dynamic programming
        if custom_cost is not None and isinstance(custom_cost, BaseCost):
            self.cost = custom_cost
        else:
            if params is None:
                self.cost = cost_factory(model=model)
            else:
                self.cost = cost_factory(model=model, **params)
        self.min_size = max(min_size, self.cost.min_size)
        self.jump = jump
        self.n_samples = None

    def _seg(self, start, end, n_bkps):
        """Recurrence to find the optimal partition of signal[start:end].

        This method is to be memoized and then used.

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

        Dynamic programming is a recurrence; intermediate results are cached to speed up
        computations. This method sets up the cache.

        Args:
            signal (array): signal. Shape (n_samples, n_features) or (n_samples,).

        Returns:
            self
        """
        # clear cache
        self.seg.cache_clear()
        # update some params
        self.cost.fit(signal)
        self.n_samples = signal.shape[0]
        return self

    def predict(self, n_bkps):
        """Return the optimal breakpoints.

        Must be called after the fit method. The breakpoints are associated with the signal passed
        to fit().

        Args:
            n_bkps (int): number of breakpoints.

        Returns:
            list: sorted list of breakpoints
        """
        partition = self.seg(0, self.n_samples, n_bkps)
        bkps = sorted(e for s, e in partition.keys())
        return bkps

    def fit_predict(self, signal, n_bkps):
        """Fit to the signal and return the optimal breakpoints.

        Helper method to call fit and predict once

        Args:
            signal (array): signal. Shape (n_samples, n_features) or (n_samples,).
            n_bkps (int): number of breakpoints.

        Returns:
            list: sorted list of breakpoints
        """
        self.fit(signal)
        return self.predict(n_bkps)
