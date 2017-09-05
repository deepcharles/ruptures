"""
For a given model and known number of change points, the change point detection problem amounts to
minimize the approximation error over all the potential breakpoint repartitions.
Thanks to the additive nature of the change point detection problem, a dynamic programming
algorithm is able to find the optimal partition which minimizes a sum of costs measuring
approximation error.
Formally,

.. math:: \widehat{\mathbf{p}}_K = \\arg \min_{\mathbf{p}} \sum_{i=1}^{K} c(y_{p_i}).

The method is implemented in :class:`ruptures.detection.Dynp`.

.. _sec-costs:

Available cost functions:
----------------------------------------------------------------------------------------------------

- Squared residuals:
    .. math:: c(y_{p_i}) = \sum_{t\in p_i} \|y_t - \\bar{y}\|^2_2

    where :math:`\\bar{y}=\\frac{1}{|p_i|} \sum\limits_{t\in p_i} y_t`.

    This cost function is suited to approximate piecewise constant signals corrupted with noise.

- Absolute deviation:
    .. math:: c(y_{p_i}) = \min_u \sum_{t\in p_i} \|y_t - u\|_1

    This cost function is suited to approximate piecewise constant signals corrupted with
    non-Gaussian noise (following for instance a heavy-tailed distribution).

- Negative maximum log-likelihood (Gaussian density):
    .. math:: c(y_{p_i}) = |p_i| \log\det\widehat{\Sigma}_i

    where :math:`\widehat{\Sigma} = \\frac{1}{|p_i|}\sum\limits_{t\in p_i} (y_t - \\bar{y}) (y_t - \\bar{y})^T`.

    This cost function is suited to approximate piecewise i.i.d. Gaussian variables, for instance
    mean-shifts and scale-shifts.


Examples
----------------------------------------------------------------------------------------------------
.. code-block:: python

    import numpy as np
    import matplotlib.pylab as plt
    import ruptures as rpt

    # creation of data
    n, dim = 500, 3
    n_bkps, sigma = 3, 1
    signal, b = rpt.pw_constant(n, dim, n_bkps, noisy=True, sigma=sigma)

    # change point detection
    model = "constantl1"  # "constantl2", "rbf"
    algo = rpt.Dynp(model="constantl1", min_size=3, jump=5).fit(signal)
    my_bkps = algo.predict(n_bkps=3)

    # show results
    fig, (ax,) = rpt.display(signal, bkps, my_bkps, figsize=(10, 6))
    plt.show()

Code explanation
====================================================================================================

.. autoclass:: ruptures.detection.Dynp
    :members:
    :special-members: __init__

"""
from functools import lru_cache

from ruptures.utils import sanity_check
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
            jump (int, optional): subsample (one every *jump* points)

        Returns:
            self
        """
        self.model = model
        self.min_size = min_size
        self.jump = jump
        self.seg = lru_cache(maxsize=None)(self._seg)  # dynamic programming
        self.cost = Cost(model=self.model)
        self.n_samples = None

    def _seg(self, start, end, n_bkps):
        """Reccurence to find best partition of signal[start:end].

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

        Dynamic programming is a recurrence. Therefore intermediate results are cached to speed up
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
