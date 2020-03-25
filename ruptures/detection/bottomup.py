r"""
.. _sec-bottup:

Bottom-up segmentation
====================================================================================================

Description
----------------------------------------------------------------------------------------------------

Bottom-up change point detection is used to perform fast signal segmentation and is implemented in
:class:`ruptures.detection.BottomUp`.
It is a sequential approach.
Contrary to binary segmentation, which is a greedy procedure, bottom-up segmentation is generous:
it starts with many change points and successively deletes the less significant ones.
First, the signal is divided in many sub-signals along a regular grid.
Then contiguous segments are successively merged according to a measure of how similar they are.
See for instance :cite:`bu-Keogh2001` or :cite:`bu-Fryzlewicz2007` for an algorithmic
analysis of :class:`ruptures.detection.BottomUp`.
The benefits of bottom-up segmentation includes low complexity (of the order of
:math:`\mathcal{O}(n\log n)`, where :math:`n` is the number of samples), the fact that it can extend
any single change point detection method to detect multiple changes points and that it can work
whether the number of regimes is known beforehand or not.

.. figure:: /images/schema_tree.png
   :scale: 50 %
   :alt: Schematic view of the bottom-up segmentation algorithm

   Schematic view of the bottom-up segmentation algorithm.

.. seealso:: :ref:`sec-binseg`.

Usage
----------------------------------------------------------------------------------------------------

Start with the usual imports and create a signal.

.. code-block:: python

    import numpy as np
    import matplotlib.pylab as plt
    import ruptures as rpt
    # creation of data
    n, dim = 500, 3  # number of samples, dimension
    n_bkps, sigma = 3, 5  # number of change points, noise standart deviation
    signal, bkps = rpt.pw_constant(n, dim, n_bkps, noise_std=sigma)

To perform a bottom-up segmentation of a signal, initialize a :class:`ruptures.detection.BottomUp`
instance.

.. code-block:: python

    # change point detection
    model = "l2"  # "l1", "rbf", "linear", "normal", "ar"
    algo = rpt.BottomUp(model=model).fit(signal)
    my_bkps = algo.predict(n_bkps=3)

    # show results
    rpt.show.display(signal, bkps, my_bkps, figsize=(10, 6))
    plt.show()

In the situation in which the number of change points is unknown, one can specify a penalty using
the ``'pen'`` parameter or a threshold on the residual norm using ``'epsilon'``.

.. code-block:: python

    my_bkps = algo.predict(pen=np.log(n)*dim*sigma**2)
    # or
    my_bkps = algo.predict(epsilon=3*n*sigma**2)

.. seealso:: :ref:`sec-general-formulation` for more information about stopping rules of sequential algorithms.

For faster predictions, one can modify the ``'jump'`` parameter during initialization.
The higher it is, the faster the prediction is achieved (at the expense of precision).

.. code-block:: python

    algo = rpt.BottomUp(model=model, jump=10).fit(signal)


Code explanation
----------------------------------------------------------------------------------------------------

.. autoclass:: ruptures.detection.BottomUp
    :members:
    :special-members: __init__


.. rubric:: References

.. bibliography:: ../biblio.bib
    :style: alpha
    :cited:
    :labelprefix: BU
    :keyprefix: bu-

"""
import heapq
from bisect import bisect_left
from functools import lru_cache

from ruptures.base import BaseCost, BaseEstimator
from ruptures.costs import cost_factory
from ruptures.utils import Bnode, pairwise


class BottomUp(BaseEstimator):

    """Bottom-up segmentation."""

    def __init__(self, model="l2", custom_cost=None, min_size=2, jump=5, params=None):
        """Initialize a BottomUp instance.


        Args:
            model (str, optional): segment model, ["l1", "l2", "rbf"]. Not used if ``'custom_cost'`` is not None.
            custom_cost (BaseCost, optional): custom cost function. Defaults to None.
            min_size (int, optional): minimum segment length. Defaults to 2 samples.
            jump (int, optional): subsample (one every *jump* points). Defaults to 5 samples.
            params (dict, optional): a dictionary of parameters for the cost instance.

        Returns:
            self
        """
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
        self.signal = None
        self.leaves = None
        self.merge = lru_cache(maxsize=None)(self._merge)

    def _grow_tree(self):
        """Grow the entire binary tree."""
        partition = [(0, self.n_samples)]
        stop = False
        while not stop:  # recursively divide the signal
            stop = True
            start, end = max(partition, key=lambda t: t[1] - t[0])
            mid = (start + end) * 0.5
            bkps = list()
            for bkp in range(start, end):
                if bkp % self.jump == 0:
                    if bkp - start >= self.min_size and end - bkp >= self.min_size:
                        bkps.append(bkp)
            if len(bkps) > 0:  # at least one admissible breakpoint was found
                bkp = min(bkps, key=lambda x: abs(x - mid))
                partition.remove((start, end))
                partition.append((start, bkp))
                partition.append((bkp, end))
                stop = False

        partition.sort()
        # compute segment costs
        leaves = list()
        for start, end in partition:
            val = self.cost.error(start, end)
            leaf = Bnode(start, end, val)
            leaves.append(leaf)
        return leaves

    def _merge(self, left, right):
        """Merge two contiguous segments."""
        assert left.end == right.start, "Segments are not contiguous."
        start, end = left.start, right.end
        val = self.cost.error(start, end)
        node = Bnode(start, end, val, left=left, right=right)
        return node

    def _seg(self, n_bkps=None, pen=None, epsilon=None):
        """Compute the bottom-up segmentation.

        The stopping rule depends on the parameter passed to the function.

        Args:
            n_bkps (int): number of breakpoints to find before stopping.
            penalty (float): penalty value (>0)
            epsilon (float): reconstruction budget (>0)

        Returns:
            dict: partition dict {(start, end): cost value,...}
        """
        leaves = sorted(self.leaves)
        removed = set()
        merged = []
        for left, right in pairwise(leaves):
            candidate = self.merge(left, right)
            heapq.heappush(merged, (candidate.gain, candidate))
        # bottom up fusion
        stop = False
        while not stop:
            stop = True

            try:
                gain, leaf = heapq.heappop(merged)
                # Ignore any merge candidates whose left or right children
                # no longer exist (because they were merged with another node).
                # It's cheaper to do this here than during the initial merge.
                while leaf.left in removed or leaf.right in removed:
                    gain, leaf = heapq.heappop(merged)
            # if merged is empty (all nodes have been merged).
            except IndexError:
                break

            if n_bkps is not None:
                if len(leaves) > n_bkps + 1:
                    stop = False
            elif pen is not None:
                if gain < pen:
                    stop = False
            elif epsilon is not None:
                if sum(leaf_tmp.val for leaf_tmp in leaves) < epsilon:
                    stop = False

            if not stop:
                # updates the list of leaves (i.e. segments of the partitions)
                # find the merged segments indexes
                keys = [leaf.start for leaf in leaves]
                left_idx = bisect_left(keys, leaf.left.start)
                leaves[left_idx] = leaf  # replace leaf.left
                del leaves[left_idx + 1]  # remove leaf.right
                # add to the set of removed segments.
                removed.add(leaf.left)
                removed.add(leaf.right)
                # add new merge candidates
                if left_idx > 0:
                    left_candidate = self.merge(leaves[left_idx - 1], leaf)
                    heapq.heappush(merged,
                                   (left_candidate.gain, left_candidate))
                if left_idx < len(leaves) - 1:
                    right_candidate = self.merge(leaf, leaves[left_idx + 1])
                    heapq.heappush(merged,
                                   (right_candidate.gain, right_candidate))

        partition = {(leaf.start, leaf.end): leaf.val for leaf in leaves}
        return partition

    def fit(self, signal):
        """Compute params to segment signal.

        Args:
            signal (array): signal to segment. Shape (n_samples, n_features) or (n_samples,).

        Returns:
            self
        """
        # update some params
        self.cost.fit(signal)
        self.merge.cache_clear()
        if signal.ndim == 1:
            n_samples, = signal.shape
        else:
            n_samples, _ = signal.shape
        self.n_samples = n_samples
        self.leaves = self._grow_tree()
        return self

    def predict(self, n_bkps=None, pen=None, epsilon=None):
        """Return the optimal breakpoints.

        Must be called after the fit method. The breakpoints are associated with the signal passed
        to fit().
        The stopping rule depends on the parameter passed to the function.

        Args:
            n_bkps (int): number of breakpoints to find before stopping.
            penalty (float): penalty value (>0)
            epsilon (float): reconstruction budget (>0)

        Returns:
            list: sorted list of breakpoints
        """
        msg = "Give a parameter."
        assert any(param is not None for param in (n_bkps, pen, epsilon)), msg

        partition = self._seg(n_bkps=n_bkps, pen=pen, epsilon=epsilon)
        bkps = sorted(e for s, e in partition.keys())
        return bkps

    def fit_predict(self, signal, n_bkps=None, pen=None, epsilon=None):
        """Fit to the signal and return the optimal breakpoints.

        Helper method to call fit and predict once

        Args:
            signal (array): signal. Shape (n_samples, n_features) or (n_samples,).
            n_bkps (int): number of breakpoints.
            penalty (float): penalty value (>0)
            epsilon (float): reconstruction budget (>0)

        Returns:
            list: sorted list of breakpoints
        """
        self.fit(signal)
        return self.predict(n_bkps=n_bkps, pen=pen, epsilon=epsilon)
