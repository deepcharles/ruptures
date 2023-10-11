r"""Bottom-up segmentation."""
import heapq
from bisect import bisect_left
from functools import lru_cache

from ruptures.base import BaseCost, BaseEstimator
from ruptures.costs import cost_factory
from ruptures.utils import Bnode, pairwise, sanity_check
from ruptures.exceptions import BadSegmentationParameters


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

    def _grow_tree(self):
        """Grow the entire binary tree."""
        partition = [(-self.n_samples, (0, self.n_samples))]
        stop = False
        while not stop:  # recursively divide the signal
            stop = True
            _, (start, end) = partition[0]
            mid = (start + end) * 0.5
            bkps = list()
            for bkp in range(start, end):
                if bkp % self.jump == 0:
                    if bkp - start >= self.min_size and end - bkp >= self.min_size:
                        bkps.append(bkp)
            if len(bkps) > 0:  # at least one admissible breakpoint was found
                bkp = min(bkps, key=lambda x: abs(x - mid))
                heapq.heappop(partition)
                heapq.heappush(partition, (-bkp + start, (start, bkp)))
                heapq.heappush(partition, (-end + bkp, (bkp, end)))
                stop = False

        partition.sort(key=lambda x: x[1])
        # compute segment costs
        leaves = list()
        for _, (start, end) in partition:
            val = self.cost.error(start, end)
            leaf = Bnode(start, end, val)
            leaves.append(leaf)
        return leaves

    @lru_cache(maxsize=None)
    def merge(self, left, right):
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
        keys = [leaf.start for leaf in leaves]
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
                left_idx = bisect_left(keys, leaf.left.start)
                # replace leaf.left
                leaves[left_idx] = leaf
                keys[left_idx] = leaf.start
                # remove leaf.right
                del leaves[left_idx + 1]
                del keys[left_idx + 1]
                # add to the set of removed segments.
                removed.add(leaf.left)
                removed.add(leaf.right)
                # add new merge candidates
                if left_idx > 0:
                    left_candidate = self.merge(leaves[left_idx - 1], leaf)
                    heapq.heappush(merged, (left_candidate.gain, left_candidate))
                if left_idx < len(leaves) - 1:
                    right_candidate = self.merge(leaf, leaves[left_idx + 1])
                    heapq.heappush(merged, (right_candidate.gain, right_candidate))

        partition = {(leaf.start, leaf.end): leaf.val for leaf in leaves}
        return partition

    def fit(self, signal) -> "BottomUp":
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
            (n_samples,) = signal.shape
        else:
            n_samples, _ = signal.shape
        self.n_samples = n_samples
        self.leaves = self._grow_tree()
        return self

    def predict(self, n_bkps=None, pen=None, epsilon=None):
        """Return the optimal breakpoints.

        Must be called after the fit method. The breakpoints are associated with the signal passed
        to [`fit()`][ruptures.detection.bottomup.BottomUp.fit].
        The stopping rule depends on the parameter passed to the function.

        Args:
            n_bkps (int): number of breakpoints to find before stopping.
            pen (float): penalty value (>0)
            epsilon (float): reconstruction budget (>0)

        Raises:
            AssertionError: if none of `n_bkps`, `pen`, `epsilon` is set.
            BadSegmentationParameters: in case of impossible segmentation
                configuration

        Returns:
            list: sorted list of breakpoints
        """
        msg = "Give a parameter."
        assert any(param is not None for param in (n_bkps, pen, epsilon)), msg

        # raise an exception in case of impossible segmentation configuration
        if not sanity_check(
            n_samples=self.cost.signal.shape[0],
            n_bkps=0 if n_bkps is None else n_bkps,
            jump=self.jump,
            min_size=self.min_size,
        ):
            raise BadSegmentationParameters

        partition = self._seg(n_bkps=n_bkps, pen=pen, epsilon=epsilon)
        bkps = sorted(e for s, e in partition.keys())
        return bkps

    def fit_predict(self, signal, n_bkps=None, pen=None, epsilon=None):
        """Fit to the signal and return the optimal breakpoints.

        Helper method to call fit and predict once

        Args:
            signal (array): signal. Shape (n_samples, n_features) or (n_samples,).
            n_bkps (int): number of breakpoints.
            pen (float): penalty value (>0)
            epsilon (float): reconstruction budget (>0)

        Returns:
            list: sorted list of breakpoints
        """
        self.fit(signal)
        return self.predict(n_bkps=n_bkps, pen=pen, epsilon=epsilon)
