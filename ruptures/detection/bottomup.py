"""

Binary changepoint detection.

Fast but approximate.

"""
from functools import lru_cache

from ruptures.costs import Cost
from ruptures.utils import Bnode, pairwise


class BottomUp:

    """Contient l'algorithme de parcours des partitions."""

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
        self.cost = Cost(model=self.model)
        self.n_samples = None
        self.leaves = None
        self.merge = lru_cache(maxsize=None)(self._merge)

    def grow_tree(self):
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

    def seg(self, n_bkps=None, pen=None, epsilon=None):
        """Computes the binary segmentation.

        The stopping rule depends on the parameter passed to the function.

        Args:
            n_bkps (int): number of breakpoints to find before stopping.
            penalty (float): penalty value (>0)
            penalty (float): penalty value

        Returns:
            dict: partition dict {(start, end): cost value,...}
        """
        leaves = list(self.leaves)
        # bottom up fusion
        stop = False
        while not stop:
            stop = True
            leaves.sort(key=lambda n: n.start)
            merged = (self.merge(left, right)
                      for left, right in pairwise(leaves))
            # find segment to merge
            leaf = min(merged, key=lambda n: n.gain)
            if n_bkps is not None:
                if len(leaves) > n_bkps + 1:
                    stop = False
            elif pen is not None:
                if leaf.gain < pen:
                    stop = False
            elif epsilon is not None:
                if sum(leaf.val for leaf in leaves) < epsilon:
                    stop = False

            if not stop:
                leaves.remove(leaf.left)
                leaves.remove(leaf.right)
                leaves += [leaf]

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
        self.leaves = self.grow_tree()
        return self

    def predict(self, n_bkps=None, pen=None, epsilon=None):
        """Return the optimal breakpoints.

        Must be called after the fit method. The breakpoints are associated with the signal passed
        to fit().
        The stopping rule depends on the parameter passed to the function.

        Args:
            n_bkps (int): number of breakpoints to find before stopping.
            penalty (float): penalty value (>0)
            penalty (float): penalty value

        Returns:
            list: sorted list of breakpoints
        """
        msg = "Give a parameter."
        assert any(param is not None for param in (n_bkps, pen, epsilon)), msg

        partition = self.seg(n_bkps=n_bkps, pen=pen, epsilon=epsilon)
        bkps = sorted(e for s, e in partition.keys())
        return bkps

    def fit_predict(self, signal, n_bkps=None, pen=None, epsilon=None):
        """Helper method to call fit and predict once."""
        self.fit(signal)
        return self.predict(n_bkps=n_bkps, pen=pen, epsilon=epsilon)
