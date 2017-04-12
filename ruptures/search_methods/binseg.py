"""

Binary changepoint detection.

Fast but approximate.

"""

from ruptures.costs import Cost
from ruptures.utils import Bnode


class Binseg:

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
        self.root = None

    def grow_tree(self):
        """Grow the entire binary tree."""
        root = Bnode(start=0, end=self.n_samples,
                     val=self.cost.error(0, self.n_samples))
        leaves = [root]
        stop = False
        while not stop:
            new_leaves = list()
            for node in leaves:
                if self.single_bkp(node) is not None:
                    new_leaves += [node.left, node.right]
            stop = len(new_leaves) == 0
            leaves = list(new_leaves)
        return root

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

        # initialization
        leaves = [self.root]
        stop = False
        while not stop:
            stop = True
            node = max(leaves, key=lambda x: x.gain)  # best segment to split

            if node.left is None or node.right is None:  # We've reach the bottom of the tree
                stop = True
            elif n_bkps is not None:
                if len(leaves) < n_bkps + 1:
                    stop = False
            elif pen is not None:
                if node.gain > pen:
                    stop = False
            elif epsilon is not None:
                if sum(leaf.val for leaf in leaves) > epsilon:
                    stop = False

            if not stop:
                leaves.remove(node)
                leaves += [node.left, node.right]

        partition = {(leaf.start, leaf.end): leaf.val for leaf in leaves}
        return partition

    def single_bkp(self, node):
        """Return the optimal 2-regime partition of [start:end] (if it exists)."""
        bkps = list()
        start, end = node.start, node.end
        for bkp in range(start, end):
            if bkp % self.jump == 0:
                if (bkp - start >= self.min_size) and (end - bkp >= self.min_size):
                    left_cost = self.cost.error(start, bkp)  # left
                    right_cost = self.cost.error(bkp, end)  # right
                    bkps += [(bkp, left_cost, right_cost)]
        if len(bkps) > 0:
            bkp, left_cost, right_cost = min(bkps, key=lambda x: x[1] + x[2])
            node.left = Bnode(start=start, end=bkp, val=left_cost)
            node.right = Bnode(start=bkp, end=end, val=right_cost)
            return node.left, node.right
        else:
            return None

    def fit(self, signal):
        """Compute params to segment signal.

        Args:
            signal (array): signal to segment. Shape (n_samples, n_features) or (n_samples,).

        Returns:
            self
        """
        # update some params
        self.cost.fit(signal)
        if signal.ndim == 1:
            n_samples, = signal.shape
        else:
            n_samples, _ = signal.shape
        self.n_samples = n_samples
        self.root = self.grow_tree()
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
