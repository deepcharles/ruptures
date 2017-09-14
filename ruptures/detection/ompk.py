"""

Kernel orthogonal matching pursuit for changepoint detection.

Fast but approximate.

"""
from itertools import product

import numpy as np

from ruptures.utils import pairwise


class OmpK:

    """Contient l'algorithme de parcours des partitions."""

    def __init__(self, min_size=2, jump=5):
        """One line description

        Detailled description

        Args:
            min_size (int, optional): minimum segment length
            jump (int, optional): subsample (one every "jump" points)

        Returns:
            self
        """
        self.min_size = min_size  # not used
        self.jump = jump  # not used
        self.n_samples = None
        self.gram = None

    def _seg(self, n_bkps=None, pen=None, epsilon=None):
        """Computes the binary segmentation.

        The stopping rule depends on the parameter passed to the function.

        Args:
            n_bkps (int): number of breakpoints to find before stopping.
            penalty (float): penalty value (>0)
            epsilon (float): reconstruction budget

        Returns:
            list: list of breakpoint indexes
        """
        stop = False
        bkps = [self.n_samples]
        inds = np.arange(1, self.n_samples + 1)
        csum = self.gram.cumsum(axis=0).cumsum(axis=1)
        residual = csum[-1, -1]

        while not stop:
            # greedy search
            correlation = np.diag(csum) * self.n_samples * self.n_samples
            correlation += inds**2 * csum[-1, -1]
            correlation -= 2 * self.n_samples * inds * csum[-1]
            correlation /= inds * inds[::-1]
            bkp = np.argmax(correlation) + 1

            # orthogonal projection (matrix form)
            # adj = np.zeros(self.gram.shape)  # adjacency matrix
            # for start, end in pairwise(sorted([0, bkp] + bkps)):
            #     duree = end - start
            #     adj[start:end, start:end] = np.ones(duree, duree) / duree
            # gram_new = self.gram + adj @ self.gram @ adj - adj @ self.gram - self.gram @ adj
            # csum = gram_new.cumsum(axis=0).cumsum(axis=1)

            # orthogonal projection (vectorized form)
            gram_new = self.gram.copy()
            # cross product
            cross_g = np.zeros(self.gram.shape)
            for start, end in pairwise(sorted([0, bkp] + bkps)):
                val = self.gram[:, start:end].mean(axis=1).reshape(-1, 1)
                cross_g[:, start:end] = val
            gram_new -= cross_g + cross_g.T
            # products of segment means
            for p, q in product(pairwise(sorted([0, bkp] + bkps)), repeat=2):
                start1, end1 = p
                start2, end2 = q
                gram_new[start1:end1, start2:end2] += self.gram[
                    start1:end1, start2:end2].mean()
            csum = gram_new.cumsum(axis=0).cumsum(axis=1)

            # stopping criterion
            stop = True
            if n_bkps is not None:
                if len(bkps) - 1 < n_bkps:
                    stop = False
            elif pen is not None:
                if residual - csum[-1, -1] > pen:
                    stop = False
            elif epsilon is not None:
                if csum[-1, -1] > epsilon:
                    stop = False
            # update
            if not stop:
                residual = csum[-1, -1]
                bkps.append(bkp)

        bkps.sort()
        return bkps

    def fit(self, gram):
        """Compute params to segment signal.

        Args:
            gram (array): Gram matrix of signal to segment. Shape (n_samples, n_samples).

        Returns:
            self
        """
        assert gram.shape[0] == gram.shape[1], "Not a square matrix."
        # update some params
        self.gram = gram
        self.n_samples, _ = self.gram.shape

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

        bkps = self._seg(n_bkps=n_bkps, pen=pen, epsilon=epsilon)
        return bkps

    def fit_predict(self, gram, n_bkps=None, pen=None, epsilon=None):
        """Helper method to call fit and predict once."""
        self.fit(gram)
        return self.predict(n_bkps=n_bkps, pen=pen, epsilon=epsilon)
