import numpy as np
from ruptures.costs.exceptions import NotEnoughPoints
from scipy.spatial.distance import pdist, squareform

valid_kernels = ["rbf", "laplacian", "cosine", "linear"]


def rbf(x):
    s = x.shape
    assert len(s) == 2
    n, _ = s
    # let's compute the pairwise kernel values
    pairwise_dists = pdist(x, 'sqeuclidean')
    pairwise_dists /= np.median(pairwise_dists)  # scaling
    K = squareform(pairwise_dists)
    K = np.exp(-K)
    norm_vect = np.diag(K)
    return norm_vect, K


def laplacian(x):
    s = x.shape
    assert len(s) == 2
    n, _ = s
    # let's compute the pairwise kernel values
    pairwise_dists = pdist(x, 'euclidean')
    pairwise_dists /= np.median(pairwise_dists)  # scaling
    K = squareform(pairwise_dists)
    K = np.exp(-K)
    norm_vect = np.diag(K)
    return norm_vect, K


def linear(x):
    s = x.shape
    assert len(s) == 2
    n, _ = s
    K = np.dot(x, x.T)
    norm_vect = np.diag(K)
    return norm_vect, K


def cosine(x):
    s = x.shape
    assert len(s) == 2
    n, _ = s
    # let's compute the pairwise kernel values
    pairwise_dists = squareform(pdist(x, 'cosine'))
    K = 1 - pairwise_dists
    norm_vect = np.diag(K)
    return norm_vect, K


KERNEL_DICT = {"rbf": rbf, "laplacian": laplacian, "cosine": cosine,
               "linear": linear}


class KernelMSE(object):

    """Kernel changepoint detection"""

    def __init__(self, kernel="rbf"):
        super().__init__()
        assert kernel in valid_kernels
        self.kernel_str = kernel
        self.kernel_func = KERNEL_DICT[kernel]

        # a boolean to keep track of the expensive computation of certain
        # parameters
        self._computed = False

    def set_params(self):
        """Computed the pairwise kernel matrix and the norm vector

        Returns:
            None: just set self.norm_vect and self.pairwise_mat, self._computed
        """
        self.norm_vect, self.pairwise_mat = self.kernel_func(self.signal)
        self._computed = True

    def error(self, start, end):
        """Computes the error on the segment [start:end].
        Here: \sum_{i} k(x_i, x_i) - \frac{1}{n} \sum_{i,j} k(x_i, x_j)

        Args:
            start (int): first index of the segment (index included)
            end (int): last index of the segment (index excluded)

        Raises:
            NotEnoughPoints: if there are not enough points to compute the cost
            for the specified segment (here, at least 3 points).

        Returns:
            float: cost on the given segment.
        """
        assert 0 <= start <= end
        if not self._computed:
            self.set_params()

        if end - start < 2:  # we need at least 3 points
            raise NotEnoughPoints
        res = np.sum(self.norm_vect[start:end])
        res -= self.pairwise_mat[start:end, start:end].sum() / (end - start)
        return res

    @property
    def K(self):
        return 0
