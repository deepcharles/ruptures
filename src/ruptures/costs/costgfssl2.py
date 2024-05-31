r"""CostL2 (least squared deviation) after GFSS rotation (low-pass graph
spectral filtering)"""

import numpy as np

from scipy.linalg import eigh
from ruptures.costs import NotEnoughPoints
from ruptures.base import BaseCost


class CostGFSSL2(BaseCost):
    """Applies the GFSS rotation to the whole signal before computing the
    standard L2 cost for fixed variance gaussian hypothesis."""

    model = "gfss_l2_cost"
    min_size = 1

    def __init__(self, laplacian_mat, cut_sparsity) -> None:
        """
        Args:
            laplacian_mat (array): the discrete Laplacian matrix of the graph: D - W
            where D is the diagonal matrix diag(d_i) of the node degrees and W the adjacency matrix
            cut_sparsity (float): frequency threshold of the GFSS spectral filter
        """
        self.cut_sparsity = cut_sparsity
        self.graph_laplacian_mat = laplacian_mat
        self.signal = None
        self.gfss_square_cumsum = None
        self.gfss_cumsum = None
        super().__init__()

    def filter(self, freqs, eps=0.00001):
        """Applies the GFSS filter to the input (spatial) frequencies.
        NOTE: the frequencies must be in increasing order.

        Args:
            freqs (array): ordered frequencies to filter.
            eps (float, optional): threshold for non zero values. Defaults to 0.00001.

        Returns:
            filtered_freqs (array): the output of the filter.
        """
        nb_zeros = np.sum(freqs < eps)
        filtered_freqs = np.minimum(1, np.sqrt(self.cut_sparsity / freqs[nb_zeros:]))
        return np.concatenate([np.zeros(nb_zeros), filtered_freqs])

    def fit(self, signal):
        """Performs pre-computations for per-segment approximation cost.

        NOTE: the number of dimensions of the signal and their ordering
        must match those of the nodes of the graph.
        The function eigh used below returns the eigenvector corresponding to
        the ith eigenvalue in the ith column eigvect[:, i]

        Args:
            signal (array): of shape [n_samples, n_dim].
        """
        self.signal = signal
        # Computation of the GFSS
        eigvals, eigvects = eigh(self.graph_laplacian_mat)
        filter_matrix = np.diag(self.filter(eigvals), k=0)
        gfss = filter_matrix.dot(eigvects.T.dot(signal.T)).T
        # Computation of the per-segment cost utils
        self.gfss_square_cumsum = np.concatenate(
            [np.zeros((1, signal.shape[1])), np.cumsum(gfss**2, axis=0)], axis=0
        )
        self.gfss_cumsum = np.concatenate(
            [np.zeros((1, signal.shape[1])), np.cumsum(gfss, axis=0)], axis=0
        )
        return self

    def error(self, start, end):
        """Return the L2 approximation cost on the segment [start:end] where
        end is excluded, over the filtered signal.

        Args:
            start (int): start of the segment
            end (int): end of the segment

        Returns:
            float: segment cost
        """
        if end - start < self.min_size:
            raise NotEnoughPoints

        sub_square_sum = self.gfss_square_cumsum[end] - self.gfss_square_cumsum[start]
        sub_sum = self.gfss_cumsum[end] - self.gfss_cumsum[start]
        return np.sum(sub_square_sum - (sub_sum**2) / (end - start))
