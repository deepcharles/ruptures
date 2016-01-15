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


def kernel_mean(signal, kernel="rbf"):
    """
    Returns the cost function.
    Here:
    \sum_{i} k(x_i, x_i) - \frac{1}{n} \sum_{i,j} k(x_i, x_j)
    Associated K (see Pelt.__init__): 0

    signal: array of shape (n_points,) or (n_points, 1)
    """
    s = signal
    if s.ndim == 1:
        s = s.reshape(-1, 1)

    # on a une liste restreinte de noyaux possibles
    assert kernel in valid_kernels

    # correspondance nom de noyau <-> fonction associée
    func_dict = {"rbf": rbf,
                 "laplacian": laplacian,
                 "cosine": cosine,
                 "linear": linear}

    ker = func_dict[kernel]
    norm_vect, K = ker(s)

    def error_func(start, end):
        assert 0 <= start <= end

        if end - start < 2:  # we need at least 3 points
            raise NotEnoughPoints
        res = np.sum(norm_vect[start:end])
        res -= K[start:end, start:end].sum() / (end - start)
        return res

    return error_func
