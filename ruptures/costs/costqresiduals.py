import numpy as np
from numpy.linalg import svd, eig
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

from ruptures.base import BaseCost
from ruptures.costs import NotEnoughPoints

class CostQResiduals(BaseCost):
    
    """Q residuals."""
    
    model = "qresid"

    def __init__(self, n_components=2):
        self.min_size = 2
        self.signal = None
        self.n_components = n_components

    def fit(self, signal):
        """Set parameters of the instance.

        Args:
            signal (array): signal. Shape (n_samples,) or (n_samples, n_features)

        Returns:
            self
        """
        if signal.ndim == 1 or (signal.ndim == 2 and signal.shape[1] == 1):
            raise ValueError("The signal must be multivariate.")
        else:
            self.signal = signal

        return self

    def error(self, start, end):
        """Return the approximation cost on the segment [start:end].

        Args:
            start (int): start of the segment
            end (int): end of the segment

        Returns:
            float: segment cost

        Raises:
            NotEnoughPoints: when the segment is too short (less than ``'min_size'`` samples).
        """
        if end - start < self.min_size:
            raise NotEnoughPoints
        sub = self.signal[start:end]

        pca = PCA(self.n_components).fit(sub)

        # loading vectors
        P = pca.components_.T
        # compute residuals
        X = sub - np.mean(sub, axis=0)
        Q = X @ (np.eye(sub.shape[1]) - P @ P.T) @ X.T
        
        return np.sum(np.diag(Q))



