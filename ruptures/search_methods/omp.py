"""

Orthogonal matching pursuit for changepoint detection.

Fast but approximate.

"""
import numpy as np
from numpy.linalg import norm, lstsq


class Omp:

    """Contient l'algorithme de parcours des partitions."""

    def __init__(self, min_size=2, jump=1):
        """One line description

        Detailled description

        Args:
            model (str): constantl1|constantl2|rbf
            min_size (int, optional): minimum segment length
            jump (int, optional): subsample (one every "jump" points)

        Returns:
            self
        """
        self.min_size = min_size
        self.jump = jump
        self.n_samples = None
        self.signal = None
        self.X = None  # discrete derivative matrix

    def projection(self, bkps):
        """Return orthogonal projection on span of piecewise constant signals."""
        proj_ind = [bkp - 1 for bkp in bkps
                    if bkp != 0 and bkp != self.n_samples]
        x_proj = self.X[:, proj_ind]
        proj_coef, _, _, _ = lstsq(x_proj, self.signal)
        proj = x_proj.dot(proj_coef)
        return proj

    def seg(self, n_bkps=None, pen=None, epsilon=None):
        """Computes the binary segmentation.

        The stopping rule depends on the parameter passed to the function.

        Args:
            n_bkps (int): number of breakpoints to find before stopping.
            penalty (float): penalty value (>0)
            epsilon (float): reconstruction budget

        Returns:
            dict: partition dict {(start, end): cost value,...}
        """
        stop = False
        bkps = [0, self.n_samples]
        inds = list(range(0, self.n_samples - 1, self.jump))
        residual = self.signal
        res_norm = norm(residual)

        while not stop:
            filter_fun = lambda ind: all(
                abs(bkp - ind) >= self.min_size for bkp in bkps)
            admissible = [ind - 1 for ind in inds if filter_fun(ind)]
            if len(admissible) == 0:  # no more admissible points
                break
            # greedy search
            x_adm = self.X[:, admissible]
            correlations = x_adm.T.dot(residual)
            correlations **= 2
            bkp_opt, _ = max(zip(
                admissible,
                correlations.sum(axis=1)), key=lambda x: x[1])
            bkp_opt += 1
            # orthogonal projection
            proj = self.projection(bkps + [bkp_opt])
            residual = self.signal - proj
            # stopping criterion
            stop = True
            if n_bkps is not None:
                if len(bkps) - 2 < n_bkps:
                    stop = False
            elif pen is not None:
                if res_norm - norm(residual) > pen:
                    stop = False
            elif epsilon is not None:
                if norm(residual) > epsilon:
                    stop = False
            # update
            if not stop:
                res_norm = norm(residual)
                bkps.append(bkp_opt)

        bkps.sort()
        return bkps[1:]

    def fit(self, signal):
        """Compute params to segment signal.

        Args:
            signal (array): signal to segment. Shape (n_samples, n_features) or (n_samples,).

        Returns:
            self
        """
        # update some params
        if signal.ndim == 1:
            self.signal = signal.reshape(-1, 1)
        else:
            self.signal = signal
        self.signal -= self.signal.mean(axis=0)
        self.n_samples, _ = self.signal.shape

        self.X = np.tri(self.n_samples, self.n_samples - 1, k=-1) * 1.
        self.X -= self.X.mean(axis=0)
        self.X /= norm(self.X, axis=0)

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

        bkps = self.seg(n_bkps=n_bkps, pen=pen, epsilon=epsilon)
        return bkps

    def fit_predict(self, signal, n_bkps=None, pen=None, epsilon=None):
        """Helper method to call fit and predict once."""
        self.fit(signal)
        return self.predict(n_bkps=n_bkps, pen=pen, epsilon=epsilon)
