r"""L1Potts."""

from ruptures.base import BaseEstimator
from ruptures.exceptions import BadSegmentationParameters
import numpy as np


class L1Potts(BaseEstimator):
    
    """Penalized change point detection for piecewise constant signals with L1 data fidelity
    
    The L1 Potts model is defined as follows:

    \min_{u} \sum_{i=1}^N w_i |f_i - u_i| + \gamma \sum_{i=1}^{N-1} \mathbb{I}(u_i \neq u_{i+1})
    
    The algorithm implemented is described in the paper
    the paper 
    Storath, Weinmann, Unser. 
    Jump-penalized least absolute values estimation of scalar or circle-valued signals, 
    Information and Inference, 2017
     
    """
    
    def __init__(self):
        """Initialize a L1Potts instance.
        """
        self.jump = 1
        self.min_size = 1
        self.n_samples = None
    
    def fit(self, signal) -> "L1Potts":
        """Set params.

        Args:
            signal (array): signal to segment. Shape (n_samples)

        Returns:
            self
        """
        # update params
        signal = signal.squeeze()
        if signal.ndim == 1:
            (n_samples,) = signal.shape
        else:
            raise BadSegmentationParameters("L1Potts only accepts 1D signals.")
        self.n_samples = n_samples
        self.signal = signal
        return self
    
    def predict(self, pen, weights=None):
        
        f = self.signal
        
        if weights is None:
            weights = np.ones_like(f)

        N = len(f)
        v = np.unique(f)

        d = lambda x,y: np.abs(x - y)
        K = len(v)
        B = np.zeros((K, N))

        # tabulation
        B[:, 0] = d(v, f[0]) * weights[0]
        for n in range(1, N):
            z = np.min(B[:, n - 1])
            B[:, n] = d(v, f[n]) * weights[n] + np.minimum(z + pen, B[:, n - 1])

        # backtracking
        u = np.zeros(N)
        l = np.argmin(B[:, N - 1])
        u[N - 1] = v[l]
        for n in range(N - 2, -1, -1):
            B[l, n] -= pen
            l = np.argmin(B[:, n])
            u[n] = v[l]

        brkps_np = np.where(np.diff(u) != 0)[0] + 1
        
        # convert to same format ruptures uses
        brkps = brkps_np.tolist() + [N]
        return brkps
    
    def fit_predict(self, signal, pen):
        """Fit to the signal and return the optimal breakpoints.

        Helper method to call fit and predict once

        Args:
            signal (array): signal. Shape (n_samples)
            pen (float): penalty value (>0)

        Returns:
            list: sorted list of breakpoints
        """
        self.fit(signal)
        return self.predict(pen)

    def _compute_functional_value(self, bkps, pen):
        """
        Compute the functional value of the L1 Potts model.
        """
        functional_value = pen * (len(bkps) - 1)
        bkps = [0] + bkps
        
        for i in range(len(bkps) - 1):
            segment = self.signal[bkps[i]:bkps[i + 1]]
            functional_value += np.sum(np.abs(np.median(segment) - segment))
            
        return functional_value
    
# example usage and comparison with ruptures Pelt method
if __name__ == "__main__":
    import ruptures as rpt
    import time
    
    # Generate a synthetic signal
    n, dim = 2000, 1
    n_bkps, sigma = 10, 0.2
    signal, bkps = rpt.pw_constant(n, dim, n_bkps, noise_std=sigma, seed=1)
    pen = 0.5

    # L1Potts (This algorithm is not implemented in ruptures yet)
    l1potts = L1Potts()
    l1potts.fit(signal)
    
    time_start = time.time()
    potts_brkps = l1potts.predict(pen)
    time_potts = time.time() - time_start
    
    # Compare with Pelt method
    costL1 = rpt.costs.CostL1()
    costL1.min_size = 1
    algo = rpt.Pelt(custom_cost=costL1, min_size=1, jump=1).fit(signal)
    
    time_start = time.time()
    pelt_bkps = algo.predict(pen=pen)
    time_pelt = time.time() - time_start

    print("+++Execution times+++")
    print("Pelt:    ", time_pelt)
    print("L1Potts: ", time_potts)
    print("Speedup: ", time_pelt / time_potts)
    print("----------------")
    
    # compare the functional values
    fval_pelt = l1potts._compute_functional_value(pelt_bkps, pen)
    fval_l1potts = l1potts._compute_functional_value(potts_brkps, pen)
    print("+++Functional values+++")
    print("Pelt:    ", fval_pelt)
    print("L1Potts: ", fval_l1potts)
    print("Difference in functional values: ", fval_pelt - fval_l1potts)
    print("----------------")
    
    # Important note: the breakpoints might be different as the solution of the optimization problem is not unique in general
    # if the breakpoints are different, both solutions are optimal in the sense of the model as they give the same minimal functional value
    print("+++Breakpoints+++")
    print("Pelt:    ", pelt_bkps)
    print("L1Potts: ", potts_brkps)

    