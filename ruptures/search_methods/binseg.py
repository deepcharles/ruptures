"""

Binary changepoint detection.

Fast but approximate.

"""
import numpy as np
from ruptures.costs import Cost
from ruptures.utils import pairwise, admissible_filter


class Binseg:

    """Contient l'algorithme de parcours des partitions."""

    def __init__(self, model="constantl2", min_size=2, jump=1):
        """One line description

        Detailled description

        Args:
            model (str): constantl1|constantl2|rbf, defaults to constantl2
            min_size (int, optional): minimum segment length
            jump (int, optional): subsample (one every "jump" points)

        Returns:
            self
        """
        self.model = model  # not used
        self.min_size = min_size
        self.jump = jump
        self.cost = Cost(model="constantl2")
        self.n_samples = None
        self.signal = None

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
        bkps = [self.n_samples]
        keep_in_mind = dict()
        stop = False
        while not stop:
            stop = True
            pot = list()  # potential breakpoints
            for start, end in pairwise([0] + bkps):
                if (start, end) in keep_in_mind:
                    bkp, gain = keep_in_mind[(start, end)]
                else:
                    bkp, gain = self.single_bkp(start, end)
                    keep_in_mind[(start, end)] = (bkp, gain)
                pot.append((bkp, gain))

            bkp, gain = max(pot, key=lambda x: x[1])
            stop = True
            if n_bkps is not None:
                if len(bkps) - 1 < n_bkps:
                    stop = False
            elif pen is not None:
                if gain > pen:
                    stop = False
            elif epsilon is not None:
                error = sum(self.cost.error(start, end)
                            for start, end in pairwise([0] + bkps))
                if error > epsilon:
                    stop = False

            if not stop:
                bkps.append(bkp)
                bkps.sort()
        partition = {(start, end): self.cost.error(start, end)
                     for start, end in pairwise([0] + bkps)}
        return partition

    def single_bkp(self, start, end):
        """Return the optimal breakpoint of [start:end] (if it exists)."""
        filtre = admissible_filter(start, end, self.jump, self.min_size)

        subsignal = self.signal[
            start:end] - self.signal[start:end].mean(axis=0)
        objective = np.sum(subsignal.cumsum(axis=0)**2, axis=1)
        sub_sampling = ((ind, val) for ind, val in enumerate(objective, start=start + 1)
                        if filtre(ind))
        try:
            bkp, _ = max(sub_sampling, key=lambda x: x[1])
        except ValueError:  # if empty sub_sampling
            return None, 0
        gain = self.cost.error(start, end)
        gain -= self.cost.error(start, bkp) + self.cost.error(bkp, end)
        return bkp, gain

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
        self.n_samples, _ = self.signal.shape
        self.cost.fit(signal)

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
