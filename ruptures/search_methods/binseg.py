"""

Binary changepoint detection.

Fast but approximate.

"""

from ruptures.costs import Cost
from ruptures.utils import pairwise


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
        stop = False
        bkps = [0, self.n_samples]
        old_cost = sum(self.cost.error(start, end)
                       for start, end in pairwise(bkps))
        while not stop:
            bkp_to_add = list()
            for (start, end) in pairwise(bkps):
                single_bkp = self.single_bkp(start, end)
                if single_bkp is None:
                    continue
                bkp, cost = single_bkp
                gain = self.cost.error(start, end) - cost
                if abs(gain) > 1e-8:
                    bkp_to_add.append((bkp, gain))
            bkp_to_add.sort(key=lambda x: x[-1], reverse=True)

            # stopping rules
            if len(bkp_to_add) > 0:
                for bkp, gain in bkp_to_add:
                    stop = True
                    if n_bkps is not None and len(bkps) - 2 < n_bkps:
                        bkps.append(bkp)
                        stop = False
                    elif pen is not None and abs(gain) > pen:
                        bkps.append(bkp)
                        stop = False
                    elif epsilon is not None and old_cost - gain > epsilon:
                        bkps.append(bkp)
                        old_cost -= gain
                        stop = False
                # update partition
                bkps.sort()
                old_cost = sum(self.cost.error(start, end)
                               for start, end in pairwise(bkps))
            else:
                stop = True

        partition = {(start, end): self.cost.error(start, end)
                     for start, end in pairwise(bkps)}
        return partition

    def single_bkp(self, start, end):
        """Return the optimal 2-regime partition of [start:end] (if it exists)."""
        bkps = list()
        for bkp in range(start, end):
            if bkp % self.jump == 0:
                if (bkp - start > self.min_size) and (end - bkp > self.min_size):
                    cost = self.cost.error(start, bkp)  # left
                    cost += self.cost.error(bkp, end)  # right
                    bkps += [(bkp, cost)]
        if len(bkps) > 0:
            return min(bkps, key=lambda x: x[1])
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
