import numpy as np

import ruptures as rpt

from . import _skip_slow, parameterized, requires_dask


class Ruptures:
    def setup(self, *args, **kwargs):
        n_samples, dim, sigma = 1000, 3, 4
        n_bkps = 4  # number of breakpoints
        signal, bkps = rpt.pw_constant(n_samples, dim, n_bkps, noise_std=sigma)
        self.signal = signal

    @parameterized(["algo"], [("Binseg", "BottomUp", "Dynp", "KernelCPD", "Pelt", "Window")])
    def time_algos(self, algo, model):
        getattr(rpt, algo)().fit_predict(self.signal, pen=10)

