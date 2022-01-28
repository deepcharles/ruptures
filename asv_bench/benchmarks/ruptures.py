import ruptures as rpt

from . import parameterized


class Ruptures:
    def setup(self, *args, **kwargs):
        n_samples, dim, sigma = 1000, 3, 4
        n_bkps = 4  # number of breakpoints
        signal, bkps = rpt.pw_constant(n_samples, dim, n_bkps, noise_std=sigma)
        self.signal = signal

    @parameterized(
        ["algo"], [("Binseg", "BottomUp", "KernelCPD", "Pelt", "Window")]
    )
    def time_algos(self, algo):
        getattr(rpt, algo)().fit_predict(self.signal, pen=10)
