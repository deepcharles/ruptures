import numpy as np
from ruptures.costs.exceptions import NotEnoughPoints
from ruptures.search_methods import changepoint


@changepoint
class HarmonicMSE:

    def error(self, start, end):

        assert 0 <= start <= end

        if end - start < 20:  # we need at least 20 points
            raise NotEnoughPoints

        res = 0
        # we loop over the dimensions:
        for sig in self.signal[start:end].T:
            # Fourier transform
            fourier = np.fft.rfft(sig.flatten())

            # coefficient of the biggest harmonic (except the DC)
            k, coef_max = max(enumerate(fourier[1:], start=1),
                              key=lambda z: abs(z[1]))
            # DC component
            coef_dc = fourier[0]

            # we put all other components to zero
            fourier_sparse = np.zeros(fourier.shape, dtype=complex)
            fourier_sparse[0] = coef_dc
            fourier_sparse[k] = coef_max

            # we reconstruct a periodic signal
            sig_reconstruct = np.fft.irfft(fourier_sparse, len(sig))

            residuals = sig - sig_reconstruct.reshape(sig.shape)
            res += abs(residuals).sum()

        return res

    def set_params(self):
        pass

    @property
    def K(self):
        return 0
