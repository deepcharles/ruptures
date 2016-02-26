import numpy as np
from ruptures.costs.exceptions import NotEnoughPoints
from ruptures.search_methods import changepoint


@changepoint
class HarmonicMSE:

    def error(self, start, end):
        """Cost between on the self.signal[start:end] signal

        Args:
            start (int): start index
            end (int): end index

        Raises:
            NotEnoughPoints: if there are not enough points

        Returns:
            float: the cost
        """
        assert 0 <= start <= end

        if end - start < 20:  # we need at least 20 points
            raise NotEnoughPoints

        res = 0
        # we loop over the dimensions:
        for sig in self.signal[start:end].T:
            # Fourier transform
            fourier = np.fft.rfft(sig.flatten())

            # We want to calculate the difference between the signal and a
            # signal reconstructed from the DC component and the biggest
            # harmonic. The difference is the residuals.

            # coefficient of the biggest harmonic (except the DC)
            k_max, _ = max(enumerate(fourier[1:], start=1),
                           key=lambda z: abs(z[1]))

            fourier[0] = 0  # DC component to 0
            fourier[k_max] = 0  # biggest harmonic to zero.

            residuals = np.fft.irfft(fourier, len(sig))
            res += abs(residuals**2).sum()

        return res

    def set_params(self):
        pass

    @property
    def K(self):
        return 0
