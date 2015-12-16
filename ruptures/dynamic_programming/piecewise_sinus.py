from ruptures.base import BaseEstimator
import numpy as np
import ruptures.dynamic_programming.dynamic_programming as d_prog
import matplotlib.mlab as ml
import scipy.interpolate as ipol


def stft(signal, Fs, Nw, noverlap, pad_to):
    """ Computes the stft at distant points then linearly interpolates the stft
    and returns the value of the transform at each time stamp where the signal
    exists."""
    ft, freqs, t = ml.specgram(x=signal, Fs=Fs, NFFT=Nw, noverlap=noverlap,
                               pad_to=pad_to,
                               window=ml.window_hanning,
                               sides="onesided",
                               detrend="mean")
    ft = ft.T
    assert ft.shape == (len(t), len(freqs))
    # we add the stft before time 0 and after time -1 (resp. equal to the first
    # and last value of the stft).
    ft = np.array([ft[0]] + list(ft) + [ft[-1]])
    t = np.array([0] + list(t) + [len(signal) - 1])
    times = np.arange(len(signal)) / Fs
    func = ipol.interp1d(t, ft.T)  # linear interpolation
    ft = np.array([func(tt) for tt in times])
    assert ft.shape == (len(times), len(freqs))

    return ft, freqs, times


class FourierError:

    def __init__(self, signal, Fs=100, Nw=100, noverlap=None, n_freq=None):
        s = np.array(signal)
        assert s.ndim == 1
        self.s = s
        self.Fs = Fs
        self.n = self.s.shape[0]
        self._Nw = Nw  # length of the sliding window
        if n_freq is not None:  # number of points two consecutive windows
            # share
            self._noverlap = noverlap
        else:
            self._noverlap = Nw - 1
        self._Fs = Fs  # sampling frequency
        if n_freq is not None:  # twice the number of frequencies where the fft
            # is computed (equivalent to the "pad_to" argument in
            # matplotlib.mlab.specgram function)
            self._nfreq = n_freq
        else:
            self._nfreq = self.Fs
        self._pad_to = 2 * self._nfreq
        self.ft, self.freqs, self.time = stft(self.s,
                                              Fs=self.Fs,
                                              Nw=self._Nw,
                                              noverlap=self._noverlap,
                                              pad_to=self._pad_to)
        # Gram matrice of the short term Fourier transform
        self.gram = np.dot(self.ft, self.ft.T)
        # Cumulated norm
        self.norm = np.diag(self.gram).cumsum()
        # integrated Gram matrix
        self.integral_m = self.gram.cumsum(axis=0).cumsum(axis=1)

    def error_func(self, start, end):
        """
        This computes the error when the segment [start, end] is approached by
        a constant (its mean):
        Let m denote the integer segment [start, end]
        error = sum_{i in m} norm(X_i)^2  - 1/(end - start + 1)
            sum_{i,j in m} <X_i|X_j>
        :param start: the first index of the segment
        :param end: the last index of the signal
        :return: float. The approximation error
        """
        if start == 0:
            return (- self.integral_m[end, end] / (end - start + 1) +
                    self.norm[end])
        else:
            res = self.integral_m[start - 1, start - 1]
            res += self.integral_m[end, end]
            res -= self.integral_m[start - 1, end]
            res -= self.integral_m[end, start - 1]
            res /= end - start + 1
            res *= - 1.
            res += self.norm[end] - self.norm[start - 1]
            return res


class Fourier(BaseEstimator):

    def __init__(self, signal, Fs=100, Nw=100, noverlap=None, n_freq=None):
        self.error = FourierError(
            signal, Fs=Fs, Nw=Nw, noverlap=noverlap, n_freq=n_freq)
        self.n = self.error.n

    def fit(self, d, jump=1, min_size=1):
        self.partition = d_prog.dynamic_prog(
            self.error.error_func, d, 0, self.n - 1, jump, min_size)
        return self.partition
