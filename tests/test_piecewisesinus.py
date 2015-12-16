import numpy as np
import ruptures.dynamic_programming.piecewise_sinus as ps
import matplotlib.pyplot as plt
import scipy.signal as si
import itertools


def create_signal(n_samples, Fs, freqs):
    """
    n_samples: int.
    Fs: int. Sampling frequency
    freqs: list of frequencies in Hz
    """
    time = np.arange(n_samples) / Fs
    n = int(n_samples / len(freqs))
    tmp_freqs = freqs + [freqs[-1]]
    signal = np.array(list(itertools.chain(*[[f] * n for f in tmp_freqs])))
    signal = signal[:n_samples]
    signal = np.cos(2 * np.pi * signal * time)
    return time, signal


def plot_specgram(time, freqs, ft):
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(time, freqs, ft.T, cmap=plt.cm.jet)
    return heatmap


def test_stft():
    n_samples, Fs = 500, 100
    time = np.arange(n_samples) / Fs
    signal = si.chirp(time, f0=0, t1=time[-1], f1=Fs / 3) + 10
    Nw, noverlap, pad_to = 40, 39, 100
    spec, freqs, t = ps.stft(
        signal, Fs=Fs, Nw=Nw, noverlap=noverlap, pad_to=pad_to)
    # plot_specgram(t, freqs, spec)
    # plt.show()


def test_fourier():
    n_samples, Fs = 2000, 100
    rfreqs = [10, 5, 20]
    #  we create a signal
    time, signal = create_signal(n_samples, Fs, rfreqs)
    # we plot the signal
    # plt.plot(time, signal)
    # plt.show()
    # we show the specgram
    Nw, noverlap, n_freq = 50, 49, 50
    # premier test
    f = ps.Fourier(signal, Fs, Nw, noverlap, n_freq)
    # we plot the results
    # ft, freqs, time = f.error.ft, f.error.freqs, f.error.time
    # plot_specgram(time, freqs, ft)
    # plt.show()
    rupt = f.fit(len(rfreqs), jump=1, min_size=10)
    rupt1 = np.array([s for (s, e) in rupt.keys() if s != 0])
    # plt.plot(time, signal)
    # plt.vlines(rupt / Fs, ymin=signal.min(), ymax=signal.max(),
    #            linestyle="dashed",
    #            linewidth=4)
    # plt.show()
    # deuxième test
    f = ps.Fourier(signal, Fs, Nw, None, None)
    rupt = f.fit(len(rfreqs), jump=1, min_size=10)
    rupt2 = np.array([s for (s, e) in rupt.keys() if s != 0])
    assert rupt1.sort() == rupt2.sort()
