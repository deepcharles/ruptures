from ruptures.datasets import pw_linear, pw_constant


def test1_datasets():
    n_regimes = 5
    n_samples = 500

    # Piecewise constant signal
    signal, chg_pts = pw_constant(n=n_samples, clusters=n_regimes,
                                  min_size=50, noisy=True, snr=0.1)
    signal, chg_pts = pw_constant()
    # Piecewise linear signal
    signal, chg_pts = pw_linear(n=n_samples, clusters=n_regimes,
                                min_size=50, noisy=True, snr=0.1)
    signal, chg_pts = pw_linear()
