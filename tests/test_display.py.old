from ruptures.datasets import pw_constant
from ruptures.show import display


def test_display():
    d = 7  # nombre de régimes
    n_samples = 1000  # nombre de points
    min_size = 50  # taille minimale de segment
    snr = 0.05  # signal to noise ratio: 0 --> pas de bruit.
    signal, chg = pw_constant(n=n_samples,
                              clusters=d,
                              min_size=min_size,
                              noisy=True,
                              snr=snr)
    fig, ax = display(signal, chg)
    fig, ax = display(signal, chg, chg)
    fig, ax = display(signal, chg,
                      figsize=(10, 10),
                      alpha=0.1,
                      color="b",
                      linewidth=1,
                      linestyle="--")
