from ruptures.datasets import pw_constant
from ruptures.show import display
import pytest


@pytest.fixture(scope="module")
def signal_bkps():
    n_samples = 300
    n_regimes = 3
    signal, bkps = pw_constant(n=n_samples,
                               clusters=n_regimes,
                               noisy=True,
                               snr=.01)
    return signal, bkps


def test_display(signal_bkps):
    signal, bkps = signal_bkps
    fig, ax = display(signal, bkps)
    fig, ax = display(signal, bkps, bkps)
    figsize = (20, 10)  # figure size
    alpha = 0.2
    color = "k"
    linewidth = 3
    linestyle = "--"
    fig, ax = display(signal, bkps, figsize=figsize, alpha=alpha, color=color,
                      linewidth=linewidth, linestyle=linestyle)
