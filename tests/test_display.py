import pytest

from ruptures.datasets import pw_constant
from ruptures.show import display
from ruptures.show.display import MatplotlibMissingError


@pytest.fixture(scope="module")
def signal_bkps():
    signal, bkps = pw_constant()
    return signal, bkps


def test_display_with_options(signal_bkps):
    try:
        signal, bkps = signal_bkps
        fig, axarr = display(signal, bkps)
        fig, axarr = display(signal, bkps, bkps)
        figsize = (20, 10)  # figure size
        fig, axarr = display(
            signal,
            bkps,
            figsize=figsize,
        )
        fig, axarr = display(
            signal[:, 0],
            bkps,
            figsize=figsize,
        )
    except MatplotlibMissingError:
        pytest.skip("matplotlib is not installed")


def test_display_without_options(signal_bkps):
    try:
        signal, bkps = signal_bkps
        fig, axarr = display(signal, bkps)
        fig, axarr = display(signal, bkps, bkps)
        figsize = (20, 10)  # figure size
        fig, axarr = display(signal, bkps)
        fig, axarr = display(signal[:, 0], bkps)
    except MatplotlibMissingError:
        pytest.skip("matplotlib is not installed")


def test_display_with_new_options(signal_bkps):
    try:
        signal, bkps = signal_bkps
        fig, axarr = display(signal, bkps)
        fig, axarr = display(signal, bkps, bkps)

        fig, axarr = display(signal, bkps, facecolor="k", edgecolor="b")
        fig, axarr = display(signal[:, 0], bkps, facecolor="k", edgecolor="b")
    except MatplotlibMissingError:
        pytest.skip("matplotlib is not installed")
