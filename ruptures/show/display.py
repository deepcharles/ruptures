"""Display module"""
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np

from ruptures.utils import pairwise
COLOR_CYCLE = ["r", "c", "m", "y", "k", "b", "g"]


def display(signal, true_chg_pts, computed_chg_pts=None, **kwargs):
    """
    Display a signal and the change points provided.
    """
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)
    n_samples, n_features = signal.shape
    # let's set all options
    figsize = (20, 10 * n_features)  # figure size
    alpha = 0.2  # transparency of the colored background
    color = "k"  # color of the lines indicating the computed_chg_pts
    linewidth = 3   # linewidth of the lines indicating the computed_chg_pts
    linestyle = "--"   # linestyle of the lines indicating the computed_chg_pts

    if "figsize" in kwargs:
        figsize = kwargs["figsize"]
    if "alpha" in kwargs:
        alpha = kwargs["alpha"]
    if "color" in kwargs:
        color = kwargs["color"]
    if "linewidth" in kwargs:
        linewidth = kwargs["linewidth"]
    if "linestyle" in kwargs:
        linestyle = kwargs["linestyle"]

    fig, axarr = plt.subplots(n_features, figsize=figsize, sharex=True)
    if n_features == 1:
        axarr = [axarr]

    for axe, sig in zip(axarr, signal.T):
        color_cycle = cycle(COLOR_CYCLE)
        # plot s
        axe.plot(range(n_samples), sig)

        # color each (true) regime
        bkps = [0] + sorted(true_chg_pts)

        for (start, end), col in zip(pairwise(bkps), color_cycle):
            axe.axvspan(max(0, start - 0.5),
                        end - 0.5,
                        facecolor=col, alpha=alpha)

        # vertical lines to mark the computed_chg_pts
        if computed_chg_pts is not None:
            for bkp in computed_chg_pts:
                if bkp != 0 and bkp < n_samples:
                    axe.axvline(x=bkp - 0.5,
                                color=color,
                                linewidth=linewidth,
                                linestyle=linestyle)

    fig.tight_layout()

    return fig, axarr
