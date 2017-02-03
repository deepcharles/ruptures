import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle
COLOR_CYCLE = ["r", "c", "m", "y", "k", "b", "g"]


def display(signal, true_chg_pts, computed_chg_pts=None, **kwargs):
    """
    Display a signal and the change points provided.
    """
    if signal.ndim == 1:
        s = signal.reshape(-1, 1)
    else:
        s = signal

    # let's set all options
    figsize = (20, 10 * s.shape[1])  # figure size
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

    fig, axarr = plt.subplots(s.shape[1], figsize=figsize, sharex=True)
    if s.shape[1] == 1:
        axarr = [axarr]

    for ax, sig in zip(axarr, s.T):
        color_cycle = cycle(COLOR_CYCLE)
        # plot s
        ax.plot(range(s.shape[0]), sig)

        # color each (true) regime
        starts = [0] + sorted(true_chg_pts[:-1])
        ends = sorted(true_chg_pts)

        for (start, end), c in zip(zip(starts, ends), color_cycle):
            ax.axvspan(start, end, facecolor=c, alpha=alpha)

        # vertical lines to mark the computed_chg_pts
        if computed_chg_pts is not None:
            starts = np.sort(computed_chg_pts)
            for start in starts:
                if start != 0:  # no need to put a vertical line at x=0
                    ax.axvline(x=start, color=color, linewidth=linewidth,
                               linestyle=linestyle)

    fig.tight_layout()

    return fig, axarr
