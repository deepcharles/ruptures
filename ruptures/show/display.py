import matplotlib.pyplot as plt
import numpy as np


def display(signal, true_chg_pts, computed_chg_pts=None, **kwargs):
    """
    Display a signal and the change points provided.
    """
    # let's set all options
    figsize = (20, 10)  # figure size
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

    fig, ax = plt.subplots(figsize=figsize)
    color_cycle = ax._get_lines.prop_cycler
    # plot signal
    ax.plot(range(signal.shape[0]), signal)

    # color each (true) regime
    ends = sorted(true_chg_pts)
    starts = np.append(0, [t + 1 for t in ends[:-1]])

    for (s, e), c in zip(zip(starts, ends), color_cycle):
        ax.axvspan(s, e, facecolor=c["color"], alpha=alpha)

    # vertical lines to mark the computed_chg_pts
    if computed_chg_pts is not None:
        starts = np.sort(computed_chg_pts)
        for s in starts:
            if s != 0:  # no need to put a vertical line at x=0
                ax.axvline(x=s, color=color, linewidth=linewidth,
                           linestyle=linestyle)

    return fig, ax
