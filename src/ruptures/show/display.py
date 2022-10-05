r"""Display."""

from itertools import cycle

import numpy as np

from ruptures.utils import pairwise

COLOR_CYCLE = ["#4286f4", "#f44174"]


class MatplotlibMissingError(RuntimeError):
    pass


def display(
    signal,
    true_chg_pts,
    computed_chg_pts=None,
    computed_chg_pts_color="k",
    computed_chg_pts_linewidth=3,
    computed_chg_pts_linestyle="--",
    computed_chg_pts_alpha=1.0,
    **kwargs
):
    """Display a signal and the change points provided in alternating colors.
    If another set of change point is provided, they are displayed with dashed
    vertical dashed lines. The following matplotlib subplots options is set by
    default, but can be changed when calling `display`):

    - figure size `figsize`, defaults to `(10, 2 * n_features)`.

    Args:
        signal (array): signal array, shape (n_samples,) or (n_samples, n_features).
        true_chg_pts (list): list of change point indexes.
        computed_chg_pts (list, optional): list of change point indexes.
        computed_chg_pts_color (str, optional): color of the lines indicating
            the computed_chg_pts. Defaults to "k".
        computed_chg_pts_linewidth (int, optional): linewidth of the lines
            indicating the computed_chg_pts. Defaults to 3.
        computed_chg_pts_linestyle (str, optional): linestyle of the lines
            indicating the computed_chg_pts. Defaults to "--".
        computed_chg_pts_alpha (float, optional): alpha of the lines indicating
            the computed_chg_pts. Defaults to "1.0".
        **kwargs : all additional keyword arguments are passed to the plt.subplots call.

    Returns:
        tuple: (figure, axarr) with a :class:`matplotlib.figure.Figure` object and an array of Axes objects.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise MatplotlibMissingError(
            "This feature requires the optional dependency matpotlib, you can install it using `pip install matplotlib`."
        )

    if type(signal) != np.ndarray:
        # Try to get array from Pandas dataframe
        signal = signal.values

    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)
    n_samples, n_features = signal.shape

    # let's set a sensible defaut size for the subplots
    matplotlib_options = {
        "figsize": (10, 2 * n_features),  # figure size
    }
    # add/update the options given by the user
    matplotlib_options.update(kwargs)

    # create plots
    fig, axarr = plt.subplots(n_features, sharex=True, **matplotlib_options)
    if n_features == 1:
        axarr = [axarr]

    for axe, sig in zip(axarr, signal.T):
        color_cycle = cycle(COLOR_CYCLE)
        # plot s
        axe.plot(range(n_samples), sig)

        # color each (true) regime
        bkps = [0] + sorted(true_chg_pts)
        alpha = 0.2  # transparency of the colored background

        for (start, end), col in zip(pairwise(bkps), color_cycle):
            axe.axvspan(max(0, start - 0.5), end - 0.5, facecolor=col, alpha=alpha)
        # vertical lines to mark the computed_chg_pts
        if computed_chg_pts is not None:
            for bkp in computed_chg_pts:
                if bkp != 0 and bkp < n_samples:
                    axe.axvline(
                        x=bkp - 0.5,
                        color=computed_chg_pts_color,
                        linewidth=computed_chg_pts_linewidth,
                        linestyle=computed_chg_pts_linestyle,
                        alpha=computed_chg_pts_alpha,
                    )

    fig.tight_layout()

    return fig, axarr
