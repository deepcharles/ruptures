r"""

.. _sec-display:

Display
====================================================================================================

Description
----------------------------------------------------------------------------------------------------

The function :func:`display` displays a signal and the change points provided in alternating colors.
If another set of change point indexes is provided, they are displayed with dashed vertical dashed lines.

Usage
----------------------------------------------------------------------------------------------------

Start with the usual imports and create a signal.

.. code-block:: python

    import numpy as np
    import matplotlib.pylab as plt
    import ruptures as rpt
    # creation of data
    n, dim = 500, 2  # number of samples, dimension
    n_bkps, sigma = 3, 5  # number of change points, noise standart deviation
    signal, bkps = rpt.pw_constant(n, dim, n_bkps, noise_std=sigma)
    rpt.display(signal, bkps)

If we computed another set of change points, for instance ``[110, 150, 320, 500]``, we can easily compare the two segmentations.

.. code-block:: python

    rpt.display(signal, bkps, [110, 150, 320, 500])

.. figure:: /images/example-display.png
    :scale: 50 %

    Example output of the function :func:`display`.

Code explanation
----------------------------------------------------------------------------------------------------

.. autofunction:: ruptures.show.display.display

"""

from itertools import cycle

import numpy as np

try:
    import matplotlib.pyplot as plt
    matplotlib_is_installed = True
except ImportError:
    matplotlib_is_installed = False

from ruptures.utils import pairwise

COLOR_CYCLE = ["#4286f4", "#f44174"]


def display(signal, true_chg_pts, computed_chg_pts=None, **kwargs):
    """
    Display a signal and the change points provided in alternating colors. If another set of change
    point is provided, they are displayed with dashed vertical dashed lines.

    Args:
        signal (array): signal array, shape (n_samples,) or (n_samples, n_features).
        true_chg_pts (list): list of change point indexes.
        computed_chg_pts (list, optional): list of change point indexes.

    Returns:
        tuple: (figure, axarr) with a :class:`matplotlib.figure.Figure` object and an array of Axes objects.

    """
    if not matplotlib_is_installed:
        raise RuntimeError(
            'This features requires the optional dependency matpotlib, you can install it using `pip install matplotlib`.')
    if type(signal) != np.ndarray:
        # Try to get array from Pandas dataframe
        signal = signal.values

    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)
    n_samples, n_features = signal.shape
    # let's set all options
    figsize = (10, 2 * n_features)  # figure size
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
