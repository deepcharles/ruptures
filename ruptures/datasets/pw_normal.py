"""
.. _sec-pw-normal:

Shift in correlation
====================================================================================================

Description
----------------------------------------------------------------------------------------------------

This function simulates a 2D signal of Gaussian i.i.d. random variables with zero mean and covariance matrix alternating between :math:`[[1, 0.9], [0.9, 1]]` and :math:`[[1, -0.9], [-0.9, 1]]` at every change point.

.. figure:: /images/correlation_shift.png
   :scale: 50 %
   :alt: Signal example

   Top and middle: 2D signal example. Bottom: Scatter plot for each regime type.

Usage
----------------------------------------------------------------------------------------------------

Start with the usual imports and create a signal.

.. code-block:: python

    import numpy as np
    import matplotlib.pylab as plt
    import ruptures as rpt
    # creation of data
    n = 500, 3  # number of samples
    n_bkps = 3  # number of change points, noise standart deviation
    signal, bkps = rpt.pw_normal(n, n_bkps)
    rpt.display(signal, bkps)

Code explanation
----------------------------------------------------------------------------------------------------

.. autofunction:: ruptures.datasets.pw_normal.pw_normal


"""

from itertools import cycle

import numpy as np
from numpy import random as rd

from ruptures.utils import draw_bkps


def pw_normal(n_samples=200, n_bkps=3):
    """Return a 2D piecewise Gaussian signal and the associated changepoints.

    Args:
        n_samples (int, optional): signal length
        n_bkps (int, optional): number of change points

    Returns:
        tuple: signal of shape (n_samples, 2), list of breakpoints

    """
    # breakpoints
    bkps = draw_bkps(n_samples, n_bkps)
    # we create the signal
    signal = np.zeros((n_samples, 2), dtype=float)
    cov1 = np.array([[1, 0.9], [0.9, 1]])
    cov2 = np.array([[1, -0.9], [-0.9, 1]])
    for sub, cov in zip(np.split(signal, bkps), cycle((cov1, cov2))):
        n_sub, _ = sub.shape
        sub += rd.multivariate_normal([0, 0], cov, size=n_sub)

    return signal, bkps
