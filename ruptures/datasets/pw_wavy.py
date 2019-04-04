"""
.. _sec-pw-wavy:

Shift in frequency (sine waves)
====================================================================================================

Description
----------------------------------------------------------------------------------------------------

This function simulates a sum-of-sine signal :math:`y_t=\sin(2\pi f_1 t)+\sin(2\pi f_2 t)` where :math:`t=0,\dots,T-1`.
The frequency vector :math:`[f_1, f_2]` alternates between :math:`[0.075, 0.1]` and :math:`[0.1, 0.125]` at each change point index.
Gaussian white noise can be added to the signal.

.. figure:: /images/sum_of_sines.png
   :scale: 50 %
   :alt: Signal example

   Top: signal example. Bottom: associated spectrogram.

Usage
----------------------------------------------------------------------------------------------------

Start with the usual imports and create a signal.

.. code-block:: python

    import numpy as np
    import matplotlib.pylab as plt
    import ruptures as rpt
    # creation of data
    n, dim = 500, 3  # number of samples, dimension
    n_bkps, sigma = 3, 5  # number of change points, noise standart deviation
    signal, bkps = rpt.pw_wavy(n, n_bkps, noise_std=sigma)
    rpt.display(signal, bkps)


Code explanation
----------------------------------------------------------------------------------------------------

.. autofunction:: ruptures.datasets.pw_wavy.pw_wavy

"""

from itertools import cycle

import numpy as np
from numpy.random import normal

from ruptures.utils import draw_bkps


def pw_wavy(n_samples=200, n_bkps=3, noise_std=None):
    """Return a 1D piecewise wavy signal and the associated changepoints.

    Args:
        n_samples (int, optional): signal length
        n_bkps (int, optional): number of changepoints
        noise_std (float, optional): noise std. If None, no noise is added

    Returns:
        tuple: signal of shape (n_samples, 1), list of breakpoints

    """
    # breakpoints
    bkps = draw_bkps(n_samples, n_bkps)
    # we create the signal
    f1 = np.array([0.075, 0.1])
    f2 = np.array([0.1, 0.125])
    freqs = np.zeros((n_samples, 2))
    for sub, val in zip(np.split(freqs, bkps[:-1]), cycle([f1, f2])):
        sub += val
    tt = np.arange(n_samples)

    # DeprecationWarning: Calling np.sum(generator) is deprecated
    # Use np.sum(np.from_iter(generator)) or the python sum builtin instead.
    signal = np.sum([np.sin(2 * np.pi * tt * f) for f in freqs.T], axis=0)

    if noise_std is not None:
        noise = normal(scale=noise_std, size=signal.shape)
        signal += noise

    return signal, bkps
