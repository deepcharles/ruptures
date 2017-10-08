"""
.. _sec-pw-constant:

Mean shift
====================================================================================================

Description
----------------------------------------------------------------------------------------------------

For a given number of samples :math:`T`, number of changepoints :math:`K` and noise variance :math:`\sigma^2`, this function generates change point indexes :math:`0<t_1<\dots<t_K<T` and a piecewise constant signal :math:`\{y_t\}_t` with additive Gaussian noise.


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
    signal, bkps = rpt.pw_constant(n, dim, n_bkps, noise_std=sigma)
    rpt.display(signal, bkps)

The mean shift amplitude is uniformly drawn from an interval that can be changed through the keyword ``'delta'``.

.. code-block:: python

    signal, bkps = rpt.pw_constant(n, dim, n_bkps, noise_std=sigma, delta=(1, 10))


Code explanation
----------------------------------------------------------------------------------------------------

.. autofunction:: ruptures.datasets.pw_constant.pw_constant

"""

import numpy as np
from numpy import random as rd

from ruptures.utils import draw_bkps


def pw_constant(n_samples=200, n_features=1, n_bkps=3, noise_std=None,
                delta=(1, 10)):
    """Return a piecewise constant signal and the associated changepoints.

    Args:
        n_samples (int): signal length
        n_features (int, optional): number of dimensions
        n_bkps (int, optional): number of changepoints
        noise_std (float, optional): noise std. If None, no noise is added
        delta (tuple, optional): (delta_min, delta_max) max and min jump values

    Returns:
        tuple: signal of shape (n_samples, n_features), list of breakpoints

    """
    # breakpoints
    bkps = draw_bkps(n_samples, n_bkps)
    # we create the signal
    signal = np.empty((n_samples, n_features), dtype=float)
    tt_ = np.arange(n_samples)
    delta_min, delta_max = delta
    # mean value
    center = np.zeros(n_features)
    for ind in np.split(tt_, bkps):
        if ind.size > 0:
            # jump value
            jump = rd.uniform(delta_min, delta_max, size=n_features)
            spin = rd.choice([-1, 1], n_features)
            center += jump * spin
            signal[ind] = center

    if noise_std is not None:
        noise = rd.normal(size=signal.shape) * noise_std
        signal = signal + noise

    return signal, bkps
