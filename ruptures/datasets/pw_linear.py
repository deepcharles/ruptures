r"""
.. _sec-pw-linear:

Shift in linear model
====================================================================================================

Description
----------------------------------------------------------------------------------------------------

This function simulates a piecewise linear model (see :ref:`sec-linear`).
The covariates standard Gaussian random variables.
The response variable is a (piecewise) linear combination of the covariates.

Usage
----------------------------------------------------------------------------------------------------

Start with the usual imports and create a signal.

.. code-block:: python

    import numpy as np
    import matplotlib.pylab as plt
    import ruptures as rpt
    # creation of data
    n, dim = 500, 3  # number of samples, dimension of the covariates
    n_bkps, sigma = 3, 5  # number of change points, noise standart deviation
    signal, bkps = rpt.pw_linear(n, dim, n_bkps, noise_std=sigma)
    rpt.display(signal, bkps)

Code explanation
----------------------------------------------------------------------------------------------------

.. autofunction:: ruptures.datasets.pw_linear.pw_linear



"""
import numpy as np
from numpy.random import normal

from . import pw_constant
from ruptures.utils import draw_bkps


def pw_linear(n_samples=200, n_features=1, n_bkps=3, noise_std=None):
    """
    Return piecewise linear signal and the associated changepoints.

    Args:
        n_samples (int, optional): signal length
        n_features (int, optional): number of covariates
        n_bkps (int, optional): number of change points
        noise_std (float, optional): noise std. If None, no noise is added
    Returns:
        tuple: signal of shape (n_samples, n_features+1), list of breakpoints
    """

    covar = normal(size=(n_samples, n_features))
    linear_coeff, bkps = pw_constant(n_samples=n_samples,
                                     n_bkps=n_bkps,
                                     n_features=n_features,
                                     noise_std=None)
    var = np.sum(linear_coeff * covar, axis=1)
    if noise_std is not None:
        var += normal(scale=noise_std, size=var.shape)
    signal = np.c_[var, covar]
    return signal, bkps
