r"""Shift in linear model"""
import numpy as np
from numpy.random import normal

from . import pw_constant
from ruptures.utils import draw_bkps


def pw_linear(n_samples=200, n_features=1, n_bkps=3, noise_std=None):
    """Return piecewise linear signal and the associated changepoints.

    Args:
        n_samples (int, optional): signal length
        n_features (int, optional): number of covariates
        n_bkps (int, optional): number of change points
        noise_std (float, optional): noise std. If None, no noise is added
    Returns:
        tuple: signal of shape (n_samples, n_features+1), list of breakpoints
    """

    covar = normal(size=(n_samples, n_features))
    linear_coeff, bkps = pw_constant(
        n_samples=n_samples, n_bkps=n_bkps, n_features=n_features, noise_std=None
    )
    var = np.sum(linear_coeff * covar, axis=1)
    if noise_std is not None:
        var += normal(scale=noise_std, size=var.shape)
    signal = np.c_[var, covar]
    return signal, bkps
