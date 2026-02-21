r"""Shift in linear model."""

import numpy as np
from typing import Optional
from numpy.typing import NDArray

from . import pw_constant


def pw_linear(
    n_samples: int = 200,
    n_features: int = 1,
    n_bkps: int = 3,
    noise_std: Optional[float] = None,
    seed: Optional[int] = None,
) -> tuple[NDArray[np.floating], list[int]]:
    """Return piecewise linear signal and the associated changepoints.

    Args:
        n_samples (int, optional): signal length
        n_features (int, optional): number of covariates
        n_bkps (int, optional): number of change points
        noise_std (float, optional): noise std. If None, no noise is added
        seed (int): random seed
    Returns:
        tuple: signal of shape (n_samples, n_features+1), list of breakpoints
    """
    rng = np.random.default_rng(seed=seed)
    covar = rng.normal(size=(n_samples, n_features))
    linear_coeff, bkps = pw_constant(
        n_samples=n_samples,
        n_bkps=n_bkps,
        n_features=n_features,
        noise_std=None,
        seed=seed,
    )
    var = np.sum(linear_coeff * covar, axis=1)
    if noise_std is not None:
        var += rng.normal(scale=noise_std, size=var.shape)
    signal = np.c_[var, covar]
    return signal, bkps
