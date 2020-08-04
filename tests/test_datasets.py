from itertools import product

import numpy as np
import pytest

from ruptures.datasets import pw_constant, pw_linear, pw_normal, pw_wavy


@pytest.mark.parametrize("func", [pw_constant, pw_linear, pw_normal, pw_wavy])
def test_empty_arg(func):
    func()


@pytest.mark.parametrize(
    "func, n_samples, n_features, n_bkps, noise_std",
    product([pw_constant], range(20, 1000, 200), range(1, 4), [2, 5, 3], [None, 1, 2]),
)
def test_constant(func, n_samples, n_features, n_bkps, noise_std):
    signal, bkps = func(
        n_samples=n_samples, n_features=n_features, n_bkps=n_bkps, noise_std=noise_std
    )
    assert signal.shape == (n_samples, n_features)
    assert len(bkps) == n_bkps + 1
    assert bkps[-1] == n_samples


@pytest.mark.parametrize(
    "func, n_samples, n_features, n_bkps, noise_std",
    product([pw_linear], range(20, 1000, 200), range(1, 4), [2, 5, 3], [None, 1, 2]),
)
def test_linear(func, n_samples, n_features, n_bkps, noise_std):
    signal, bkps = func(
        n_samples=n_samples, n_features=n_features, n_bkps=n_bkps, noise_std=noise_std
    )
    assert signal.shape == (n_samples, n_features + 1)
    assert len(bkps) == n_bkps + 1
    assert bkps[-1] == n_samples


@pytest.mark.parametrize(
    "func, n_samples, n_bkps, noise_std",
    product([pw_wavy], range(20, 1000, 200), [2, 5, 3], [None, 1, 2]),
)
def test_wavy(func, n_samples, n_bkps, noise_std):
    signal, bkps = func(n_samples=n_samples, n_bkps=n_bkps, noise_std=noise_std)
    assert signal.shape == (n_samples,)
    assert len(bkps) == n_bkps + 1
    assert bkps[-1] == n_samples


@pytest.mark.parametrize(
    "func, n_samples, n_bkps", product([pw_normal], range(20, 1000, 200), [2, 5, 3])
)
def test_normal(func, n_samples, n_bkps):
    signal, bkps = func(n_samples=n_samples, n_bkps=n_bkps)
    assert signal.shape == (n_samples, 2)
    assert len(bkps) == n_bkps + 1
    assert bkps[-1] == n_samples
