# Piecewise constant (`pw_constant`)

## Description

For a given number of samples $T$, number $K$ of change points and noise variance $\sigma^2$, the function [`pw_constant`][ruptures.datasets.pw_constant.pw_constant] generates change point dexes $0 < t_1 < \dots < t_K < T$ and a piecewise constant signal $\{y_t\}_t$ with additive Gaussian noise.

## Usage

Start with the usual imports and create a signal.

```python
import numpy as np
import matplotlib.pylab as plt
import ruptures as rpt

# creation of data
n, dim = 500, 3  # number of samples, dimension
n_bkps, sigma = 3, 5  # number of change points, noise standard deviation
signal, bkps = rpt.pw_constant(n, dim, n_bkps, noise_std=sigma)
rpt.display(signal, bkps)
```

The mean shift amplitude is uniformly drawn from an interval that can be changed through the keyword `delta`.

```python
signal, bkps = rpt.pw_constant(n, dim, n_bkps, noise_std=sigma, delta=(1, 10))
```
