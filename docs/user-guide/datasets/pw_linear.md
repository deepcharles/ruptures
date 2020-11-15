# Piecewise linear (`pw_linear`)

## Description

This function [`pw_linear`][ruptures.datasets.pw_linear.pw_linear] simulates a piecewise linear model (see [Cost linear](../costs/costlinear.md)).
The covariates are standard Gaussian random variables.
The response variable is a (piecewise) linear combination of the covariates.

## Usage

Start with the usual imports and create a signal.

```python
import numpy as np
import matplotlib.pylab as plt
import ruptures as rpt

# creation of data
n, dim = 500, 3  # number of samples, dimension of the covariates
n_bkps, sigma = 3, 5  # number of change points, noise standart deviation
signal, bkps = rpt.pw_linear(n, dim, n_bkps, noise_std=sigma)
rpt.display(signal, bkps)
```