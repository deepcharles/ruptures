# Gaussian process change (`CostNormal`)

## Description

This cost function detects changes in the mean and covariance matrix of a sequence of multivariate Gaussian random variables.
Formally, for a signal $\{y_t\}_t$ on an interval $I$,
$$
c(y_{I}) = |I| \log\det(\widehat{\Sigma}_I + \epsilon\text{Id})
$$
where $\widehat{\Sigma}_I$ is the empirical covariance matrix of the sub-signal $\{y_t\}_{t\in I}$ and $\epsilon>0$ is a small constant added to cope with badly conditioned covariance matrices (new in version 1.1.5, see [Issue 196](https://github.com/deepcharles/ruptures/issues/196)).
It is robust to strongly dependant processes; for more information, see [[Lavielle1999]](#Lavielle1999) (univariate case) and [[Lavielle2006]](#Lavielle2006) (multivariate case).


## Usage

Start with the usual imports and create a signal.

```python
import numpy as np
import matplotlib.pylab as plt
import ruptures as rpt

# creation of data
n, dim = 500, 3  # number of samples, dimension
n_bkps, sigma = 3, 5  # number of change points, noise standart deviation
signal, bkps = rpt.pw_constant(n, dim, n_bkps, noise_std=sigma)
```

Then create a [`CostNormal`][ruptures.costs.costnormal.CostNormal] instance and print the cost of the sub-signal `signal[50:150]`.

```python
c = rpt.costs.CostNormal().fit(signal)
print(c.error(50, 150))
```

You can also compute the sum of costs for a given list of change points.

```python
print(c.sum_of_costs(bkps))
print(c.sum_of_costs([10, 100, 200, 250, n]))
```

In order to use this cost class in a change point detection algorithm (inheriting from [`BaseEstimator`][ruptures.base.BaseEstimator]), either pass a [`CostNormal`][ruptures.costs.costnormal.CostNormal] instance (through the argument `custom_cost`) or set `model="normal"`.

```python
c = rpt.costs.CostNormal()
algo = rpt.Dynp(custom_cost=c)
# is equivalent to
algo = rpt.Dynp(model="normal")
```

To set the small diagonal bias to 0 (default behaviour in versions 1.1.4 and before), simply do the following (change `Dynp` by the search method you need).
```python
c = rpt.costs.CostNormal(add_small_diag=False)
algo = rpt.Dynp(custom_cost=c)
# or, equivalently,
algo = rpt.Dynp(model="normal", params={"add_small_diag": False})
```

## References

<a id="Lavielle1999">[Lavielle1999]</a>
Lavielle, M. (1999). Detection of multiples changes in a sequence of dependant variables. Stochastic Processes and Their Applications, 83(1), 79–102.

<a id="Lavielle2006">[Lavielle2006]</a>
Lavielle, M., & Teyssière, G. (2006). Detection of multiple change-points in multivariate time series. Lithuanian Mathematical Journal, 46(3).