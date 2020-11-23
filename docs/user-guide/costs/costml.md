# Change detection with a Mahalanobis-type metric (`CostMl`)

## Description

Given a positive semi-definite matrix $M\in\mathbb{R}^{d\times d}$,
this cost function detects changes in the mean of the embedded signal defined by the pseudo-metric

$$
\| x - y \|_M^2 = (x-y)^t M (x-y).
$$

Formally, for a signal $\{y_t\}_t$ on an interval $I$, the cost function is equal to

$$
c(y_{I}) = \sum_{t\in I} \| y_t - \bar{\mu} \|_{M}^2
$$

where $\bar{\mu}$ is the empirical mean of the sub-signal $\{y_t\}_{t\in I}$.
The matrix $M$ can for instance be the result of a similarity learning algorithm [[Xing2003](#Xing2003), [Truong2019](#Truong2019)] or the inverse of the empirical covariance matrix (yielding the Mahalanobis distance).

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

Then create a [`CostMl`][ruptures.costs.costml.CostMl] instance and print the cost of the sub-signal `signal[50:150]`.

```python
M = np.eye(dim)
c = rpt.costs.CostMl(metric=M).fit(signal)
print(c.error(50, 150))
```

You can also compute the sum of costs for a given list of change points.

```python
print(c.sum_of_costs(bkps))
print(c.sum_of_costs([10, 100, 200, 250, n]))
```

In order to use this cost class in a change point detection algorithm (inheriting from [`BaseEstimator`][ruptures.base.BaseEstimator]), either pass a [`CostMl`][ruptures.costs.costml.CostMl] instance (through the argument `custom_cost`) or set `model="mahalanobis"`.

```python
c = rpt.costs.CostMl(metric=M)
algo = rpt.Dynp(custom_cost=c)
# is equivalent to
algo = rpt.Dynp(model="mahalanobis", params={"metric": M})
```

## References

<a id="Xing2003">[Xing2003]</a>
Xing, E. P., Jordan, M. I., & Russell, S. J. (2003). Distance metric learning, with application to clustering with side-Information. Advances in Neural Information Processing Systems (NIPS), 521–528.

<a id="Truong2019">[Truong2019]</a>
Truong, C., Oudre, L., & Vayatis, N. (2019). Supervised kernel change point detection with partial annotations. Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 1–5.