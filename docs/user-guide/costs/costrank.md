# Rank-based cost function (`CostRank`)

## Description

This cost function detects general distribution changes in multivariate signals, using a rank transformation [[Lung-Yut-Fong2015]](#Lung-Yut-Fong2015).
Formally, for a signal $\{y_t\}_t$ on an interval $[a, b)$,

$$
c_{rank}(a, b) = -(b - a) \bar{r}_{a..b}' \hat{\Sigma}_r^{-1} \bar{r}_{a..b}
$$

where $\bar{r}_{a..b}$ is the empirical mean of the sub-signal $\{r_t\}_{t=a+1}^b$, and $\hat{\Sigma}_r$ is the covariance matrix of the complete rank signal $r$.

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
```

Then create a [`CostRank`][ruptures.costs.costrank.CostRank] instance and print the cost of the sub-signal `signal[50:150]`.

```python
c = rpt.costs.CostRank().fit(signal)
print(c.error(50, 150))
```

You can also compute the sum of costs for a given list of change points.

```python
print(c.sum_of_costs(bkps))
print(c.sum_of_costs([10, 100, 200, 250, n]))
```

In order to use this cost class in a change point detection algorithm (inheriting from [`BaseEstimator`][ruptures.base.BaseEstimator]), either pass a [`CostRank`][ruptures.costs.costrank.CostRank] instance (through the argument `custom_cost`) or set `model="rank"`.

```python
c = rpt.costs.CostRank()
algo = rpt.Dynp(custom_cost=c)
# is equivalent to
algo = rpt.Dynp(model="rank")
```

## References

<a id="Lung-Yut-Fong2015">[Lung-Yut-Fong2015]</a>
Lung-Yut-Fong, A., Lévy-Leduc, C., & Cappé, O. (2015). Homogeneity and change-point detection tests for multivariate data using rank statistics. Journal de La Société Française de Statistique, 156(4), 133–162.