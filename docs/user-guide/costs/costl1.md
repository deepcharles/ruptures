# Least absolute deviation (`CostL1`)

## Description

This cost function detects changes in the median of a signal.
Overall, it is a robust estimator of a shift in the central point (mean, median, mode) of a distribution [[Bai1995]](#Bai1995).
Formally, for a signal $\{y_t\}_t$ on an interval $I$,

$$
c(y_{I}) = \sum_{t\in I} \|y_t - \bar{y}\|_1
$$

where $\bar{y}$ is the componentwise median of $\{y_t\}_{t\in I}$.

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

Then create a [`CostL1`][ruptures.costs.costl1.CostL1] instance and print the cost of the sub-signal `signal[50:150]`.

```python
c = rpt.costs.CostL1().fit(signal)
print(c.error(50, 150))
```

You can also compute the sum of costs for a given list of change points.

```python
print(c.sum_of_costs(bkps))
print(c.sum_of_costs([10, 100, 200, 250, n]))
```

In order to use this cost class in a change point detection algorithm (inheriting from [`BaseEstimator`][ruptures.base.BaseEstimator], either pass a [`CostL1`][ruptures.costs.costl1.CostL1] instance (through the argument `custom_cost`) or set `model="l1"`.

```python
c = rpt.costs.CostL1()
algo = rpt.Dynp(custom_cost=c)
# is equivalent to
algo = rpt.Dynp(model="l1")
```

## References

<a id="Bai1995">[Bai1995]</a>
Bai, J. (1995). Least absolute deviation of a shift. Econometric Theory, 11(3), 403–436.