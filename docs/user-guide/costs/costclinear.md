# Continuous linear change (`CostCLinear`)

## Description

For a given set of indexes (also called knots) $t_k$ ($k=1,\dots,K$), a linear spline $f$ is such that:

1. $f$ is affine on each interval $t_k..t_{k+1}$, i.e. $f(t)=\alpha_k (t-t_k) + \beta_k$ ($\alpha_k, \beta_k \in \mathbb{R}^d$) for all $t=t_k,t_k+1,\dots,t_{k+1}-1$;
2. $f$ is continuous.

The cost function [`CostCLinear`][ruptures.costs.costclinear.CostCLinear] measures the error when approximating the signal with a linear spline.
Formally, it is defined for $0<a<b\leq T$ by

$$
c(y_{a..b}) := \sum_{t=a}^{b-1} \left\lVert y_t - y_{a-1} - \frac{t-a+1}{b-a}(y_{b-1}-y_{a-1}) \right\rVert^2
$$

and $c(y_{0..b}):=c(y_{1..b})$ (by convention).

## Usage

Start with the usual imports and create a signal with piecewise linear trends.

```python
import numpy as np
import matplotlib.pylab as plt
import ruptures as rpt

# creation of data
n_samples, n_dims = 500, 3  # number of samples, dimension
n_bkps, sigma = 3, 5  # number of change points, noise standard deviation
signal, bkps = rpt.pw_constant(n_samples, n_dims, n_bkps, noise_std=sigma)
signal = np.cumsum(signal, axis=1)
```

Then create a [`CostCLinear`][ruptures.costs.costclinear.CostCLinear] instance and print the cost of the sub-signal `signal[50:150]`.

```python
c = rpt.costs.CostCLinear().fit(signal)
print(c.error(50, 150))
```

You can also compute the sum of costs for a given list of change points.

```python
print(c.sum_of_costs(bkps))
print(c.sum_of_costs([10, 100, 200, 250, n]))
```

In order to use this cost class in a change point detection algorithm (inheriting from [`BaseEstimator`][ruptures.base.BaseEstimator]), either pass a [`CostCLinear`][ruptures.costs.costclinear.CostCLinear] instance (through the argument `custom_cost`) or set `model="clinear"`.

```python
c = rpt.costs.CostCLinear()
algo = rpt.Dynp(custom_cost=c)
# is equivalent to
algo = rpt.Dynp(model="clinear")
```
