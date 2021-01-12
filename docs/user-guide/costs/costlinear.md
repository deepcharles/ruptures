# Linear model change (`CostLinear`)

## Description

Let $0 < t_1 < t_2 < \dots < n$ be unknown change points indexes.
Consider the following multiple linear regression model

$$
y_t = x_t' \delta_j + \varepsilon_t, \quad \forall t=t_j,\dots,t_{j+1}-1
$$

for $j>1$.
Here, the observed dependant variable is $y_t\in\mathbb{R}$, the covariate vector is $x_t \in\mathbb{R}^p$, the disturbance is $\varepsilon_t\in\mathbb{R}$.
The vectors $\delta_j\in\mathbb{R}^p$ are the parameter vectors (or regression coefficients).

The least-squares estimates of the break dates is obtained by minimizing the sum of squared residuals [[Bai2003]](#Bai2003).
Formally, the associated cost function on an interval $I$ is

$$
c(y_{I}) = \min_{\delta\in\mathbb{R}^p} \sum_{t\in I} \|y_t - \delta' x_t \|_2^2.
$$

## Usage

Start with the usual imports and create a signal with piecewise linear trends.

```python
import numpy as np
import matplotlib.pylab as plt
import ruptures as rpt

# creation of data
n, n_reg = 2000, 3  # number of samples, number of regressors (including intercept)
n_bkps = 3  # number of change points
# regressors
tt = np.linspace(0, 10 * np.pi, n)
X = np.vstack((np.sin(tt), np.sin(5 * tt), np.ones(n))).T
# parameter vectors
deltas, bkps = rpt.pw_constant(n, n_reg, n_bkps, noise_std=None, delta=(1, 3))
# observed signal
y = np.sum(X * deltas, axis=1)
y += np.random.normal(size=y.shape)
# display signal
rpt.show.display(y, bkps, figsize=(10, 6))
plt.show()
```

Then create a [`CostLinear`][ruptures.costs.costlinear.CostLinear] instance and print the cost of the sub-signal `signal[50:150]`.

```python
# stack observed signal and regressors.
# first dimension is the observed signal.
signal = np.column_stack((y.reshape(-1, 1), X))
c = rpt.costs.CostLinear().fit(signal)
print(c.error(50, 150))
```

You can also compute the sum of costs for a given list of change points.

```python
print(c.sum_of_costs(bkps))
print(c.sum_of_costs([10, 100, 200, 250, n]))
```

In order to use this cost class in a change point detection algorithm (inheriting from [`BaseEstimator`][ruptures.base.BaseEstimator]), either pass a [`CostLinear`][ruptures.costs.costlinear.CostLinear] instance (through the argument `custom_cost`) or set `model="linear"`.

```python
c = rpt.costs.CostLinear()
algo = rpt.Dynp(custom_cost=c)
# is equivalent to
algo = rpt.Dynp(model="linear")
```

## References

<a id="Bai2003">[Bai2003]</a>
J. Bai and P. Perron. Critical values for multiple structural change tests. Econometrics Journal, 6(1):72–78, 2003.
