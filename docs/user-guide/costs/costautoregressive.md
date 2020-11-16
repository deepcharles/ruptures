# Autoregressive model change (`CostAR`)

## Description

Let $0<t_1<t_2<\dots<n$ be unknown change points indexes.
Consider the following piecewise autoregressive model

$$
    y_t = z_t' \delta_j + \varepsilon_t, \quad \forall t=t_j,\dots,t_{j+1}-1
$$

where $j>1$ is the segment number, $z_t=[y_{t-1}, y_{t-2},\dots,y_{t-p}]$ is the lag vector,and $p>0$ is the order of the process.

The least-squares estimates of the break dates is obtained by minimizing the sum of squared
residuals [[Bai2000]](#Bai2000).
Formally, the associated cost function on an interval $I$ is

$$
c(y_{I}) = \min_{\delta\in\mathbb{R}^p} \sum_{t\in I} \|y_t - \delta' z_t \|_2^2.
$$

Currently, this function is limited to 1D signals.

## Usage

Start with the usual imports and create a signal with piecewise linear trends.

```python
from itertools import cycle
import numpy as np
import matplotlib.pylab as plt
import ruptures as rpt

# creation of data
n = 2000
n_bkps, sigma = 4, 0.5  # number of change points, noise standart deviation
bkps = [400, 1000, 1300, 1800, n]
f1 = np.array([0.075, 0.1])
f2 = np.array([0.1, 0.125])
freqs = np.zeros((n, 2))
for sub, val in zip(np.split(freqs, bkps[:-1]), cycle([f1, f2])):
    sub += val
tt = np.arange(n)
signal = np.sum((np.sin(2 * np.pi * tt * f) for f in freqs.T))
signal += np.random.normal(scale=sigma, size=signal.shape)
# display signal
rpt.show.display(signal, bkps, figsize=(10, 6))
plt.show()
```

Then create a [CostAR][ruptures.costs.costautoregressive.CostAR] instance and print the cost of the sub-signal
`signal[50:150]`.
The autoregressive order can be specified through the keyword ``'order'``.

```python
c = rpt.costs.CostAR(order=10).fit(signal)
print(c.error(50, 150))
```

You can also compute the sum of costs for a given list of change points.

```python
print(c.sum_of_costs(bkps))
print(c.sum_of_costs([10, 100, 200, 250, n]))
```

In order to use this cost class in a change point detection algorithm (inheriting from
[BaseEstimator][ruptures.base.BaseEstimator]), either pass a [CostAR][ruptures.costs.costautoregressive.CostAR] instance (through the argument
``'custom_cost'``) or set `model="ar"`.
Additional parameters can be passed to the cost instance through the keyword ``'params'``.

```python
c = rpt.costs.CostAR(order=10)
algo = rpt.Dynp(custom_cost=c)
# is equivalent to
algo = rpt.Dynp(model="ar", params={"order": 10})
```

## Reference

<a id="Bai2000">[Bai2000]</a>
Bai, J. (2000). Vector autoregressive models with structural changes in regression coefficients and in variance–covariance matrices. Annals of Economics and Finance, 1(2), 301–336.