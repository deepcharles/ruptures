# Kernelized mean change (`CostRbf`)

## Description

Given a positive semi-definite kernel $k(\cdot, \cdot) : \mathbb{R}^d\times \mathbb{R}^d \mapsto \mathbb{R}$ and its associated feature map $\Phi:\mathbb{R}^d \mapsto \mathcal{H}$ (where $\mathcal{H}$ is an appropriate Hilbert space), this cost function detects changes in the mean of the embedded signal $\{\Phi(y_t)\}_t$ [[Garreau2018](#Garreau2018), [Arlot2019](#Arlot2019)].
Formally, for a signal $\{y_t\}_t$ on an interval $I$,

$$
c(y_{I}) = \sum_{t\in I} \| \Phi(y_t) - \bar{\mu} \|_{\mathcal{H}}^2
$$

where $\bar{\mu}$ is the empirical mean of the embedded sub-signal $\{\Phi(y_t)\}_{t\in I}$.
Here the kernel is the radial basis function (rbf):

$$
k(x, y) = \exp(-\gamma \| x - y \|^2 )
$$

where $\| \cdot \|$ is the Euclidean norm and $\gamma>0$ is the so-called bandwidth parameter and is determined according to median heuristics (i.e. equal to the inverse of median of all pairwise distances).

In a nutshell, this cost function is able to detect changes in the distribution of an iid sequence of random variables.
Because it is non-parametric, it is performs reasonably well on a wide range of tasks.

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

Then create a [`CostRbf`][ruptures.costs.costrbf.CostRbf] instance and print the cost of the sub-signal `signal[50:150]`.

```python
c = rpt.costs.CostRbf().fit(signal)
print(c.error(50, 150))
```

You can also compute the sum of costs for a given list of change points.

```python
print(c.sum_of_costs(bkps))
print(c.sum_of_costs([10, 100, 200, 250, n]))
```

In order to use this cost class in a change point detection algorithm (inheriting from [`BaseEstimator`][ruptures.base.BaseEstimator]), either pass a [`CostRbf`][ruptures.costs.costrbf.CostRbf] instance (through the argument `custom_cost`) or set `model="rbf"`.

```python
c = rpt.costs.CostRbf()
algo = rpt.Dynp(custom_cost=c)
# is equivalent to
algo = rpt.Dynp(model="rbf")
```

## References

<a id="Garreau2018">[Garreau2018]</a>
Garreau, D., & Arlot, S. (2018). Consistent change-point detection with kernels. Electronic Journal of Statistics, 12(2), 4440–4486.

<a id="Arlot2019">[Arlot2019]</a>
Arlot, S., Celisse, A., & Harchaoui, Z. (2019). A kernel multiple change-point algorithm via model selection. Journal of Machine Learning Research, 20(162), 1–56.
