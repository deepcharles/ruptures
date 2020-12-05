# Kernelized mean change (`CostCosine`)

## Description

Given a positive semi-definite kernel $k(\cdot, \cdot) : \mathbb{R}^d\times \mathbb{R}^d \mapsto \mathbb{R}$ and its associated feature map $\Phi:\mathbb{R}^d \mapsto \mathcal{H}$ (where $\mathcal{H}$ is an appropriate Hilbert space), this cost function detects changes in the mean of the embedded signal $\{\Phi(y_t)\}_t$ [[Arlot2019](#Arlot2019)].
Formally, for a signal $\{y_t\}_t$ on an interval $I$,

$$
c(y_{a..b}) = \sum_{t=a}^{b-1} \| \Phi(y_t) - \bar{\mu} \|_{\mathcal{H}}^2
$$

where $\bar{\mu}_{a..b}$ is the empirical mean of the embedded sub-signal $\{\Phi(y_t)\}_{a\leq t < b-1}$.
Here the kernel is the cosine similarity:

$$
k(x, y) = \frac{\langle x\mid y\rangle}{\|x\|\|y\|}
$$

where $\langle \cdot\mid\cdot \rangle$ and $\| \cdot \|$ are the Euclidean scalar product and norm respectively.
In other words, it is equal to the L2-normalized dot product of vectors.
This cost function has been used for music segmentation tasks [[Cooper2002](#Cooper2002)] and topic segmentation of text [[Hearst1994](#Hearst1994)].

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

Then create a [`CostCosine`][ruptures.costs.costcosine.CostCosine] instance and print the cost of the sub-signal `signal[50:150]`.

```python
c = rpt.costs.CostCosine().fit(signal)
print(c.error(50, 150))
```

You can also compute the sum of costs for a given list of change points.

```python
print(c.sum_of_costs(bkps))
print(c.sum_of_costs([10, 100, 200, 250, n]))
```

In order to use this cost class in a change point detection algorithm (inheriting from [`BaseEstimator`][ruptures.base.BaseEstimator]), either pass a [`CostCosine`][ruptures.costs.costcosine.CostCosine] instance (through the argument `custom_cost`) or set `model="cosine"`.

```python
c = rpt.costs.CostCosine()
algo = rpt.Dynp(custom_cost=c)
# is equivalent to
algo = rpt.Dynp(model="cosine")
```

## References

<a id="Hearst1994">[Hearst1994]</a>
Hearst, M. A. (1994). Multi-paragraph segmentation of expository text. In Proceedings of the Annual Meeting of the Association for Computational Linguistics (pp. 9–16). Las Cruces, New Mexico, USA.

<a id="Cooper2002">[Cooper2002]</a>
Cooper, M., & Foote, J. (2002). Automatic music summarization via similarity analysis. In Proceedings of the International Conference on Music Information Retrieval (ISMIR) (pp. 81–85). Paris, France.

<a id="Arlot2019">[Arlot2019]</a>
Arlot, S., Celisse, A., & Harchaoui, Z. (2019). A kernel multiple change-point algorithm via model selection. Journal of Machine Learning Research, 20(162), 1–56.
