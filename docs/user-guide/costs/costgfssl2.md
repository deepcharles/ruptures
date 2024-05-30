# Graph Fourier Scan Statistic least squared deviation (`CostGFSSL2`)

## Description

This cost function is specifically designed to detect localized mean-shifts in a graph signal. It relies on the least squared deviation of a graph signal filtered with a low-pass graph spectral filter $h$.

Formally, let $G = (V, E, W)$ a graph containing $p =|V|$ nodes, with a weighted adjacency matrix $W$. We define its combinatorial Laplacian matrix $L$ as classically done in Graph Signal Processing [[Shuman2013](#Shuman2013)]:

$$
L = D - W = U \Lambda U^T,
$$

where

- $D$ is the diagional degree matrix of the graph: $D = \text{diag}(d_1, \ldots, d_p)$.
- $U$ is the orthogonal matrix whose columns $\{u_i\}_{i=1}^p$ are the eigenvectors of $L$
- $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_p)$ contains the eigenvalues of $L$.

Let $\{y_t\}_t, ~ y_t \in \mathbb{R}^p$, a multivariate signal on an interval $I$. The Graph Fourier Scan Statistic (GFSS) [[Sharpnack2016](#Sharpnack2016)] of a graph signal $y_t$ is $\|G(y_t)\|_2^2$ with:

$$
G(y) = \sum_{i=2}^p h(\lambda_i) (u_i^T y) u_i \quad \text{ and } \quad h(\lambda) = \min \left( 1, \sqrt{\frac{\rho}{\lambda}} \right)
$$

where $\rho$ is the so-called cut-sparsity. The cost function [`CostGFSSL2`][ruptures.costs.costgfssl2.CostGFSSL2] over interval $I$ is given by:

$$
c(y_{I}) = \sum_{t\in I} \|G(y_t - \bar{y})\|_2^2
$$

where $\bar{y}$ is the mean of $\{y_t\}_{t\in I}$. Note that $G: \mathbb{R}^p \rightarrow \mathbb{R}^p$ is linear.

## Usage

Start with the usual imports and create a graph signal. Note that the signal dimension must equal the number of nodes of the underlying graph.

*Note*: the below graph and signal are meaningless. For relevant use-cases, see [Graph signal change point detection](../../examples/merging-cost-functions.ipynb).

```python
import numpy as np
import networkx as nx
import matplotlib.pylab as plt
import ruptures as rpt

# creation of the graph
nb_nodes = 30
G = nx.gnp_random_graph(n=nb_nodes, p=0.5)

# creation of the signal
n, dim = 500, nb_nodes  # number of samples, dimension
n_bkps, sigma = 3, 1  # number of change points, noise standard deviation
signal, bkps = rpt.pw_constant(n, dim, n_bkps, noise_std=sigma)
```

Then create a [`CostGFSSL2`][ruptures.costs.costgfssl2.CostGFSSL2] instance and print the cost of the sub-signal `signal[50:150]`. The initialization of the class requires the Laplacian matrix of the underlying graph and the value of the cut-sparsity $\rho$ should be set based on the eigenvalues of $L$ (see [Graph signal change point detection](../../examples/merging-cost-functions.ipynb)).

```python
# creation of the cost function instance
rho = 1  # the cut-sparsity
c = rpt.costs.CostGFSSL2(nx.laplacian_matrix(G).toarray(), rho)
c.fit(signal)
print(c.error(50, 150))
```

You can also compute the sum of costs for a given list of change points.

```python
print(c.sum_of_costs(bkps))
print(c.sum_of_costs([10, 100, 200, 250, n]))
```

In order to use this cost class in a change point detection algorithm (inheriting from [`BaseEstimator`][ruptures.base.BaseEstimator]), pass a [`CostGFSSL2`][ruptures.costs.costgfssl2.CostGFSSL2] instance (through the argument `custom_cost`).

```python
c = rpt.costs.CostL2()
algo = rpt.Dynp(custom_cost=c)
```

## References

<a id="Sharpnack2016">[Sharpnack2016]</a>
Sharpnack, J., Rinaldo, A., and Singh, A. (2016). Detecting Anomalous Activity on Networks With the Graph Fourier Scan Statistic. EEE Transactions on Signal Processing, 64(2):364–379.

<a id="Shuman2013">[Shuman2013]</a>
Shuman, D. I., Narang, S. K., Frossard, P., Ortega, A., and Vandergheynst, P. (2013). The emerging field of signal processing on graphs: Extending high-dimensional data analysis to networks and other irregular domains. EEE Signal Processing Magazine, 30(3):83–98.
