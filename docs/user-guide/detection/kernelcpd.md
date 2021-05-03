# Kernel change point detection

## Problem formulation

In this section, the kernel change point detection setting is briefly described.
The interested reader can refer to [[Celisse2018](#Celisse2018), [Arlot2019](#Arlot2019)] for a more complete introduction.<br>
Let $y = \{y_0,y_1,\dots,y_{T-1}\}$ denote a $\mathbb{R}^d$-valued signal with $T$ samples.
This signal is mapped onto a [reproducing Hilbert space (rkhs)](https://en.wikipedia.org/wiki/Reproducing_kernel_Hilbert_space) $\mathcal{H}$ associated with a user-defined kernel function $k(\cdot, \cdot):\mathbb{R}^d\times\mathbb{R}^d\rightarrow\mathbb{R}$.
The mapping function $\phi:\mathbb{R}^d\rightarrow\mathcal{H}$ onto this rkhs is implicitly defined by $\phi(y_t) = k(y_t, \cdot)\in\mathcal{H}$ resulting in the following inner-product and norm:

$$
\langle\phi(y_s)\mid\phi(y_t)\rangle_{\mathcal{H}} = k(y_s,y_t)
$$

and

$$
\|\phi(y_t)\|_{\mathcal{H}}^2 = k(y_t,y_t)
$$

for any samples $y_s,y_t\in\mathbb{R}^d$.
Kernel change point detection consists in finding mean-shifts in the mapped signal $\phi(y)$ by minimizing $V(\cdot)$ where

$$
V(t_1,\dots,t_K) := \sum_{k=0}^K\sum_{t=t_k}^{t_{k+1}-1} \|\phi(y_t)-\bar{\mu}_{t_k..t_{k+1}}\|^2_{\mathcal{H}}
$$

where $\bar{\mu}_{t_k..t_{k+1}}$ is the empirical mean of the sub-signal $\phi(y_{t_k}), \phi(y_{t_k+1}),\dots,\phi(y_{t_{k+1}-1})$, and $t_1,t_2,\dots,t_K$ are change point indexes, in increasing order.
(By convention $t_0=0$ and $t_{K+1}=T$.)

**If the number of changes is known beforehand**, we solve the following optimization problem, over all possible change positions $t_1<t_2<\dots<t_K$ (where the number $K$ of changes is provided by the user):

$$
\hat{t}_1,\dots,\hat{t}_K := \arg\min_{t_1,\dots,t_K} V(t_1,\dots,t_K).
$$

The exact optimization procedure is described in [[Celisse2018]](#Celisse2018).

**If the number of changes is not known**, we solve the following penalized optimization problem

$$
\hat{K}, \{\hat{t}_1,\dots,\hat{t}_{\hat{K}}\} := \arg\min_{K, \{t_1,\dots, t_K\}} V(t_1,\dots, t_K) + \beta K
$$

where $\beta>0$ is the smoothing parameter (provided by the user) and $\hat{K}$ is the estimated number of change points.
Higher values of $\beta$ produce lower $\hat{K}$.
The exact optimization procedure is described in [[Killick2012]](#Killick2012).

## Available kernels
We list below a number of kernels that are already implemented in `ruptures`.
In the following, $u$ and $v$ are two d-dimensional vectors and $\|\cdot\|$ is the Euclidean norm.

| Kernel                     | Description                                                                                         | Cost function                                        |
| -------------------------- | --------------------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| Linear<br>`model="linear"` | $k_{\text{linear}}(u, v) = u^T v$.                                                                  | [`CostL2`](../../user-guide/costs/costl2.md)         |
| Gaussian<br>`model="rbf"`  | $k_{\text{Gaussian}}(u,v)=\exp(-\gamma \|u-v\|^2)$<br>where $\gamma>0$ is a user-defined parameter. | [`CostRbf`](../../user-guide/costs/costrbf.md)       |
| Cosine<br>`model="cosine"` | $k_{\text{cosine}}(u, v) = (u^T v)/(\|u\|\|v\|)$                                                    | [`CostCosine`](../../user-guide/costs/costcosine.md) |


## Implementation and usage

Kernel change point detection is implemented in the class [`KernelCPD`][ruptures.detection.kernelcpd.KernelCPD], which is a C implementation of dynamic programming and PELT.
To see it in action, please look at the gallery of examples, in particular:

- [Kernel change point detection: a performance comparison](../../examples/kernel-cpd-performance-comparison.ipynb)

The exact class API is available [here][ruptures.detection.kernelcpd.KernelCPD].

## References

<a id="Gretton2012">[Gretton2012]</a>
Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., & Smola, A. (2012). A kernel two-sample test. The Journal of Machine Learning Research, 13, 723–773.

<a id="Killick2012">[Killick2012]</a>
Killick, R., Fearnhead, P., & Eckley, I. (2012). Optimal detection of changepoints with a linear computational cost. Journal of the American Statistical Association, 107(500), 1590–1598.

<a id="Celisse2018">[Celisse2018]</a>
Celisse, A., Marot, G., Pierre-Jean, M., & Rigaill, G. (2018). New efficient algorithms for multiple change-point detection with reproducing kernels. Computational Statistics and Data Analysis, 128, 200–220.

<a id="Arlot2019">[Arlot2019]</a>
Arlot, S., Celisse, A., & Harchaoui, Z. (2019). A kernel multiple change-point algorithm via model selection. Journal of Machine Learning Research, 20(162), 1–56.
