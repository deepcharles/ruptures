# Penalized L1 Potts segmentation (`L1Potts`)

## Description

The method is implemented in [`L1Potts`][ruptures.detection.l1potts.L1Potts].
It computes the global minimizer of the **L1 Potts functional** for piecewise constant 1D signals:

$$
\min_{u \in \mathbb{R}^N} \;\; \gamma \sum_{i=1}^{N-1} \mathbb{1}(u_i \neq u_{i+1}) \;+\; \sum_{i=1}^{N} w_i \, |f_i - u_i|
$$

where $f$ is the observed signal, $w$ are non-negative per-sample weights, and $\gamma > 0$ is the jump penalty.

The L1 fit makes the estimator robust to heavy-tailed noise and outliers, in contrast to the L2 Potts model used by other ruptures detectors (`Pelt(model="l2")`, `Dynp(model="l2")`).

The implementation is **Algorithm 1 of [[Storath2017]](#Storath2017)**, which solves the problem exactly in $\mathcal{O}(KN)$ time, where $N$ is the number of samples and $K$ the number of distinct values in the signal. The algorithm is much faster than `Pelt(model="l1")` (a 20–30× speedup is typical on a few-thousand-sample noisy signal). It uses a Viterbi-type dynamic program over (level, sample) pairs, where the candidate levels are the unique observed values — the optimal segment level is always one of them, since the weighted L1 median lies in the data.

`L1Potts` accepts only 1D signals. Penalty-only mode (`predict(pen=...)`) is the only supported prediction mode; `n_bkps` and `epsilon` are not.


## Usage

```python
import numpy as np
import matplotlib.pylab as plt
import ruptures as rpt

# creation of data with heavy-tailed (Laplace) noise
n, sigma = 500, 1.0
n_bkps = 3
signal, bkps = rpt.pw_constant(n, 1, n_bkps, noise_std=sigma)
signal = signal.ravel() + np.random.default_rng(0).laplace(scale=sigma, size=n)

# change point detection
algo = rpt.L1Potts().fit(signal)
my_bkps = algo.predict(pen=3.0)

# show results
rpt.show.display(signal, bkps, my_bkps, figsize=(10, 6))
plt.show()
```

To downweight known outlier samples, pass per-sample weights to `fit`:

```python
weights = np.ones(n)
weights[outlier_indices] = 1e-3  # near-zero weight effectively ignores those samples
my_bkps = rpt.L1Potts().fit(signal, weights=weights).predict(pen=3.0)
```

`fit_predict` is also available:

```python
my_bkps = rpt.L1Potts().fit_predict(signal, pen=3.0, weights=weights)
```


## References

<a id="Storath2017">[Storath2017]</a>
Storath, M., Weinmann, A., & Unser, M. (2017). Jump-penalized least absolute values estimation of scalar or circle-valued signals. *Information and Inference: A Journal of the IMA*. Preprint: <https://bigwww.epfl.ch/preprints/storath1602p.pdf>
