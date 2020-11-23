# Dynamic programming (`Dynp`)

## Description

The method is implemented in both [`Dynp`][ruptures.detection.dynp.Dynp], which is a full native python implementation for which the user can choose any cost functions defined in `ruptures.costs`

It finds the (exact) minimum of the sum of costs by computing the cost of all subsequences of a given signal.
It is called "dynamic programming" because the search over all possible segmentations is ordered using a dynamic programming approach.

In order to work, **the user must specify in advance the number of changes to detect**.
(Consider using penalized methods when this number is unknown.)

The complexity of the dynamic programming approach is of the order $\mathcal{O}(CKn^2)$, where $K$ is the number of change points to detect, $n$ the number of samples and $C$ the complexity of calling the considered cost function on one sub-signal.
Consequently, piecewise constant models (`model=l2`) are significantly faster than linear or autoregressive models.

To reduce the computational cost, you can consider only a subsample of possible change point indexes, by changing the `min_size` and `jump` arguments when instantiating [Dynp](#ruptures.detection.Dynp):

- `min_size` controls the minimum distance between change points; for instance, if `min_size=10`, all change points will be at least 10 samples apart.
- `jump` controls the grid of possible change points; for instance, if `jump=k`, only changes at `k, 2*k, 3*k,...` are considered.

## Usage

```python
import numpy as np
import matplotlib.pylab as plt
import ruptures as rpt

# creation of data
n, dim = 500, 3
n_bkps, sigma = 3, 5
signal, bkps = rpt.pw_constant(n, dim, n_bkps, noise_std=sigma)

# change point detection
model = "l1"  # "l2", "rbf"
algo = rpt.Dynp(model=model, min_size=3, jump=5).fit(signal)
my_bkps = algo.predict(n_bkps=3)

# show results
rpt.show.display(signal, bkps, my_bkps, figsize=(10, 6))
plt.show()
```