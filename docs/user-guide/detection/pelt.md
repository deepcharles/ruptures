# Linearly penalized segmentation (`Pelt`)

# Description

The method is implemented in [`Pelt`][ruptures.detection.pelt.Pelt].

Because the enumeration of all possible partitions impossible, the algorithm relies on a pruning rule.
Many indexes are discarded, greatly reducing the computational cost while retaining the
ability to find the optimal segmentation.
The implementation follows [[Killick2012]](#Killick2012).
In addition, under certain conditions on the change point repartition, the avarage computational complexity is of the order of $\mathcal{O}(CKn)$, where $K$ is the number of change points to detect, $n$ the number of samples and $C$ the complexity of calling the considered cost function on one sub-signal.
Consequently, piecewise constant models (`model=l2`) are significantly faster than linear or autoregressive models.

To reduce the computational cost, you can consider only a subsample of possible change point indexes, by changing the `min_size` and `jump` arguments when instantiating [Pelt](#ruptures.detection.Pelt):

- `min_size` controls the minimum distance between change points; for instance, if `min_size=10`, all change points will be at least 10 samples apart.
- `jump` controls the grid of possible change points; for instance, if `jump=k`, only changes at `k, 2*k, 3*k,...` are considered.


## Usage

```python
import numpy as np
import matplotlib.pylab as plt
import ruptures as rpt

# creation of data
n, dim = 500, 3
n_bkps, sigma = 3, 1
signal, bkps = rpt.pw_constant(n, dim, n_bkps, noise_std=sigma)

# change point detection
model = "l1"  # "l2", "rbf"
algo = rpt.Pelt(model=model, min_size=3, jump=5).fit(signal)
my_bkps = algo.predict(pen=3)

# show results
fig, ax_arr = rpt.display(signal, bkps, my_bkps, figsize=(10, 6))
plt.show()
```

## References

<a id="Killick2012">[Killick2012]</a>
Killick, R., Fearnhead, P., & Eckley, I. (2012). Optimal detection of changepoints with a linear computational cost. Journal of the American Statistical Association, 107(500), 1590–1598.
