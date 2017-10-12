# Python library for change point detection in time series

Link to documentation

## Install
Download package and inside the folder:
> python3 setup.py install

or
> python3 setup.py develop

## Simple example

```python
import matplotlib.pyplot as plt
import ruptures as rpt

n_samples, dim, sigma = 1000, 3, 4
n_bkps = 4  # nombre de ruptures
# generate signal
signal, bkps = rpt.pw_constant(n_samples, dim, n_bkps, noise_std=sigma)
# detection
algo = rpt.Pelt(model="rbf").fit(signal)
result = algo.predict(pen=10)
# display
rpt.display(signal, bkps, result)
plt.show()
```
