# Custom cost class

Users who are interested in detecting a specific type of change can easily do so by creating a custom cost function.
Provided, they subclass the base cost function [`BaseCost`][ruptures.base.BaseCost], they will be able to seamlessly run the algorithms implemented in `ruptures`.

!!! important
    The custom cost class must at least implement the two following methods: `.fit(signal)` and `.error(start, end)` (see [user guide](../../custom-cost-function.md)).

## Example

Let $\{y_t\}_t$ denote a 1D piecewise stationary random process.
Assume that the $y_t$ are independent and exponentially distributed with a scale parameter that shifts at some unknown instants $t_1,t_2,\dots$
The change points estimates are the minimizers of the negative log-likelihood, and the associated cost function is given by

$$
c(y_I) = |I| \log \bar{\mu}_I
$$

where $I,\, y_I$ and $\bar{\mu}_I$ are respectively an interval, the sub-signal on this interval and the empirical mean of this sub-signal.
The following code implements this cost function:

```python
from math import log
from ruptures.base import BaseCost


class MyCost(BaseCost):
    """Custom cost for exponential signals."""

    # The 2 following attributes must be specified for compatibility.
    model = ""
    min_size = 2

    def fit(self, signal):
        """Set the internal parameter."""
        self.signal = signal
        return self

    def error(self, start, end):
        """Return the approximation cost on the segment [start:end].

        Args:
            start (int): start of the segment
            end (int): end of the segment

        Returns:
            float: segment cost
        """
        sub = self.signal[start:end]
        return (end - start) * log(sub.mean())
```

!!! warning
    For compatibility reasons, the static attributes `model` and `min_size` must be explicitly specified:

    - `model` is simply a string containing the name of the cost function (can be empty);
    - `min_size` is a positive integer that indicates the minimum segment size (in number of samples) on which the cost function can be applied.

This cost function can now be used with all algorithms from `ruptures`.
For instance,

```python
import numpy as np
import matplotlib.pylab as plt
import ruptures as rpt

# creation of data
a = np.random.exponential(scale=1, size=100)
b = np.random.exponential(scale=2, size=200)
signal, bkps = np.r_[a, b, a], [100, 300, 400]
# cost
algo = rpt.Pelt(custom_cost=MyCost()).fit(signal)
my_bkps = algo.predict(pen=10)
# display
rpt.display(signal, bkps, my_bkps)
plt.show()
```
