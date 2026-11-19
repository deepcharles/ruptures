# Poisson rate changes

Use `model="poisson"` for nonnegative integer counts measured over equal
exposures, such as requests per minute. Each segment has its own constant
rate for each feature. A feature with only zeros in a segment has rate zero.

For a segment of length $n$ with total count $s$, the fitted rate is $s/n$
and the cost is $-s\log(s/n)$, with zero cost for $s=0$. This omits terms
that sum to the same constant for every segmentation. For multiple features
the costs add. Cumulative counts make the work per segment independent of
its length.

```python
import numpy as np
import ruptures as rpt

counts = np.repeat([0, 6, 1], 30)
ends = rpt.Pelt(model="poisson", jump=1).fit_predict(counts, pen=5)
# [30, 60, 90]
```
