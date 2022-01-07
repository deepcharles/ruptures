# Rand index (`randindex`)

## Description

The Rand index ($RI$) measures the similarity between two segmentations and is
equal to the proportion of aggreement between two partitions.
Formally, for $\mathcal{T}_1$ and $\mathcal{T}_2$ two partitions of $\{1, 2,\dots,T\}$,

$$
RI := \frac{N_0 + N_1}{T(T+1)/2}
$$

where

- $N_0$ is the number of pairs of samples that belong to the same segment
according to $\mathcal{T}_1$ and $\mathcal{T}_2$,
- $N_1$ is the number of pairs of samples that belong to different segments
according to $\mathcal{T}_1$ and $\mathcal{T}_2$.

$RI$ is between 0 (total disagreement) and 1 (total agreement).
It is available in the [`randindex`][ruptures.metrics.randindex.randindex]
function which uses the efficient implementation of [[Prates2021]](#Prates2021).

## Usage

Start with the usual imports and create two segmentations to compare.

```python
from ruptures.metrics import randindex

bkps1, bkps2 = [100, 200, 500], [105, 115, 350, 400, 500]
print(randindex(bkps1, bkps2))
```

## References

<a id="Prates2021">[Prates2021]</a>
Prates, L. (2021). A more efficient algorithm to compute the Rand Index for
change-point problems. ArXiv:2112.03738.