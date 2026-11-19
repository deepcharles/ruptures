import numpy as np
import pytest
from scipy.special import gammaln
from scipy.stats import poisson

from ruptures import Pelt
from ruptures.costs import CostPoisson, cost_factory
from ruptures.exceptions import NotEnoughPoints


def test_poisson_log_likelihood():
    counts = np.array([[0, 4], [0, 2], [3, 0], [0, 9]], dtype=np.uint8)
    cost = cost_factory("poisson").fit(counts)
    for start in range(len(counts)):
        for end in range(start + 1, len(counts) + 1):
            segment = counts[start:end].astype(float)
            nll = -poisson.logpmf(segment, segment.mean(axis=0)).sum()
            constant = (segment + gammaln(segment + 1)).sum()
            assert cost.error(start, end) == pytest.approx(nll - constant)
    with pytest.raises(NotEnoughPoints):
        cost.error(2, 2)


def test_poisson_pelt():
    counts = np.repeat([0, 6, 1], 30)
    assert Pelt(model="poisson", jump=1).fit_predict(counts, pen=5) == [30, 60, 90]


@pytest.mark.parametrize(
    "counts", [[-1], [1.5], [np.nan], [np.inf], np.ones((2, 2, 2))]
)
def test_invalid_poisson_counts(counts):
    with pytest.raises(ValueError):
        CostPoisson().fit(np.asarray(counts))
