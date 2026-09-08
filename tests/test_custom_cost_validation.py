"""custom_cost must be validated, not silently replaced by the default cost.

Regression tests for the silent fallback where a non-BaseCost object
passed as custom_cost caused detection to run with the default 'l2'
cost.
"""

import pytest

import ruptures as rpt
from ruptures.base import BaseCost

DETECTORS = [rpt.Pelt, rpt.Binseg, rpt.BottomUp, rpt.Window, rpt.Dynp]


class NotACost:
    """Implements the cost interface but does not subclass BaseCost."""

    model = "duck"
    min_size = 2

    def fit(self, signal):
        self.signal = signal
        return self

    def error(self, start, end):
        return 0.0


class ValidCost(BaseCost):
    model = "valid"
    min_size = 2

    def fit(self, signal):
        self.signal = signal.reshape(-1, 1) if signal.ndim == 1 else signal
        return self

    def error(self, start, end):
        sub = self.signal[start:end]
        return float(((sub - sub.mean(axis=0)) ** 2).sum())


@pytest.mark.parametrize("detector", DETECTORS)
def test_non_basecost_instance_raises(detector):
    with pytest.raises(TypeError, match="BaseCost"):
        detector(custom_cost=NotACost())


@pytest.mark.parametrize("detector", DETECTORS)
def test_cost_class_instead_of_instance_raises(detector):
    # The mistake from issue #342: passing the class, not an instance.
    with pytest.raises(TypeError, match="BaseCost"):
        detector(custom_cost=NotACost)


@pytest.mark.parametrize("detector", DETECTORS)
def test_valid_custom_cost_is_actually_used(detector):
    algo = detector(custom_cost=ValidCost())
    assert isinstance(algo.cost, ValidCost)
