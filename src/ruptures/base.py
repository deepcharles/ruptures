r"""All estimators and cost functions are subclasses of.

[`BaseEstimator`][ruptures.base.BaseEstimator] and
[`BaseCost`][ruptures.base.BaseCost] respectively.
"""
import abc
from ruptures.utils import pairwise


class BaseEstimator(metaclass=abc.ABCMeta):
    """Base class for all change point detection estimators.

    Notes:
        All estimators should specify all the parameters that can be set
        at the class level in their ``__init__`` as explicit keyword
        arguments (no ``*args`` or ``**kwargs``).
    """

    @abc.abstractmethod
    def fit(self, *args, **kwargs):
        """To call the segmentation algorithm."""
        pass

    @abc.abstractmethod
    def predict(self, *args, **kwargs):
        """To call the segmentation algorithm."""
        pass

    @abc.abstractmethod
    def fit_predict(self, *args, **kwargs):
        """To call the segmentation algorithm."""
        pass


class BaseCost(object, metaclass=abc.ABCMeta):
    """Base class for all segment cost classes.

    Notes:
        All classes should specify all the parameters that can be set
        at the class level in their ``__init__`` as explicit keyword
        arguments (no ``*args`` or ``**kwargs``).
    """

    @abc.abstractmethod
    def fit(self, *args, **kwargs):
        """Set the parameters of the cost function, for instance the Gram
        matrix, etc."""
        pass

    @abc.abstractmethod
    def error(self, start, end):
        """Returns the cost on segment [start:end]."""
        pass

    def sum_of_costs(self, bkps):
        """Returns the sum of segments cost for the given segmentation.

        Args:
            bkps (list): list of change points. By convention, bkps[-1]==n_samples.

        Returns:
            float: sum of costs
        """
        soc = sum(self.error(start, end) for start, end in pairwise([0] + bkps))
        return soc

    @property
    @abc.abstractmethod
    def model(self):
        pass
