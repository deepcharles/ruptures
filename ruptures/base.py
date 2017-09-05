"""Base class for change point detection estimators."""
import abc


class BaseEstimator(metaclass=abc.ABCMeta):

    """Base class for all change point detection estimators.

    Notes
    -----
    All estimators should specify all the parameters that can be set
    at the class level in their ``__init__`` as explicit keyword
    arguments (no ``*args`` or ``**kwargs``).
    """

    @abc.abstractmethod
    def fit(self, *args, **kwargs):
        """ To call the segmentation algorithm"""
        pass

    @abc.abstractmethod
    def predict(self, *args, **kwargs):
        """ To call the segmentation algorithm"""
        pass

    @abc.abstractmethod
    def fit_predict(self, *args, **kwargs):
        """ To call the segmentation algorithm"""
        pass
