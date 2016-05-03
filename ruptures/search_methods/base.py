import abc
from functools import lru_cache


class BaseClass(metaclass=abc.ABCMeta):

    """Base class for the class which will hold the changepoint detection
    algorithms."""

    def __init__(self):
        self.cache = lru_cache(maxsize=None)(self.cache)

    @property
    def signal(self):
        return self._signal

    @signal.setter
    def signal(self, s):
        # since signal has changed, we reset the cache.
        self.cache.cache_clear()
        if s.ndim == 1:
            self._signal = s.reshape(-1, 1)
        else:
            self._signal = s
        self.n = self.signal.shape[0]

    @abc.abstractmethod
    def error(self, start, end):
        """Compute the cost to minimize on the segment start:end

        Args:
            start (int): start index of the segment
            end (int): end index of the segment

        Returns:
            float: the cost value
        """
        pass

    @abc.abstractmethod
    def fit(self, signal):
        """ To call the segmentation algorithm"""
        pass
