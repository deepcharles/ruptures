import abc
from ruptures.search_methods.memoizedict import MemoizeDict


class BaseClass(metaclass=abc.ABCMeta):
    """Base class for the class which will hold the changepoint detection
    algorithms."""

    def __init__(self):
        self.error = MemoizeDict(self.error)
        self.search_method = MemoizeDict(self.search_method)
        self.set_params = MemoizeDict(self.set_params)

    @property
    def signal(self):
        return self._signal

    @signal.setter
    def signal(self, s):
        # since signal has changed, we reset the search_method and error cache.
        self.search_method = MemoizeDict(self.search_method.func)
        self.error = MemoizeDict(self.error.func)
        self.set_params = MemoizeDict(self.set_params.func)
        if s.ndim == 1:
            self._signal = s.reshape(-1, 1)
        else:
            self._signal = s
        self.n = self.signal.shape[0]

    @abc.abstractmethod
    def set_params(self, *args, **kwargs):
        """In case some parameters have to be set before computing the cost on a
        segment.

        Returns:
            None: Returns nothing, just set the relevant attributes
        """
        pass

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
    def search_method(self, start, end, *args, **kwargs):
        """Search the partition space for the best partition

        Args:
            start (int): start index of the segment
            end (int): end index of the segment
            *args: arguments
            **kwargs: arguments

        Returns:
            dict: {(start, end): cost value associated with the partition }
        """
        pass

    @abc.abstractmethod
    def fit(self, signal):
        """ To call the segmentation algorithm"""
        pass
