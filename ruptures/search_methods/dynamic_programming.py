from math import ceil
from ruptures.base import BaseClass
import abc
from ruptures.search_methods import MemoizeDict
from ruptures.search_methods import sanity_check


class Dynp(BaseClass, metaclass=abc.ABCMeta):
    """Optimisation using dynamic programming. Given a error function, it computes
        the best partition for which the sum of errors is minimum."""

    def __init__(self):
        super(Dynp, self).__init__()

    def reset_params(self, jump=1, min_size=2):
        """Reset all the necessary parameters to compute a partition

        Args:
            n_regimes (int): number of segments that are expected
            jump (int): number of points between two potential changepoints
            min_size (int): minimum size of a segment

        Returns:
            None:
        """
        self.jump = jump
        self.min_size = min_size
        # do not forget to reset the self.search_method if important attributes
        # have been changed.
        self.search_method = MemoizeDict(self.search_method.func)

    def search_method(self, start, end, n_regimes):
        """Finds the best partition for the segment [start:end]

        Args:
            start (int): start index.
            end (int): end index.
            n_regimes (int, optional): number of segments. Defaults to 2.
            jump (int, optional): number of points between two potential
            changepoints. Defaults to 1
            min_size (int, optional): minimum size of a segment. Defaults to 2

        Returns:
            dict: {(start, end): cost value on the segment,...}
        """
        n = end - start
        if not sanity_check(n, n_regimes, self.jump, self.min_size):
            return {(start, end): float("inf")}

        elif n_regimes == 2:  # two segments
            """ Initialization step. """
            # Admissible breakpoints
            admissible_bkps = range(
                start + ceil(self.min_size / self.jump) * self.jump,
                end - self.min_size + 1,
                self.jump)

            error_list = [(bkp, self.error(start, bkp), self.error(bkp, end))
                          for bkp in admissible_bkps]

            best_bkp, left_error, right_error = min(
                error_list, key=lambda z: z[1] + z[2])
            return {(start, best_bkp): left_error,
                    (best_bkp, end): right_error}

        else:
            # to store the current value of the minimum
            current_min = None
            # to store the breaks corresponding to the current minimum
            current_breaks = None

            # Admissible breakpoints
            admissible_bkps = range(
                start + ceil(self.min_size / self.jump) * self.jump,
                end - self.min_size + 1,
                self.jump)

            for tmp_bkp in admissible_bkps:

                if sanity_check(end - tmp_bkp, n_regimes - 1,
                                self.jump, self.min_size):

                    left_err = self.error(start, tmp_bkp)

                    right_partition = self.search_method(
                        tmp_bkp,
                        end,
                        n_regimes - 1)

                    tmp_min = left_err + sum(right_partition.values())

                    if current_min is None:
                        current_min = tmp_min
                        current_breaks = right_partition.copy()
                        current_breaks.update({(start, tmp_bkp): left_err})
                    elif tmp_min < current_min:
                        current_min = tmp_min
                        current_breaks = right_partition.copy()
                        current_breaks.update({(start, tmp_bkp): left_err})

            return current_breaks

    def fit(self, signal, n_regimes, jump, min_size):
        self.reset_params(jump, min_size)
        self.n_regimes = n_regimes
        self.signal = signal
        n_samples = self.signal.shape[0]
        self.partition = self.search_method(
            0, n_samples, self.n_regimes)

        # we return the end of each segment of the partition
        self.bkps = sorted(e for (s, e) in self.partition)

        return self.bkps
