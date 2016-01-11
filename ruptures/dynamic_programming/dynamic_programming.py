from ruptures.dynamic_programming import MemoizeDict
from math import ceil
from ruptures.base import BaseEstimator


@MemoizeDict
def _sanity_check(d, le, jump, min_size):
    """
    :param d: int. Number of regimes
    :param le: int. Length of the signal
    :param jump: int. The start index of each regime can only be a multiple of
        "jump" (and the end index = -1 modulo "jump")
    :param min_size: int. The minimum size of a segment
    :return: bool. True if there exists a potential configuration of
        breakpoints for the given parameters. False if it does not.
    """
    assert isinstance(d, int)
    assert isinstance(le, int)
    assert isinstance(jump, int)
    assert isinstance(min_size, int)

    q = int(le / jump)  # number of possible breakpoints
    if d > q + 1:  # Are there enough points for the given number of regimes?
        return False
    if (d - 1) * ceil(min_size / jump) * jump + min_size > le:
        return False
    return True


def sanity_check(d, start, end, jump, min_size=1):
    """
    :param d: int. Number of regimes
    :param start: int. Index of the first point of the segment
    :param end: int. Index of the last point of the segment
    :param jump: int. The start index of each regime can only be a multiple of
        "jump" (and the end index = -1 modulo "jump")
    :param min_size: int. The minimum size of a segment
    :return: bool. True if there exists a potential configuration of
        breakpoints for the given parameters. False if it does not.
    """
    # changements: l = end - start + 1  # signal length
    l = end - start  # signal length
    return _sanity_check(d, l, jump, min_size)


@MemoizeDict
def dynamic_prog(err_func, d, start, end, jump=1, min_size=1):
    """
    Optimisation using dynamic programming. Given a error function, it computes
        the best partition for which the sum of errors is minimum.
    The error function returns the error on a segment (err_func(start, end)).

    :param err_func: function. error(start, end) must return the approximation
        error on the segment [start, end]
    :param d: int. The number of regimes (so there are d-1 breakpoints)
    :param start: int. First index of the segment on which the computation is
        made
    :param end: int. Last index of the segment on which the computation is made
    :param jump: int. The start index of each regime can only be a multiple of
        "jump" (and the end index = -1 modulo "jump"). This allows to perform
        less approximation error estimations.
    :return: dictionary. Each key is a tuple (start, end) representing a
        segment, each item is the approximation error. (so there are d items in
        the dictionary).
    """
    if not sanity_check(d, start, end, jump, min_size):
        return {(start, end): float("inf")}
    elif d == 2:  # two segments
        """ Initialization step. """
        error_list = [(breakpoint,
                       err_func(start, breakpoint),
                       err_func(breakpoint, end))
                      for breakpoint in range(start + ceil(min_size / jump) *
                                              jump, end - min_size + 1,
                                              jump)]

        best_bkp, left_error, right_error = min(
            error_list, key=lambda z: z[1] + z[2])
        return {(start, best_bkp): left_error, (best_bkp, end): right_error}

    else:
        current_min = None  # to store the current value of the maximum
        # to store the breaks corresponding to the current maximum
        current_breaks = None
        for tmp_bkp in range(start + ceil(min_size / jump) * jump, end -
                             min_size + 1, jump):
            if sanity_check(d - 1, tmp_bkp, end, jump, min_size):
                tmp_err = err_func(start, tmp_bkp)
                tmp = dynamic_prog(
                    err_func, d - 1, tmp_bkp, end, jump, min_size)
                tmp_min = sum(tmp.values()) + tmp_err
                if current_min is None:
                    current_min = tmp_min
                    current_breaks = tmp.copy()
                    current_breaks.update({(start, tmp_bkp): tmp_err})
                elif tmp_min < current_min:
                    current_min = tmp_min
                    current_breaks = tmp.copy()
                    current_breaks.update({(start, tmp_bkp): tmp_err})
        return current_breaks


class dynp(BaseEstimator):
    """Wrapper qui sera utilisé pour appeler la fonction dynamic_prog.
    Tous les paramètres sont définis dans __init__ et la méthode fit
    calcule la segmentation et renvoie les temps de ruptures."""

    def __init__(self, error_func, n, n_regimes, min_size=2, jump=1):
        self.error_func = error_func
        assert isinstance(n, int)
        assert n > n_regimes  # at least three points
        self.n = n
        self.n_regimes = n_regimes
        assert min_size > 0
        self.min_size = min_size
        self.jump = jump
        self.chg = list()  # will contain the changepoint indexes.

    def fit(self):
        """Returns the changepoint indexes."""
        self.chg = list()
        dp = dynamic_prog(self.error_func, self.n_regimes, 0,
                          self.n, self.jump, self.min_size)
        self.chg = sorted([s for (s, e) in dp])
        return self.chg
