import numpy as np


def convert_np(signal):
    """
    To convert any iterable to a numpy array with proper dimensions.

    :param signal: iterable.
    :return: numpy array.
    """
    s = np.array(signal)
    if s.ndim == 0:
        s = s.reshape(1, 1)
    elif s.ndim == 1:
        s = s[:, np.newaxis]
    return s
