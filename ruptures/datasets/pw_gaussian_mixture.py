import numpy as np
from numpy import random as rd

from ruptures.utils import draw_bkps


def select_mean_and_standard_deviation(ratio):
    for i in range(len(ratio) - 1):
        if rd.random() > ratio[i]:
            return i
    return len(ratio) - 1


def get_gaussian_mixture_segment(n_features, ratio, means, standard_deviations, length):
    result = np.empty((length, n_features), dtype=float)
    for i in range(length):
        mean_index = select_mean_and_standard_deviation(ratio)
        result[i] = rd.normal(means[mean_index], standard_deviations[mean_index])
    return result


def generate_means_and_standard_deviations(n_components, noise_std, mean_range):
    means = [0] * n_components
    standard_deviations = [0] * n_components
    mean_min, mean_max = mean_range
    for i in range(n_components):
        means[i] = rd.random() * (mean_max - mean_min)
        print("mean {} : {}".format(i, means[i]))
        if noise_std is not None:
            standard_deviations[i] = rd.random() * noise_std
        print("std {} : {}".format(i, standard_deviations[i]))
    return means, standard_deviations


def pw_gaussian_mixture(n_samples=200, n_features=1, n_bkps=3, noise_std=None, n_components=2, ratio=[0.8, 0.2],
                        mean_range=(1, 10)):
    """Return a piecewise constant signal and the associated changepoints.

    Args:
        n_samples (int): signal length
        n_features (int, optional): number of dimensions
        n_bkps (int, optional): number of changepoints
        noise_std (float, optional): noise std. If None, no noise is added
        n_components (int, optional): the number of gaussian distributions to be mixed
        ratio (float[], optional): an array of floats of length n_components. Must sum to 1. Represents the ratio
                                    of different gaussian components

    Returns:
        tuple: signal of shape (n_samples, n_features), list of breakpoints

    """
    breakpoints = draw_bkps(n_samples, n_bkps)
    # we create the signal
    signal = np.empty((n_samples, n_features), dtype=float)
    indices = np.arange(n_samples)

    subsection = 0

    for subarray in np.split(indices, breakpoints):
        if subarray.size > 0:
            print("generating subsection {}".format(subsection))
            subsection += 1
            means, standard_deviations = generate_means_and_standard_deviations(n_components, noise_std, mean_range)
            gaussian_mixture_segment = get_gaussian_mixture_segment(n_features, ratio, means, standard_deviations,
                                                                    subarray.size)
            signal[subarray] = gaussian_mixture_segment

    return signal, breakpoints
