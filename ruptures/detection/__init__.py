r"""
====================================================================================================
Search methods
====================================================================================================

The :mod:`ruptures.detection` module implements the change point detection methods.

.. toctree::
    :glob:
    :maxdepth: 1

    dynp
    pelt
    binseg
    bottomup
    window
.. greedy/index
"""
# Exact methods
# ==================================================================================================
# - Least squares estimation
# - Least absolute deviation estimation
# - Linear model
# - Exact kernel change point detection
# - Penalized least squares
# - Penalized least absolute deviation
# - Penalized linear model
# - Maximum likelihood estimation (multivariate Gaussian variables)
# - Penalized maximum likelihood estimation (multivariate Gaussian density)

# Approximate methods:
# ==================================================================================================
# - Binary segmentation:
#     - Least squares, least absolute deviation, kernel error,
#       maximum likelihood (multivariate Gaussian density).
# - Bottom-up:
#     - Least squares, least absolute deviation, kernel error,
#       maximum likelihood (multivariate Gaussian density).
# - Sliding window:
#     - Student t-test, likelihood ratio test (multivariate Gaussian density),
#       (kernel) maximum mean discrepancy.
# - Greedy detection:
#     - Least squares, kernel error, autoregressive model :ref:`greedy-ar`, linear model.

from .binseg import Binseg
from .bottomup import BottomUp
from .dynp import Dynp
from .omp import Omp
from .ompk import OmpK
from .pelt import Pelt
from .window import Window
from .greedyar import GreedyAR
from .greedylinear import GreedyLinear
