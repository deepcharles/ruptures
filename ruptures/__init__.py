"""
Offline change point detection for Python
====================================================================================================
"""

from .exceptions import NotEnoughPoints
from .datasets import pw_constant, pw_normal, pw_linear, pw_wavy
from .detection import (Binseg, BottomUp, Dynp, Omp, OmpK, Pelt, Window,
                        GreedyAR, GreedyLinear)
from .show import display
