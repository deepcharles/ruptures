"""
Offline change point detection for Python
====================================================================================================
"""

from .exceptions import NotEnoughPoints
from .datasets import pw_constant
from .detection import (Binseg, BottomUp, Dynp, Omp, OmpK, Pelt, Window,
                        GreedyAR, GreedyLinear)
from .show import display
