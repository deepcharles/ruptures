"""
Offline change point detection for Python
====================================================================================================
"""

from .costs import NotEnoughPoints
from .datasets import pw_constant
from .detection import (Binseg, BottomUp, Dynp, Omp, OmpK, Pelt, Window,
                        WinLr, WinMmd, GreedyAR, GreedyLinear)
