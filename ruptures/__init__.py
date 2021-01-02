"""
Offline change point detection for Python
====================================================================================================
"""

from .datasets import pw_constant, pw_linear, pw_normal, pw_wavy
from .detection import Binseg, BottomUp, Dynp, KernelCPD, Pelt, Window
from .exceptions import NotEnoughPoints
from .datasets import pw_constant, pw_normal, pw_linear, pw_wavy
from .detection import (
    Binseg,
    BottomUp,
    Dynp,
    KernelCPD,
    Pelt,
    Window,
)
from .show import display
