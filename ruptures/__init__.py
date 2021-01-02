"""Offline change point detection for Python."""

from .datasets import pw_constant, pw_linear, pw_normal, pw_wavy
from .detection import Binseg, BottomUp, Dynp, KernelCPD, Pelt, Window
from .exceptions import NotEnoughPoints
from .show import display

__version__ = "1.1.2dev0"
