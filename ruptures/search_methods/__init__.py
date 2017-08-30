"""Import modules"""
# from .abstraction import changepoint
from .base import BaseClass
from .binseg import Binseg
from .bottomup import BottomUp
from .dynp import Dynp
from .omp import Omp
from .ompk import OmpK
from .pelt import Pelt
from .sanity_check import sanity_check
from .window import Window
from .window_lr_cov import WinLr
from .window_mmd import WinMmd
from .greedyar import GreedyAR
from .greedylinear import GreedyLinear

METHODS = {"dynp": Dynp,
           "pelt": Pelt}
