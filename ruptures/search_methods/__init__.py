from .base import BaseClass
from .memoizedict import MemoizeDict
from .sanity_check import sanity_check
from .dynamic_programming import Dynp
from .pelt import Pelt

METHODS = {"dynp": Dynp,
           "pelt": Pelt}

from .abstraction import changepoint
