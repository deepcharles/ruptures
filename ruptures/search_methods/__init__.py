"""Import modules"""
from .base import BaseClass
from .sanity_check import sanity_check
from .dynp import Dynp
from .pelt import Pelt
from .binseg import Binseg
from .omp import Omp
METHODS = {"dynp": Dynp,
           "pelt": Pelt}

from .abstraction import changepoint
