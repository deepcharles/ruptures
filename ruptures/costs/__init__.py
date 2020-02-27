r"""
.. _sec-costs:

====================================================================================================
Cost functions
====================================================================================================

.. toctree::
    :glob:
    :maxdepth: 1

    costl1
    costl2
    costnormal
    costrbf
    costlinear
    costautoregressive
    costml
    costrank
    costcustom

"""

from ruptures.exceptions import NotEnoughPoints
from .factory import cost_factory
from .costl1 import CostL1
from .costl2 import CostL2
from .costlinear import CostLinear
from .costrbf import CostRbf
from .costnormal import CostNormal
from .costautoregressive import CostAR
from .costml import CostMl
from .costrank import CostRank
