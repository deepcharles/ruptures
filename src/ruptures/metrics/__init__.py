r"""
====================================================================================================
Evaluation
====================================================================================================

:mod:`ruptures.metrics` provides metrics to evaluate change point detection performances and
:mod:`ruptures.show` provides a display function for visual inspection.

.. toctree::
    :glob:
    :maxdepth: 1

    hausdorff
    randindex
    precision
    display

"""
from .hausdorff import hausdorff
from .timeerror import meantime
from .precisionrecall import precision_recall
from .hamming import hamming
from .randindex import randindex
