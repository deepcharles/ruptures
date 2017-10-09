r"""
====================================================================================================
Evaluation metrics
====================================================================================================

:mod:`ruptures.metrics` provides metrics to evaluate change point detection performances.

.. toctree::
    :glob:
    :maxdepth: 1

    hausdorff
    randindex
    precision

"""
from .hausdorff import hausdorff
from .timeerror import meantime
from .precisionrecall import precision_recall
from .hamming import hamming
from .randindex import randindex
