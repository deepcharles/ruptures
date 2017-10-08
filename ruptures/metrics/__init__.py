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
from .zerooneloss import zero_one_loss
from .precisionrecall import precision_recall, pr_curve
from .hamming import hamming
from .randindex import randindex
from .auc_score import auc
