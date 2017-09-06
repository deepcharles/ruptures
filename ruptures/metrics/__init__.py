"""
:mod:`ruptures` provides evaluation to evaluate change point detection performance.

.. _sec-hausdorff:

Hausdorff metric
----------------------------------------------------------------------------------------------------

.. _sec-randindex:

Rand index
----------------------------------------------------------------------------------------------------


.. _sec-precision:

Precision and Recall
----------------------------------------------------------------------------------------------------

Also PR curve and AUC.

"""
from .hausdorff import hausdorff
from .timeerror import meantime
from .zerooneloss import zero_one_loss
from .precisionrecall import precision_recall, pr_curve
from .hamming import hamming
from .randindex import randindex
from .auc_score import auc
