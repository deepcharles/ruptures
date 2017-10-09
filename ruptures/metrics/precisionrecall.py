r"""
.. _sec-precision:

Precision and recall
====================================================================================================


Description
----------------------------------------------------------------------------------------------------

A true changepoint is declared "detected" (or positive) if there is at least one computed changepoint at less than "margin" points from it.
Formally, assume a set of change point indexes :math:`t_1,t_2,\dots` and their estimates :math:`\hat{t}_1, \hat{t}_2,\dots`
In the context of change point detection, precision and recall are defined as follows:

    .. math:: \text{precision}:=|\text{TP}|/|\{\hat{t}_l\}_l| \quad \text{and}\quad\text{recall}:=|\text{TP}|/|\{t_k\}_k|

where, for a given margin :math:`M`, true positives :math:`\text{TP}` are true change points for which there is an estimated one at less than :math:`M` samples, *i.e*

    .. math:: \text{TP}:= \{t_k\,|\, \exists\, \hat{t}_l\,\, \text{s.t.}\, |\hat{t}_l - t_k|<M \}.

.. figure:: /images/precision_recall.png
   :scale: 50 %
   :alt: Schematic view of the precision and recall

   Schematic example: true segmentation in gray, estimated segmentation in dashed lines and margin in dashed areas. Here, precision is 2/3 and recall is 2/2.


Usage
----------------------------------------------------------------------------------------------------

Start with the usual imports and create two segmentations to compare.

.. code-block:: python

    from ruptures.metrics import precision_recall
    bkps1, bkps2 = [100, 200, 500], [105, 115, 350, 400, 500]
    p, r = precision_recall(bkps1, bkps2)
    print((p, r))


The margin paramater :math:`M` can be changed through the keyword ``'margin'`` (default is 10 samples).

.. code-block:: python

    p, r = precision_recall(bkps1, bkps2, margin=10)
    print((p, r))
    p, r = precision_recall(bkps1, bkps2, margin=20)
    print((p, r))


Code explanation
----------------------------------------------------------------------------------------------------

.. autofunction:: ruptures.metrics.precisionrecall.precision_recall

"""
from itertools import groupby, product

import numpy as np

from ruptures.metrics.sanity_check import sanity_check
from ruptures.utils import unzip


def precision_recall(true_bkps, my_bkps, margin=10):
    """Calculate the precision/recall of an estimated segmentation compared with the true segmentation.

    Args:
        true_bkps (list): list of the last index of each regime (true
            partition).
        my_bkps (list): list of the last index of each regime (computed
            partition).
        margin (int, optional): allowed error (in points).

    Returns:
        tuple: (precision, recall)
    """
    sanity_check(true_bkps, my_bkps)
    assert margin > 0, "Margin of error must be positive (margin = {})".format(
        margin)

    if len(my_bkps) == 1:
        return 0, 0

    used = set()
    true_pos = set(true_b
                   for true_b, my_b in product(true_bkps[:-1], my_bkps[:-1])
                   if my_b - margin < true_b < my_b + margin and
                   not (my_b in used or used.add(my_b)))

    tp_ = len(true_pos)
    precision = tp_ / (len(my_bkps) - 1)
    recall = tp_ / (len(true_bkps) - 1)
    return precision, recall
