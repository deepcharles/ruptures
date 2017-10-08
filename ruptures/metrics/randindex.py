r"""
.. _sec-randindex:

Rand index
====================================================================================================

Description
----------------------------------------------------------------------------------------------------

The Rand index measures the similarity between two segmentations.
Formally, for a signal :math:`\{y_t\}_t` and a segmentation :math:`\mathcal{S}`, denote by :math:`A` the associated membership matrix:

    .. math::
        \mathcal{A}_{ij} &= 1 \text{ if both samples } y_i \text{ and } y_j \text{ are in the same segment according to } \mathcal{S} \\
        &= 0 \quad\text{otherwise}

Let :math:`\hat{\mathcal{S}}` be the estimated segmentation and :math:`\hat{A}`, the associated membership matrix.
Then the Rand index is equal to

    .. math::
        \frac{\sum_{i<j} \mathbb{1}(A_{ij} = \hat{A}_{ij})}{T(T-1)/2}

where :math:`T` is the number of samples.
It has a value between 0 and 1: 0 indicates that the two segmentations do not agree on any pair of points and 1 indicates that the two segmentations are exactly the same.


.. figure:: /images/randindex.png
   :scale: 50 %
   :alt: Schematic view of the RandIndex metric

   Schematic example: true segmentation in gray, estimated segmentation in dashed lines and their associated membership matrices. Rand index is equal to 1 minus the gray area.


Usage
----------------------------------------------------------------------------------------------------

Start with the usual imports and create two segmentations to compare.

.. code-block:: python

    from ruptures.metrics import randindex
    bkps1, bkps2 = [100, 200, 500], [105, 115, 350, 400, 500]
    print(randindex(bkps1, bkps2))

Code explanation
----------------------------------------------------------------------------------------------------

.. autofunction:: ruptures.metrics.randindex.randindex

"""
from ruptures.metrics import hamming


def randindex(bkps1, bkps2):
    """Rand index for two partitions. The result is scaled to be within 0 and 1.

    Args:
        bkps1 (list): list of the last index of each regime.
        bkps2 (list): list of the last index of each regime.

    Returns:
        float: Rand index
    """
    return 1 - hamming(bkps1, bkps2)
