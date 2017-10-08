r"""
.. _sec-hausdorff:

Hausdorff metric
====================================================================================================

Description
----------------------------------------------------------------------------------------------------

The Hausdorff metric measures the worst prediction error.
Assume a set of change point indexes :math:`t_1,t_2,\dots` and their estimates :math:`\hat{t}_1, \hat{t}_2,\dots`.
The Hausdorff metric is then equal to

    .. math:: \text{Hausdorff}(\{t_k\}_k, \{\hat{t}_k\}_k) :=  \max \{ \max_k \min_l |t_k - \hat{t}_l| \, , \max_k \min_l |\hat{t}_k - t_l|\}.


.. figure:: /images/hausdorff.png
   :scale: 50 %
   :alt: hausdorff metric

   Schematic example: true segmentation in gray, estimated segmentation in dashed lines. Here, Hausdorff is equal to :math:`\max(\Delta t_1, \Delta t_2, \Delta t_3)`.

Usage
----------------------------------------------------------------------------------------------------

Start with the usual imports and create two segmentations to compare.

.. code-block:: python

    from ruptures.metrics import hausdorff
    bkps1, bkps2 = [100, 200, 500], [105, 115, 350, 400, 500]
    print(hausdorff(bkps1, bkps2))


Code explanation
----------------------------------------------------------------------------------------------------

.. autofunction:: ruptures.metrics.hausdorff.hausdorff

"""
import numpy as np
from scipy.spatial.distance import cdist
from ruptures.metrics.sanity_check import sanity_check


def hausdorff(bkps1, bkps2):
    """Compute the Hausdorff distance between changepoints.

    Args:
        bkps1 (list): list of the last index of each regime.
        bkps2 (list): list of the last index of each regime.

    Returns:
        float: Hausdorff distance.
    """
    sanity_check(bkps1, bkps2)
    bkps1_arr = np.array(bkps1[:-1]).reshape(-1, 1)
    bkps2_arr = np.array(bkps2[:-1]).reshape(-1, 1)
    pw_dist = cdist(bkps1_arr, bkps2_arr)
    res = max(pw_dist.min(axis=0).max(), pw_dist.min(axis=1).max())
    return res
