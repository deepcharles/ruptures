Exact change point detection
====================================================================================================

The :mod:`ruptures.detection` module implements the change point detection methods.

.. toctree::
    :glob:
    :maxdepth: 1

    dynp
    pelt

Exact change point detection is implemented in :class:`ruptures.detection.Dynp` and
:class:`ruptures.detection.Pelt`.
Depending on whether the number of change points is known beforehand, one can choose one or the
other.
Both classes solve a combinatorial minimization problem.

General formulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Known number of change points
----------------------------------------------------------------------------------------------------

For a given model and known number of change points, the change point detection problem amounts to
minimize the approximation error over all the potential breakpoint repartitions.
Thanks to the additive nature of the change point detection problem, a dynamic programming
algorithm is able to find the optimal partition which minimizes a sum of costs measuring
approximation error.
Formally,

.. math:: \widehat{\mathbf{p}}_K = \arg \min_{\mathbf{p}} \sum_{i=1}^{K} c(y_{p_i}).

The method is implemented in :class:`ruptures.detection.Dynp` (see :cite:`a-bai2003computation`)

Unknown number of change points
----------------------------------------------------------------------------------------------------

For a given model, the change point detection problem amounts to
minimize the penalized approximation error over all the potential breakpoint repartitions
The penalty is proportional to the number of change points.
The higher the penalty value, the less change points are predicted.

Formally,

.. math:: \widehat{\mathbf{p}}_{\beta} = \arg \min_{\mathbf{p}} \sum_{i=1}^{|\mathbf{p}|} c(y_{p_i})\quad + \beta |\mathbf{p}|.

The method is implemented in :class:`ruptures.detection.Pelt` (see :cite:`a-Killick2012a`)

.. available models

.. automodule:: ruptures.costs


.. rubric:: References

.. bibliography:: ../biblio.bib
    :style: alpha
    :cited:
    :labelprefix: A
    :keyprefix: a-

