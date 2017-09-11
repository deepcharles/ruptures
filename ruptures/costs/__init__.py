r"""
.. _sec-costs:

****************************************************************************************************
Available models
****************************************************************************************************

- Squared residuals:

    .. math:: c(y_{p_i}) = \sum_{t\in p_i} \|y_t - \bar{y}\|^2_2

    where :math:`\bar{y}=\frac{1}{|p_i|} \sum\limits_{t\in p_i} y_t`.

    This cost function is suited to approximate piecewise constant signals corrupted with noise.

- Absolute deviation:
    .. math:: c(y_{p_i}) = \min_u \sum_{t\in p_i} \|y_t - u\|_1

    This cost function is suited to approximate piecewise constant signals corrupted with
    non-Gaussian noise (following for instance a heavy-tailed distribution).

- Gaussian time series
    Negative maximum log-likelihood (Gaussian density):

    .. math:: c(y_{p_i}) = |p_i| \log\det\widehat{\Sigma}_i

    where :math:`\widehat{\Sigma} = \frac{1}{|p_i|}\sum\limits_{t\in p_i} (y_t - \bar{y}) (y_t - \bar{y})^T`.

    This cost function is suited to approximate piecewise i.i.d. Gaussian variables, for instance
    mean-shifts and scale-shifts.

- Structural changes:

- Piecewise autoregressive model:

- Kernel change point detection:
    Gaussian kernel, radial basis function (rbf).


.. note::

   To specify a custom cost function, use :class:`ruptures.BaseCost`.

"""

from ruptures.exceptions import NotEnoughPoints
from .factory import cost_factory
from .costl1 import CostL1
from .costl2 import CostL2
from .costlinear import CostLinear
from .costrbf import CostRbf
from .costnormal import CostNormal
