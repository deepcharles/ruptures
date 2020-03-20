.. _sec-custom-cost:

Custom cost class
====================================================================================================

Users who are interested in detecting a specific type of change can easily do so by creating a custom cost function.
Provided, they use the base cost function :class:`ruptures.base.BaseCost`, they will be able to seamlessly run the algorithms implemented in :mod:`ruptures`.

.. seealso:: :ref:`sec-new-cost`

.. rubric:: Example

Let :math:`\{y_t\}_t` denote a 1D piecewise stationary random process.
Assume that the :math:`y_t` are independent and exponentially distributed with a scale parameter that shifts at some unknown instants :math:`t_1,t_2,\dots`
The change points estimates are the minimizers of the negative log-likelihood, and the associated cost function is given by

.. math::
    c(y_I) = |I| \log \bar{\mu}_I

where :math:`I,\, y_I` and :math:`\bar{\mu}_I` are respectively an interval, the sub-signal on this interval and the empirical mean of this sub-signal.
The following code implements this cost function:

.. code-block:: python

    from math import log
    from ruptures.base import BaseCost
    
    class MyCost(BaseCost):

        """Custom cost for exponential signals."""
        
        # The 2 following attributes must be specified for compatibility.
        model = ""
        min_size = 2 

        def fit(self, signal):
            """Set the internal parameter."""    
            self.signal = signal
            return self

        def error(self, start, end):
            """Return the approximation cost on the segment [start:end].

            Args:
                start (int): start of the segment
                end (int): end of the segment

            Returns:
                float: segment cost
            """
            sub = self.signal[start:end]
            return (end-start)*log(sub.mean())


This cost function can now be used with all algorithms from :mod:`ruptures`.
For instance,

.. code-block:: python

    import numpy as np
    import matplotlib.pylab as plt
    import ruptures as rpt
    # creation of data
    a = np.random.exponential(scale=1, size=100)
    b = np.random.exponential(scale=2, size=200)
    signal, bkps = np.r_[a, b, a], [100, 300, 400]
    # cost
    algo = rpt.Pelt(custom_cost=MyCost()).fit(signal)
    my_bkps = algo.predict(pen=10)
    # display
    rpt.display(signal, bkps, my_bkps)
    plt.show()