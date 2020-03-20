.. ruptures documentation master file, created by
   sphinx-quickstart on Tue Sep  5 13:51:56 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to :mod:`ruptures`
====================================================================================================

:mod:`ruptures` is designed to perform offline change point algorithms within the Python language.
Also in this library, new methods are presented.

.. note::

    .. code-block:: python
        :caption: Basic usage   

        import matplotlib.pyplot as plt
        import ruptures as rpt
        # generate signal
        n_samples, dim, sigma = 1000, 3, 4
        n_bkps = 4  # number of breakpoints
        signal, bkps = rpt.pw_constant(n_samples, dim, n_bkps, noise_std=sigma)
        # detection
        algo = rpt.Pelt(model="rbf").fit(signal)
        result = algo.predict(pen=10)
        # display
        rpt.display(signal, bkps, result)
        plt.show()  

Getting started
====================================================================================================

.. toctree::
    :maxdepth: 1
    :titlesonly:

    general-info


Documentation
====================================================================================================

The complete documentation can be found here.

.. toctree::
    :maxdepth: 1
    :titlesonly:

    cpd
    detection/index
    costs/index
    datasets/index
    metrics/index


How to cite
====================================================================================================

If you use ruptures in a scientific publication, we would appreciate citations to the following paper:

- C. Truong, L. Oudre, N. Vayatis. Selective review of offline change point detection methods. Signal Processing, 167:107299, 2020. `[journal] <https://doi.org/10.1016/j.sigpro.2019.107299>`_ `[pdf] <http://www.laurentoudre.fr/publis/TOG-SP-19.pdf>`_


Contact
====================================================================================================

Charles Truong.


Indices and tables
====================================================================================================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
