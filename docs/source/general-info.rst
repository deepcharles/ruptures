License
----------------------------------------------------------------------------------------------------

This project is under :ref:`BSD license <license>`.


Installation
----------------------------------------------------------------------------------------------------

With `pip3 <https://pypi.python.org/pypi/pip>`_ from terminal: ``$ pip3 install ruptures``.

Or download the source codes from `latest release <https://reine.cmla.ens-cachan.fr/c.truong/ruptures/repository/latest/archive.zip>`_ and run the following lines from inside the folder ``$ python3 setup.py install`` or ``$ python3 setup.py develop``.


User guide
----------------------------------------------------------------------------------------------------

This section explains how to use implemented algorithms.
:mod:`ruptures` has an object-oriented modelling approach: change point detection algorithms are
broken down into two conceptual objects that inherits from base classes: :class:`BaseEstimator` and 
:class:`BaseCost`.


Initializing a new estimator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each change point detection algorithm inherits from the base class :class:`ruptures.base.BaseEstimator`.
When a class that inherits from the base estimator is created, the ``.__init__()`` method initializes
an estimator with the following arguments:

* ``'model'``: "l1", "l2", "normal", "rbf", "linear", "ar". Cost function to use to compute the approximation error.
* ``'cost'``: a custom cost function to the detection algorithm. Should be a :class:`BaseCost` instance.
* ``'jump'``: reduce the set of possible change point indexes; predicted change points can only be a multiple of ``'jump'``.
* ``'min_size'``: minimum number of samples between two change points.

Making a prediction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The main methods are ``.fit()``, ``.predict()``, ``.fit_predict()``:

- ``.fit()``: generally takes a signal as input and fit the algorithm on the data
- ``.predict()``: performs the change point detection. This method returns a list of indexes corresponding to the end of each regimes. By design, the last element of this list is the number of samples.
- ``.fit_predict()``: helper method which calls ``.fit()`` and ``.predict()`` successively.


.. _sec-new-cost:

Creating a new cost function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In order to define custom cost functions, simply create a class that inherits from 
:class:`ruptures.base.BaseCost` and implement the methods ``.fit(signal)`` and ``.error(start, end)``:

- The method ``.fit(signal)`` takes a signal as input and sets parameters. It returns ``'self'``.
- The method ``.error(start, end)`` takes two indexes ``'start'`` and ``'end'``  and returns the cost on the segment start:end.

An example can be found in :ref:`sec-custom-cost`.

.. Tutorials
    ----------------------------------------------------------------------------------------------------
    Advanced tutorials are created as Jupyter notebooks. You can find them in **LINK**.

