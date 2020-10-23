# Fitting and prediction: estimator basics

`ruptures` has an object-oriented modelling approach (largely inspired by [scikit-learn](https://scikit-learn.org/stable/getting_started.html)): change point detection algorithms are broken down into two conceptual objects that inherits from base classes: `BaseEstimator` and
`BaseCost`.


## Initializing a new estimator

Each change point detection algorithm inherits from the base class `ruptures.base.BaseEstimator`.
When a class that inherits from the base estimator is created, the `.__init__()` method initializes
an estimator with the following arguments:

* `model`: "l1", "l2", "normal", "rbf", "linear", etc. Cost function to use to compute the approximation error.
* `cost`: a custom cost function to the detection algorithm. Should be a `BaseCost` instance.
* `jump`: reduce the set of possible change point indexes; predicted change points can only be a multiple of `jump`.
* `min_size`: minimum number of samples between two change points.

## Making a prediction

The main methods are `.fit()`, `.predict()`, `.fit_predict()`:

- `.fit()`: generally takes a signal as input and fit the algorithm to the data.
- `.predict()`: performs the change point detection. This method returns a list of indexes corresponding to the end of each regimes. By design, the last element of this list is the number of samples.
- ``.fit_predict()``: helper method which calls ``.fit()`` and ``.predict()`` successively.