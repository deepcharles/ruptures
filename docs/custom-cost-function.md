# Creating a custom cost function

In order to define custom cost functions, simply create a class that inherits from
`ruptures.base.BaseCost` and implement the methods `.fit(signal)` and `.error(start, end)`:

- The method `.fit(signal)` takes a signal as input and sets parameters. It returns `'self'`.
- The method `.error(start, end)` takes two indexes `'start'` and `'end'`  and returns the cost on the segment start:end.

An example can be found in LINK.
