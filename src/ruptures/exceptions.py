"""The :mod:`ruptures.exceptions` module includes all custom warnings and error
classes used across ruptures."""


class NotEnoughPoints(Exception):

    """Raise this exception when there is not enough point to calculate a cost
    function."""

    pass
