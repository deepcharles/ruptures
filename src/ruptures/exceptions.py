"""The `ruptures.exceptions` module includes all custom warnings and error
classes used across ruptures."""

__all__ = ["BadSegmentationParameters", "NotEnoughPoints"]


class NotEnoughPoints(Exception):
    """Raise this exception when there is not enough point to calculate a cost
    function."""


class BadSegmentationParameters(Exception):
    """Raise this exception when a segmentation is not possible given the
    parameters."""
