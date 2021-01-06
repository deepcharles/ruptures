"""Factory function for Cost classes."""

from ruptures.base import BaseCost


def cost_factory(model, *args, **kwargs):
    for cls in BaseCost.__subclasses__():
        if cls.model == model:
            return cls(*args, **kwargs)
    raise ValueError("Not such model: {}".format(model))
