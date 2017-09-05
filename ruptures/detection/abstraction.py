from . import METHODS


def changepoint(cls):
    """Decorator to help the creation of changepoint detection classes.
    Decorated classes can be instanciated which Dynp or Pelt (or any method
    from METHODS) as parent class (dynamic inheritance).


    Returns:
        function: decorator
    """
    def decorator(method, *args, **kwargs):
        """Returns an instance of the decorated class inherited from the
        METHODS[method] class

        Args:
            method (str): method key
            *args: various arguments
            **kwargs: various arguments

        Returns:
            class instance:
        """
        assert method in METHODS
        base = METHODS[method]
        wrapped_cls = type(cls.__name__, (cls, base), {})
        return wrapped_cls(*args, **kwargs)
    return decorator
