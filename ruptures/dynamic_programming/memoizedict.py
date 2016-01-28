from collections import namedtuple
from inspect import signature, _empty


class MemoizeDict(dict):
    """A function decorated with MemoizeDict CAN accept named variables.
    """

    def __init__(self, func, omit=[]):
        self.func = func
        self.omit = omit

        sig = signature(self.func)
        self.params = list(sig.parameters)
        arguments = namedtuple('arguments', self.params, verbose=False)
        defaults = [v.default for k, v in sig.parameters.items()]
        self.defaults = [None if v == _empty else v for v in defaults]
        arguments.__new__.__defaults__ = tuple(self.defaults)
        self.arguments = arguments

    def __call__(self, *args, **kwargs):
        param_dict = dict(zip(self.params, args), **kwargs)
        key = self.arguments(**param_dict)
        return self[key]

    def __missing__(self, key):
        result = self[key] = self.func(*key)
        return result
