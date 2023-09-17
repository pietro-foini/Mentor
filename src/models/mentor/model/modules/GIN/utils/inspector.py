import inspect
from collections import OrderedDict


class Inspector(object):
    """Inspector class."""

    def __init__(self, base_class):
        self.base_class = base_class
        self.params = {}

    def inspect(self, func, pop_first=False):
        params = inspect.signature(func).parameters
        params = OrderedDict(params)
        if pop_first:
            params.popitem(last=False)
        self.params[func.__name__] = params

    def keys(self, func_names=None):
        keys = []
        for func in func_names or list(self.params.keys()):
            keys += self.params[func].keys()
        return set(keys)

    def __implements__(self, cls, func_name):
        if cls.__name__ == "MessagePassing":
            return False
        if func_name in cls.__dict__.keys():
            return True
        return any(self.__implements__(c, func_name) for c in cls.__bases__)

    def implements(self, func_name):
        return self.__implements__(self.base_class.__class__, func_name)

    def distribute(self, func_name, kwargs):
        out = {}
        for key, param in self.params[func_name].items():
            data = kwargs.get(key, inspect.Parameter.empty)
            if data is inspect.Parameter.empty:
                if param.default is inspect.Parameter.empty:
                    raise TypeError(f"Required parameter {key} is empty.")
                data = param.default
            out[key] = data
        return out
