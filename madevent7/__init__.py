import torch
from ._madevent_py import *

def _init():
    """
    Monkey-patch Function and Mapping classes for a more pythonic experience.
    """

    def function_call(self, *args):
        if not hasattr(self, "runtime"):
            self.runtime = FunctionRuntime(self)
        return self.runtime.call(args)

    def map_forward(self, inputs, conditions=[]):
        if not hasattr(self, "forward_runtime"):
            self.forward_runtime = FunctionRuntime(self.forward_function())
        outputs = self.forward_runtime.call([*inputs, *conditions])
        return outputs[:-1], outputs[-1]

    def map_inverse(self, inputs, conditions=[]):
        if not hasattr(self, "inverse_runtime"):
            self.inverse_runtime = FunctionRuntime(self.inverse_function())
        outputs = self.inverse_runtime.call([*inputs, *conditions])
        return outputs[:-1], outputs[-1]

    Function.__call__ = function_call
    Mapping.map_forward = map_forward
    Mapping.map_inverse = map_inverse

_init()
