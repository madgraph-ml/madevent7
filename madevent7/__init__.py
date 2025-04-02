import ctypes
import os
import platform

# pre-load libmadevent
ctypes.CDLL(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "lib",
        "libmadevent.dylib" if platform.system() == "Darwin" else "libmadevent.so"
    ),
    mode=ctypes.RTLD_GLOBAL
)

# pre-load torch
import torch

from ._madevent_py import *
#from .function_module import FunctionModule

def _init():
    """
    Monkey-patch Function and Mapping classes for a more pythonic experience.
    """

    def function_call(self, *args):
        if not hasattr(self, "runtime"):
            self.runtime = FunctionRuntime(self)
        out = self.runtime.call(args)
        if len(out) == 1:
            return out[0]
        else:
            return out

    def function_generator_call(self, *args):
        if not hasattr(self, "runtime"):
            self.runtime = FunctionRuntime(self.function())
        out = self.runtime.call(args)
        if len(out) == 1:
            return out[0]
        else:
            return out

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
    FunctionGenerator.__call__ = function_generator_call
    Mapping.map_forward = map_forward
    Mapping.map_inverse = map_inverse

_init()
