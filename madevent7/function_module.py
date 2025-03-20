import torch
import torch.nn as nn
from torch.autograd.function import FunctionCtx, once_differentiable
import .madevent_py as me

class FunctionModule(nn.Module):
    def __init__(
        self,
        function: me.Function,
        context: me.Context = me.Context.default_context(),
    ):
        super().__init__()
        context.define_function_globals(function)
        self.global_params = nn.ParameterDict({
            nn.Parameter(
                context.get_global(name).torch(),
                context.global_requires_grad(name),
            )
            for name in function.globals
        })
        self.runtime = me.FunctionRuntime(function, context)

    def forward(self, *args: torch.Tensor) -> list[torch.Tensor]:
        return AutogradWrapper.apply(self, args)


class AutogradWrapper(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx, module: FunctionModule, args: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        if torch.is_grad_enabled():
            return module.runtime.call(args)
        else:
            outputs, local_grads = module.runtime.call_with_grad(args)
            ctx.module = module
            ctx.save_for_backward(local_grads)
            return outputs

    @staticmethod
    @once_differentiable
    def backward(ctx: FunctionCtx, *output_grads: torch.Tensor):
        input_grads, global_grads = ctx.module.runtime.call_backward(
            output_grads, ctx.saved_tensors
        )
        for name, grad in global_grads.items():
            param = ctx.module.global_params[name]
            if param.grad is None:
                param.grad = grad
            else:
                param.grad += grad
        return input_grads

