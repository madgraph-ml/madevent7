from . import _madevent_py_loader as me

from madnis.integrator import Integrand, ChannelGrouping
from madnis.nn import Distribution
import torch
import torch.nn as nn

class IntegrandDistribution(nn.Module, Distribution):
    def __init__(self, channels: list[me.Integrand]):
        pass

    def sample(
        self,
        n: int,
        c: torch.Tensor | None = None,
        channel: torch.Tensor | list[int] | int | None = None,
        return_log_prob: bool = False,
        return_prob: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        raise NotImplementedError(
            "IntegrandDistribution does not support sampling directly. "
            "Use the underlying me.Integrand object instead."
        )

    def prob(
        self,
        x: torch.Tensor,
        c: torch.Tensor | None = None,
        channel: torch.Tensor | list[int] | int | None = None,
    ) -> torch.Tensor:
        pass

def build_madnis_integrand(
    channels: list[me.Integrand],
    cwnet: me.ChannelWeightNetwork | None = None,
    symfact: list[int | None] | None = None,
    context: me.Context = me.default_context(),
) -> tuple[Integrand, Distribution, nn.Module | None]:
    integrand = Integrand(
        function=
        input_dim=channels[0].random_dim(),
        channel_count=len(channels),
        remapped_dim=cwnet.preprocessing().output_dim(),
        has_channel_weight_prior=cwnet is not None,
        channel_grouping=None if symfact is None else ChannelGrouping(symfact),
        function_includes_sampling=True,
        #discrete_dims=
        #discrete_dims_position=
        #discrete_prior_prob_function=
        #discrete_prior_prob_mode=
        #discrete_mode=
    )
    flow = IntegrandDistribution(channels)
    cwnet_module = None if cwnet is None else FunctionModule(cwnet.function(), context)

    return integrand, flow, cwnet_module
