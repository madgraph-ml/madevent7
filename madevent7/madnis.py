from typing import Callable

from . import _madevent_py_loader as me
from .torch import FunctionModule

from madnis.integrator import Integrand, ChannelGrouping
from madnis.nn import Distribution
import torch
import torch.nn as nn

MADNIS_INTEGRAND_FLAGS = (
    me.Integrand.sample |
    me.Integrand.return_latent |
    me.Integrand.return_channel |
    me.Integrand.return_chan_weights |
    me.Integrand.return_cwnet_input |
    me.Integrand.return_discrete_latent
)

class IntegrandDistribution(nn.Module, Distribution):
    def __init__(
        self,
        channels: list[me.Integrand],
        channel_remap_function: Callable[[torch.Tensor], torch.Tensor],
        context: me.Context
    ):
        super().__init__()
        self.channel_count = len(channels)
        self.channel_remap_function = channel_remap_function
        self.latent_dims, self.latent_float = channels[0].latent_dims()
        multi_prob = me.MultiChannelFunction(
            [me.IntegrandProbability(chan) for chan in channels]
        )
        self.integrand_prob = FunctionModule(multi_prob.function(), context)

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
        channel_perm = None
        if isinstance(channel, torch.Tensor):
            channel = self.channel_remap_function(channel)
            channel_perm = torch.argsort(channel)
            x = x[channel_perm]
            channel = channel.bincount(minlength=self.channel_count)
        elif channel is None:
            channel = torch.tensor([len(x)])
        else:
            raise NotImplementedError("channel argument type not supported")

        prob_args = [
            xi if is_float else xi[:,0].to(torch.int64)
            for xi, is_float in zip(x.split(self.latent_dims, dim=1), self.latent_float)
        ]
        prob = self.integrand_prob(*prob_args, channel)
        if channel_perm is None:
            return prob
        else:
            channel_perm_inv = torch.argsort(channel_perm)
            return prob[channel_perm_inv]

def build_madnis_integrand(
    channels: list[me.Integrand],
    cwnet: me.ChannelWeightNetwork | None = None,
    channel_grouping: ChannelGrouping | None = None,
    context: me.Context = me.default_context(),
) -> tuple[Integrand, Distribution, nn.Module | None]:
    channel_count = len(channels)
    multi_integrand = me.MultiChannelFunction(channels)
    multi_runtime = me.FunctionRuntime(multi_integrand.function(), context)

    def integrand_function(channels):
        channel_perm = torch.argsort(channels)
        channels = channels.bincount(minlength=channel_count)
        (
            full_weight, latent, inv_prob, chan_index, alphas_prior, alpha_selected, y, *rest
        ) = multi_runtime(channels)

        x_parts = [latent, *rest]
        x = torch.cat(
            [xi.double().reshape(latent.shape[0], -1) for xi in x_parts], dim=1
        )
        prob = 1 / inv_prob
        weight = torch.where(alpha_selected == 0., 0., full_weight * prob / alpha_selected)
        channel_perm_inv = torch.argsort(channel_perm)
        return (
            x[channel_perm_inv],
            prob[channel_perm_inv],
            weight[channel_perm_inv],
            y[channel_perm_inv],
            alphas_prior[channel_perm_inv],
            chan_index[channel_perm_inv]
        )

    def update_mask(mask: torch.Tensor):
        context.get_global(cwnet.mask_name()).torch()[0, :] = mask.double()

    integrand = Integrand(
        function=integrand_function,
        input_dim=sum(channels[0].latent_dims()[0]),
        channel_count=len(channel_grouping.channels),
        remapped_dim=cwnet.preprocessing().output_dim(),
        has_channel_weight_prior=cwnet is not None,
        channel_grouping=channel_grouping,
        function_includes_sampling=True,
        update_active_channels_mask=update_mask,
    )
    flow = IntegrandDistribution(channels, integrand.remap_channels, context)
    cwnet_module = None if cwnet is None else FunctionModule(cwnet.mlp().function(), context)
    return integrand, flow, cwnet_module
