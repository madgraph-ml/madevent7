import pytest
from pytest import approx
import madevent7 as me
from madevent7.torch import FunctionModule
import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)

@pytest.fixture
def mlp():
    ctx = me.Context()
    mlp = me.MLP(10, 1, 32, 3, me.MLP.leaky_relu, "")
    mlp.initialize_globals(ctx)
    return FunctionModule(mlp.function(), ctx)

def test_properties():
    mlp = me.MLP(10, 1, 32, 3, me.MLP.leaky_relu, "")
    assert mlp.input_dim() == 10
    assert mlp.output_dim() == 1

def test_initialization(mlp):
    assert torch.all(mlp.global_params["layer1_weight"] != 0)
    assert torch.all(mlp.global_params["layer1_bias"] != 0)
    assert torch.all(mlp.global_params["layer2_weight"] != 0)
    assert torch.all(mlp.global_params["layer2_bias"] != 0)
    assert torch.all(mlp.global_params["layer3_weight"] == 0)
    assert torch.all(mlp.global_params["layer3_bias"] == 0)

def test_training(mlp):
    mlp_torch = nn.Sequential(
        nn.Linear(10,32),
        nn.LeakyReLU(),
        nn.Linear(32,32),
        nn.LeakyReLU(),
        nn.Linear(32,1),
    )

    with torch.no_grad():
        for i in range(3):
            mlp_torch[2*i].weight[:] = mlp.global_params[f"layer{i+1}_weight"][0]
            mlp_torch[2*i].bias[:] = mlp.global_params[f"layer{i+1}_bias"][0]

    opt_me7 = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    opt_torch = torch.optim.Adam(mlp_torch.parameters(), lr=1e-3)

    for i in range(10):
        x = torch.randn((128, 10))

        loss_me7 = (x**2 - mlp(x)[0]).square().mean()
        opt_me7.zero_grad()
        loss_me7.backward()
        opt_me7.step()

        loss_torch = (x**2 - mlp_torch(x)).square().mean()
        opt_torch.zero_grad()
        loss_torch.backward()
        opt_torch.step()

        with torch.no_grad():
            for i in range(3):
                assert mlp_torch[2*i].weight.numpy() == approx(
                    mlp.global_params[f"layer{i+1}_weight"][0].numpy()
                )
                assert mlp_torch[2*i].bias.numpy() == approx(
                    mlp.global_params[f"layer{i+1}_bias"][0].numpy()
                )
                assert mlp_torch[2*i].weight.grad.numpy() == approx(
                    mlp.global_params[f"layer{i+1}_weight"].grad[0].numpy()
                )
                assert mlp_torch[2*i].bias.grad.numpy() == approx(
                    mlp.global_params[f"layer{i+1}_bias"].grad[0].numpy()
                )
