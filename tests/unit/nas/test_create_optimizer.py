import pytest
import torch
from torch.nn import Module
from torch.optim import SGD, Adam, NAdam, RMSprop

from aiaccel.nas.create_optimizer import create_optimizer


class DummyModule(Module):
    def parameters(self):
        return [torch.randn(3, 1, requires_grad=True)]


@pytest.mark.parametrize(
    "optimizer_name, optimizer_type",
    [
        ("MomentumSGD", SGD),
        ("NAG", SGD),
        ("RMSprop", RMSprop),
        ("Adam", Adam),
        ("AMSGrad", Adam),
        ("NAdam", NAdam),
    ],
)
def test_create_optimizer(optimizer_name, optimizer_type):
    nn_module = DummyModule()
    hyperparameters = {
        "optimizer": optimizer_name,
        "initial_lr": 0.01,
        "momentum": 0.9,
        "dampening": 0,
        "weight_decay": 0.0005,
        "smoothing": 0.99,
        "eps": 1e-8,
        "beta1": 0.9,
        "beta2": 0.999,
        "momentum_decay": 0.004,
    }
    optimizer = create_optimizer(nn_module, hyperparameters)
    assert isinstance(optimizer, optimizer_type)
