from __future__ import annotations

from sys import float_info
from typing import TYPE_CHECKING

from torch.optim import SGD, Adam, NAdam, RMSprop

if TYPE_CHECKING:
    from torch.nn import Module

    from aiaccel.nas.utils.utils import ParameterType


def create_optimizer(
    nn_module: Module,
    hyperparameters: dict[str, ParameterType],
) -> SGD | RMSprop | Adam | NAdam:
    """Creates optimizer used in supernet train.

    Valid optimzier names are: MomentumSGD, NAG, RMSprop, Adam, AMSGrad, and
    NAdam.

    Args:
        nn_model (nn.Module): A netwark model.
        hyperparameters (dict[str, ParameterType]): A configuration
            dict of optimizer which specifies optimizer name and required
            parameters.

    Returns:
        Optimizer: An optimizer object.
    """
    if hyperparameters["optimizer"] == "MomentumSGD":
        optimizer = SGD(
            params=nn_module.parameters(),
            lr=hyperparameters["initial_lr"],
            momentum=hyperparameters["momentum"],
            dampening=hyperparameters["dampening"],
            weight_decay=hyperparameters["weight_decay"],
            nesterov=False,
        )
    elif hyperparameters["optimizer"] == "NAG":
        optimizer = SGD(
            params=nn_module.parameters(),
            lr=hyperparameters["initial_lr"],
            momentum=hyperparameters["momentum"] if hyperparameters["momentum"] != 0 else float_info.epsilon,
            dampening=0,
            weight_decay=hyperparameters["weight_decay"],
            nesterov=True,
        )
    elif hyperparameters["optimizer"] == "RMSprop":
        optimizer = RMSprop(
            params=nn_module.parameters(),
            lr=hyperparameters["initial_lr"],
            alpha=hyperparameters["smoothing"],
            eps=hyperparameters["eps"],
            weight_decay=hyperparameters["weight_decay"],
            momentum=hyperparameters["momentum"],
        )
    elif hyperparameters["optimizer"] == "Adam":
        optimizer = Adam(
            params=nn_module.parameters(),
            lr=hyperparameters["initial_lr"],
            betas=(hyperparameters["beta1"], hyperparameters["beta2"]),
            eps=hyperparameters["eps"],
            weight_decay=hyperparameters["weight_decay"],
            amsgrad=False,
        )
    elif hyperparameters["optimizer"] == "AMSGrad":
        optimizer = Adam(
            params=nn_module.parameters(),
            lr=hyperparameters["initial_lr"],
            betas=(hyperparameters["beta1"], hyperparameters["beta2"]),
            eps=hyperparameters["eps"],
            weight_decay=hyperparameters["weight_decay"],
            amsgrad=True,
        )
    else:
        assert hyperparameters["optimizer"] == "NAdam"
        optimizer = NAdam(
            params=nn_module.parameters(),
            lr=hyperparameters["initial_lr"],
            betas=(hyperparameters["beta1"], hyperparameters["beta2"]),
            eps=hyperparameters["eps"],
            weight_decay=hyperparameters["weight_decay"],
            momentum_decay=hyperparameters["momentum_decay"],
        )
    return optimizer
