from __future__ import annotations

from typing import TYPE_CHECKING, Any

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from fnmatch import fnmatch

from torch import nn

import lightning as lt
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig

if TYPE_CHECKING:
    from torch import optim


@dataclass
class OptimizerConfig:
    """
    Configuration for the optimizer and scheduler in a LightningModule.

    Args:
        optimizer_generator (Callable[..., optim.optimizer.Optimizer]): A callable that generates the optimizer.
        params_transformer (Callable[..., Iterator[tuple[str, Any]]] | None): A callable that transforms the parameters
            into a format suitable for the optimizer. If None, the parameters are used as is. Defaults to None.
        scheduler_generator (Callable[..., optim.lr_scheduler.LRScheduler] | None):
            A callable that generates the learning rate scheduler. If None, no scheduler is used. Defaults to None.
        scheduler_interval (str | None): The interval at which the scheduler is called. Defaults to "step".
        scheduler_monitor (str | None): The metric to monitor for the scheduler. Defaults to "validation/loss".
    """

    optimizer_generator: Callable[..., optim.optimizer.Optimizer]
    params_transformer: Callable[..., Iterator[tuple[str, Any]]] | None = None

    scheduler_generator: Callable[..., optim.lr_scheduler.LRScheduler] | None = None
    scheduler_interval: str | None = "step"
    scheduler_monitor: str | None = "validation/loss"


def build_param_groups(
    named_params: Iterator[tuple[str, nn.Parameter]],
    groups: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Build parameter groups for the optimizer based on the provided patterns.

    Args:
        named_params (Iterator[tuple[str, nn.Parameter]]): An iterator of named parameters.
        groups (list[dict[str, Any]]): A list of dictionaries where each dictionary contains
            a "pattern" key that specifies the parameter names to match (``fnmatch``), and other optional keys.

    Example:
    In your config file, you might have:

    .. code-block:: yaml

        optimizer_config:
          _target_: aiaccel.torch.lightning.OptimizerConfig
          optimizer_generator:
            _partial_: True
            _target_: torch.optim.AdamW
            weight_decay: 0.01
          param_transformer:
              _partial_: True
              _target_: aiaccel.torch.lightning.build_param_groups
              groups:
                - pattern: "*bias"
                  lr: 0.01
                - pattern: "*weight"
                  lr: 0.001

    This will create two parameter groups: one for biases with a learning rate of 0.01 and another for weights with
    a learning rate of 0.001.
    """
    remaining = dict(named_params)

    param_groups = []
    for spec in groups:
        matched_params = []
        for target in [spec["pattern"]] if isinstance(spec["pattern"], str) else spec["pattern"]:
            matched_params += [remaining.pop(name) for name in list(remaining.keys()) if fnmatch(name, target)]

        assert len(matched_params) > 0

        param_groups.append({"params": matched_params} | {k: v for k, v in spec.items() if k != "pattern"})

    param_groups.append({"params": list(remaining.values())})

    return param_groups


class OptimizerLightningModule(lt.LightningModule):
    """
    LightningModule subclass for models that use custom optimizers and schedulers.

    Args:
        optimizer_config (OptimizerConfig): Configuration object for the optimizer.

    Attributes:
        optcfg (OptimizerConfig): Configuration object for the optimizer.

    Methods:
        configure_optimizers: Configures the optimizer and scheduler for training.
    """

    def __init__(self, optimizer_config: OptimizerConfig):
        super().__init__()

        self.optcfg = optimizer_config

    def configure_optimizers(self) -> optim.optimizer.Optimizer | OptimizerLRSchedulerConfig:
        """
        Configures the optimizer and scheduler for training.

        Returns:
            Union[optim.optimizer.Optimizer, OptimizerLRSchedulerConfig]: The optimizer and scheduler configuration.
        """

        params = self.named_parameters()
        if self.optcfg.params_transformer is not None:
            params = self.optcfg.params_transformer(params)

        optimizer = self.optcfg.optimizer_generator(params)

        if self.optcfg.scheduler_generator is None:
            return optimizer
        else:
            assert self.optcfg.scheduler_interval is not None
            assert self.optcfg.scheduler_monitor is not None
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": self.optcfg.scheduler_generator(optimizer=optimizer),
                    "interval": self.optcfg.scheduler_interval,
                    "monitor": self.optcfg.scheduler_monitor,
                },
            }
