from __future__ import annotations

from typing import TYPE_CHECKING

from collections.abc import Callable
from dataclasses import dataclass

import lightning as lt
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig

if TYPE_CHECKING:
    from torch import optim


@dataclass
class OptimizerConfig:
    """
    Configuration class for the optimizer and scheduler in the LightningModule.
    """

    optimizer_generator: Callable[..., optim.optimizer.Optimizer]
    scheduler_generator: Callable[..., optim.lr_scheduler.LRScheduler] | None = None
    scheduler_interval: str | None = "step"
    scheduler_monitor: str | None = "validation/loss"


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
        optimizer = self.optcfg.optimizer_generator(params=self.parameters())
        if self.optcfg.scheduler_generator is None:
            return optimizer
        else:
            assert self.optcfg.scheduler_interval is not None
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": self.optcfg.scheduler_generator(optimizer=optimizer),
                    "interval": self.optcfg.scheduler_interval,
                    "monitor": self.optcfg.scheduler_monitor,
                },
            }
