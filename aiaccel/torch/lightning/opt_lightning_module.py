from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch.optim import Optimizer

import lightning as lt
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig


@dataclass
class OptimizerConfig:
    optimizer_generator: Callable[..., torch.optim.Optimizer]
    scheduler_generator: Callable[..., torch.optim.lr_scheduler.LRScheduler] | None = None
    scheduler_interval: str | None = "step"
    scheduler_monitor: str | None = "validation/loss"


class OptimizerLightningModule(lt.LightningModule):
    def __init__(self, optimizer_config: OptimizerConfig):
        super().__init__()

        self.optcfg = optimizer_config

    def configure_optimizers(self) -> Optimizer | OptimizerLRSchedulerConfig:
        optimizer = self.optcfg.optimizer_generator(params=self.parameters())
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
