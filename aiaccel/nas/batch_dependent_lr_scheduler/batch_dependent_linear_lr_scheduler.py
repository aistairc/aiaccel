from __future__ import annotations

from typing import TYPE_CHECKING

from aiaccel.nas.batch_dependent_lr_scheduler import AbstractBatchDependentLRScheduler

if TYPE_CHECKING:
    import torch


class BatchDependentLinearLRScheduler(AbstractBatchDependentLRScheduler):
    """Scheduler for learning rate which decays linearly.

    Args:
        optimizer (torch.optim.Optimizer): Wrapped optimizer.
        num_epochs (int): the number of epochs.
        num_batches (int): the number of batches.
        start_factor (float): The number we multiply learning rate in the first
            epoch. The multiplication factor changes towards end_factor in the
            following epochs. Default: 1./3.
        end_factor (float): The number we multiply learning rate at the end of
            linear changing process. Default: 1.0.
        total_epochs (int): The number of epochs that multiplicative factor
            reaches to 1. Default: 5.
        warmup_epochs (int, optional): the number of epoches in which the
            learning rate is monotonically warmed up. Defaults to 5.
        h_size (int, optional): . Defaults to 3.
        last_batch_id (int, optional): The index of last batch. Defaults to -1.
        last_epoch (int, optional): The index of last epoch. Defaults to -1.
        verbose (bool, optional): If True, prints a message to stdout for each
            update per epoch. Defaults to False.

    Example:
        ::

            for epoch in range(num_epochs)]
                for batch_id, (x, label) in enumerate(dataloader):
                    scheduler.batch_step()
                    ...
                optimizer.step()
                scheduler.step()

    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_epochs: int,
        num_batches: int,
        start_factor: float,
        end_factor: float,
        total_epochs: int = 5,
        warmup_epochs: int = 5,
        h_size: int = 3,
        last_batch_id: int = -1,
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:
        super().__init__(optimizer, num_epochs, num_batches, warmup_epochs, h_size, last_batch_id, last_epoch, verbose)
        self.start_factor = start_factor
        self.end_factor = end_factor
        self.diff_factor = self.end_factor - self.start_factor
        self.total_epochs = total_epochs
        self.total_iter = self.total_epochs * self.num_batches

    def get_lr_after_warmup(self) -> list[float]:
        if self.last_epoch == 0:
            return [group["lr"] * self.start_factor for group in self.optimizer.param_groups]
        if self.last_epoch > self.total_epochs:
            return [group["lr"] for group in self.optimizer.param_groups]
        return [
            group["lr"]
            * (
                1.0
                + self.diff_factor
                / (
                    self.total_iter * self.start_factor
                    + (self.last_epoch * self.num_batches + self.last_batch_id - 1) * self.diff_factor
                )
            )
            for group in self.optimizer.param_groups
        ]
