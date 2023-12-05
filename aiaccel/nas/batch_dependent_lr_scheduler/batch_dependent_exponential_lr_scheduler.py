from __future__ import annotations

import torch

from aiaccel.nas.batch_dependent_lr_scheduler import AbstractBatchDependentLRScheduler


class BatchDependentExponentialLRScheduler(AbstractBatchDependentLRScheduler):
    """Scheduler for learning rate which decays by "gamma" every batch.

    Args:
        optimizer (torch.optim.Optimizer): Wrapped optimizer.
        num_epochs (int): the number of epochs.
        num_batches (int): the number of batches.
        gamma (float): Multiplicative factor of learning rate decay.
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
        gamma: float,
        warmup_epochs: int = 5,
        h_size: int = 3,
        last_batch_id: int = -1,
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:
        super().__init__(optimizer, num_epochs, num_batches, warmup_epochs, h_size, last_batch_id, last_epoch, verbose)
        self.gamma = gamma ** (1 / self.num_batches)

    def get_lr_after_warmup(self) -> list[float]:
        if self.last_epoch == 0:
            return [group["lr"] for group in self.optimizer.param_groups]
        return [group["lr"] * self.gamma for group in self.optimizer.param_groups]
