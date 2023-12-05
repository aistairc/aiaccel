from __future__ import annotations

import math

import torch

from aiaccel.nas.batch_dependent_lr_scheduler import AbstractBatchDependentLRScheduler


class BatchDependentCosineLRScheduler(AbstractBatchDependentLRScheduler):
    """Scheduler for learning rate which decays by cosine function.

    The phase of the cosine function increases from 0 at the end of warmup to
    pi at the end of epoch scan.

    Args:
        optimizer (torch.optim.Optimizer): Wrapped optimizer.
        num_epochs (int): the number of epochs.
        num_batches (int): the number of batches.
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
        warmup_epochs: int = 5,
        h_size: int = 3,
        last_batch_id: int = -1,
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:
        super().__init__(optimizer, num_epochs, num_batches, warmup_epochs, h_size, last_batch_id, last_epoch, verbose)

        self.num_epochs_after_warmup = float(self.num_epochs - self.warmup_epochs)

    def get_lr_after_warmup(self) -> list[float]:
        effective_step = self.get_effective_step()
        magnification = 0.5 * (1 + math.cos(math.pi * effective_step))
        return [group["initial_lr"] * magnification for group in self.optimizer.param_groups]

    def get_effective_step(self) -> float:
        """Gets effective step which takes 0.0 at the end of warmup epoch and
        1.0 at the end of epoch scan.

        Increment in the batch index by 1 increases the effective step by
        1 / (num_batches*num_epochs_after_warmup).

        Returns:
            float: Effective step.
        """
        epoch_after_warmup = self.last_epoch - self.warmup_epochs
        return (epoch_after_warmup + self.last_batch_id / self.num_batches) / self.num_epochs_after_warmup
