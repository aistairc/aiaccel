"""
In the original function `adjust_learning_rate`, step-wise update of learning
rate is specified by `type="linear"`.
However, this update method is usually called like __multi-step__, for
example, in PyTorch.
Therefore, an alternatively-implemented class corresponding to this function
with `type="linear"` is named as `BatchDependentMultiStepLRScheduler`.

Further, in the original function, computed learning rate changes discontinuely
at the moment when running epoch reaches the `warmup_epoch` if `warmup_epoch`
is grater than the smallest step milestone.
Since this behavior does not seem to be intended,
`BatchDependentMultiStepLRScheduler` is implemented while fixing this behavior.
That is, the learning rate just before and after warmup_epoch does not differ
drastically when using `BatchDependentMultiStepLRScheduler`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiaccel.nas.batch_dependent_lr_scheduler import AbstractBatchDependentLRScheduler

if TYPE_CHECKING:
    import torch


class BatchDependentMultiStepLRScheduler(AbstractBatchDependentLRScheduler):
    """Scheduler for learning rate which decays by the specified rate "gamma"
    once the number of epoch reaches one of the milestones.

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
        milestones: list[int],
        num_epochs: int,
        num_batches: int,
        gamma=0.1,
        warmup_epochs: int = 5,
        h_size: int = 3,
        last_batch_id: int = -1,
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:
        super().__init__(optimizer, num_epochs, num_batches, warmup_epochs, h_size, last_batch_id, last_epoch, verbose)
        self.milestones = milestones
        self.gamma = gamma

    def get_lr_after_warmup(self) -> list[float]:
        if self.last_epoch not in self.milestones:
            return [group["lr"] for group in self.optimizer.param_groups]
        return [
            group["initial_lr"] * self.gamma ** (self.milestones.index(self.last_epoch) + 1)
            for group in self.optimizer.param_groups
        ]
