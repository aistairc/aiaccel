from __future__ import annotations

from typing import TYPE_CHECKING

from torch.optim.lr_scheduler import LRScheduler

if TYPE_CHECKING:
    from torch.optim import Optimizer


class AbstractBatchDependentLRScheduler(LRScheduler):
    """An Abstract class for scheduler for learning rate depending on batch.

    Args:
        optimizer (torch.optim.Optimizer): Wrapped optimizer.
        num_epochs (int): the number of epochs.
        num_batches (int): the number of batches.
        warmup_epochs (int, optional): the number of epoches in which the
            learning rate is monotonically warmed up. Defaults to 5.
        h_size (int, optional): _description_. Defaults to 3.
        last_batch_id (int, optional): The index of last batch. Defaults to -1.
        last_epoch (int, optional): The index of last epoch. Defaults to -1.
        verbose (bool, optional): If True, prints a message to stdout for each
            update per epoch. Defaults to False.

    Example::

            for epoch in range(num_epochs):
                for batch_id, (x, label) in enumerate(dataloader):
                    scheduler.batch_step()
                    ...
                optimizer.step()
                scheduler.step()

    """

    def __init__(
        self,
        optimizer: Optimizer,
        num_epochs: int,
        num_batches: int,
        warmup_epochs: int = 5,
        h_size: int = 3,
        last_batch_id: int = -1,
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:
        self._initial_batch_id = last_batch_id
        self.num_epochs = num_epochs
        self.num_batches = num_batches
        self.warmup_epochs = warmup_epochs
        self.h_size = h_size
        self.last_batch_id = last_batch_id

        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.is_before_warmup():
            return self.get_lr_before_warmup()
        return self.get_lr_after_warmup()

    def step(self, epoch: int | None = None) -> None:
        """Updates internal variables.

        This method does not update the learning rate and should be called per
        epoch.

        Args:
            epoch (int | None, optional): The index of epoch. Defaults to
            None.
        """
        self.last_batch_id = self._initial_batch_id
        self._step_count += 1

        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch

        self._last_lr = []
        for i, lr in enumerate(self.optimizer.param_groups):
            self._last_lr.append(lr)
            self.print_lr(self.verbose, i, lr, epoch)

    def is_before_warmup(self) -> bool:
        """Checks the epoch is in the warmup sequence.

        Returns:
            bool: True if the epoch is in the warmup sequence.
        """
        return self.last_epoch < self.warmup_epochs

    def get_lr_before_warmup(self) -> list[float]:
        """Gets list of learning rates in the warmup sequence.

        Returns:
            list[float]: A list of learning rates.
        """
        magnification = (
            1
            / self.h_size
            * (
                (self.last_epoch + float(self.last_batch_id + 1) / self.num_batches)
                * (self.h_size - 1)
                / self.warmup_epochs
                + 1
            )
        )
        return [magnification * group["initial_lr"] for group in self.optimizer.param_groups]

    def get_lr_after_warmup(self) -> list[float]:
        raise NotImplementedError

    def batch_step(self, batch_id: int | None = None) -> None:
        """Updates internal variables and learning rates.

        This method should be called per batch.

        Args:
            batch_id (int | None, optional): The index of batch. Defaults to
            None.
        """
        if batch_id is None:
            self.last_batch_id += 1
        else:
            self.last_batch_id = batch_id
        values = self.get_lr()

        for param_group, lr in zip(self.optimizer.param_groups, values):
            param_group["lr"] = lr
