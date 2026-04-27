# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from collections.abc import Callable

import torch


class SequentialLR(torch.optim.lr_scheduler.SequentialLR):
    """
    Build a sequential learning rate scheduler from scheduler factory functions.

    This wrapper makes it easier to define multiple schedulers in Hydra or
    OmegaConf-based configurations, where each entry in ``schedulers_fn`` is a
    partially configured callable that receives ``optimizer`` and returns a
    scheduler instance.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer passed to each scheduler factory.
        schedulers_fn (list[Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler]]):
            Factory functions that create schedulers for ``optimizer``.
        milestones (list[int]): Epoch indices at which to switch to the next scheduler.

    Example:
        .. code-block:: yaml

            scheduler_generator:
              _partial_: True
              _convert_: all
              _target_: aiaccel.torch.lr_schedulers.SequentialLR
              schedulers_fn:
                - _target_: torch.optim.lr_scheduler.LinearLR
                  _partial_: True
                  start_factor: 1.0e-3
                  end_factor: 1.0
                  total_iters: 5000
                - _target_: torch.optim.lr_scheduler.CosineAnnealingLR
                  _partial_: True
                  T_max: 95000
              milestones: [5000]
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        schedulers_fn: list[Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler]],
        milestones: list[int],
    ):
        super().__init__(optimizer, [fn(optimizer) for fn in schedulers_fn], milestones)
