# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from collections.abc import Callable

import torch


class SequentialLR(torch.optim.lr_scheduler.SequentialLR):
    """
    A wrapper of torch.optim.lr_scheduler.SequentialLR to use list of functions
    to create schedulers.

    Args:
        optimizer: Optimizer.
        schedulers_fn: List of functions to create schedulers.
        milestones: List of epoch indices. Must be increasing.

    ... code-block:: yaml

        scheduler_generator:
          _partial_: True
          _convert_: "all"
          _target_: aiaccel.lr_schedulers.SequentialLR
          schedulers_fn:
            - _target_: torch.optim.lr_scheduler.LinearLR
              _partial_: True
              start_factor: 1.e-3
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
