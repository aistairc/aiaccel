# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

import lightning as lt
from lightning.pytorch.utilities import rank_zero_warn


class PrintUnusedParam(lt.Callback):
    """Warn once when trainable parameters do not receive gradients."""

    def __init__(self) -> None:
        super().__init__()
        self._has_warned = False

    def on_after_backward(self, trainer: lt.Trainer, pl_module: lt.LightningModule) -> None:  # type: ignore[override]
        """Emit a warning for parameters that never collected gradients."""
        if self._has_warned or not trainer.is_global_zero:
            return

        for name, param in pl_module.named_parameters():
            if param.requires_grad and param.grad is None:
                rank_zero_warn(f"{name} is unused")

        self._has_warned = True
