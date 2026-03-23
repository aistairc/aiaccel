# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from typing import Any

import torch
from torch.optim.swa_utils import get_ema_avg_fn

from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import WeightAveraging


class NoBufferWeightAveraging(WeightAveraging):
    """Weight averaging callback that ignores buffers during averaging and swapping.

    Checkpoints still store the current model buffers together with the averaged
    parameters so loading preserves non-averaged buffer state.

    Example:
        >>> from lightning.pytorch import Trainer
        >>> from torch.optim.swa_utils import get_ema_avg_fn
        >>> ema = NoBufferWeightAveraging(avg_fn=get_ema_avg_fn(0.999))
        >>> trainer = Trainer(callbacks=[ema])
    """

    def __init__(
        self,
        device: torch.device | str | int | None = None,
        **kwargs: Any,
    ):
        """Initialize the callback.

        Args:
            device: Device that stores the averaged model. If ``None``, the
                current model device is used.
            **kwargs: Additional arguments forwarded to
                ``lightning.pytorch.callbacks.WeightAveraging``. ``use_buffers``
                is always fixed to ``False`` in this subclass.
        """
        super().__init__(device, use_buffers=False, **kwargs)

    def _swap_models(self, pl_module: LightningModule) -> None:
        assert self._average_model is not None

        for average_param, current_param in zip(
            self._average_model.module.parameters(),
            pl_module.parameters(),
            strict=True,
        ):
            tmp = average_param.data.clone()
            average_param.data.copy_(current_param.data)
            current_param.data.copy_(tmp)

    def _copy_average_to_current(self, pl_module: LightningModule) -> None:
        assert self._average_model is not None

        for average_param, current_param in zip(
            self._average_model.module.parameters(),
            pl_module.parameters(),
            strict=True,
        ):
            current_param.data.copy_(average_param.data)

    def on_save_checkpoint(
        self,
        trainer: Any,
        pl_module: LightningModule,
        checkpoint: dict[str, Any],
    ) -> None:
        super().on_save_checkpoint(trainer, pl_module, checkpoint)

        if self._average_model is None:
            return

        current_model_state: dict[str, torch.Tensor] | None = checkpoint.get("current_model_state")
        average_model_state: dict[str, torch.Tensor] | None = checkpoint.get("state_dict")
        if current_model_state is None or average_model_state is None:
            return

        # Save averaged parameters together with the current model buffers.
        buffer_names = {name for name, _ in pl_module.named_buffers()}
        for buffer_name in buffer_names:
            if buffer_name in current_model_state:
                average_model_state[buffer_name] = current_model_state[buffer_name].clone()


class NoBufferEMAWeightAveraging(NoBufferWeightAveraging):
    """Exponential moving average (EMA) version of ``NoBufferWeightAveraging``."""

    def __init__(
        self,
        device: torch.device | str | int | None = None,
        decay: float = 0.999,
        **kwargs: Any,
    ):
        """Initialize the callback.

        Args:
                take two models as input and update the first model in-place.
            device: Device that stores the averaged model. If ``None``, the
                current model device is used.
            decay: Decay factor for the exponential moving average. Should be between
                0 and 1. Default is 0.999.
            **kwargs: Additional arguments forwarded to
                ``lightning.pytorch.callbacks.WeightAveraging``. ``use_buffers``
                is always fixed to ``False`` in this subclass.
        """

        super().__init__(device, avg_fn=get_ema_avg_fn(decay), **kwargs)  # type: ignore[no-untyped-call]
