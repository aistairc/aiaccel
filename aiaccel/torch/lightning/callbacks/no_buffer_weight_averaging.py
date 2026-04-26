# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from typing import TYPE_CHECKING, Any

import torch

from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import Callback


def _missing_weight_averaging_api() -> ImportError:
    return ImportError(
        "NoBufferWeightAveraging requires lightning.pytorch.callbacks.WeightAveraging "
        "and EMAWeightAveraging. Please install a Lightning version that provides "
        "these callbacks."
    )


if TYPE_CHECKING:

    class _WeightAveraging(Callback):
        def __init__(
            self,
            device: torch.device | str | int | None = None,
            use_buffers: bool = False,
            **kwargs: Any,
        ) -> None: ...

    class _EMAWeightAveraging(Callback):
        def __init__(
            self,
            device: torch.device | str | int | None = None,
            use_buffers: bool = False,
            decay: float = 0.999,
            update_every_n_steps: int = 1,
            update_starting_at_step: int | None = None,
            update_starting_at_epoch: int | None = None,
            **kwargs: Any,
        ) -> None: ...

else:
    try:
        from lightning.pytorch.callbacks import EMAWeightAveraging as _EMAWeightAveraging
        from lightning.pytorch.callbacks import WeightAveraging as _WeightAveraging
    except ImportError:

        class _WeightAveraging(Callback):  # type: ignore[no-redef]
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                raise _missing_weight_averaging_api()

        class _EMAWeightAveraging(Callback):  # type: ignore[no-redef]
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                raise _missing_weight_averaging_api()


class _NoBufferWeightAveragingMixin:
    """Shared behavior for weight averaging callbacks that must ignore buffers."""

    _average_model: Any | None

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
        super().on_save_checkpoint(trainer, pl_module, checkpoint)  # type: ignore[misc]

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


class NoBufferWeightAveraging(_NoBufferWeightAveragingMixin, _WeightAveraging):
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


class NoBufferEMAWeightAveraging(_NoBufferWeightAveragingMixin, _EMAWeightAveraging):
    """Exponential moving average (EMA) callback that ignores buffers."""

    def __init__(
        self,
        device: torch.device | str | int | None = None,
        decay: float = 0.999,
        update_every_n_steps: int = 1,
        update_starting_at_step: int | None = None,
        update_starting_at_epoch: int | None = None,
        **kwargs: Any,
    ):
        """Initialize the callback.

        Args:
            device: Device that stores the averaged model. If ``None``, the
                current model device is used.
            decay: Decay factor for the exponential moving average. Should be between
                0 and 1. Default is 0.999.
            update_every_n_steps: Update EMA every N optimizer steps.
            update_starting_at_step: Start EMA updates at or after this optimizer step.
            update_starting_at_epoch: Start EMA updates at or after this epoch.
            **kwargs: Additional arguments forwarded to
                ``lightning.pytorch.callbacks.EMAWeightAveraging``. ``use_buffers``
                is always fixed to ``False`` in this subclass.
        """
        super().__init__(
            device=device,
            use_buffers=False,
            decay=decay,
            update_every_n_steps=update_every_n_steps,
            update_starting_at_step=update_starting_at_step,
            update_starting_at_epoch=update_starting_at_epoch,
            **kwargs,
        )
