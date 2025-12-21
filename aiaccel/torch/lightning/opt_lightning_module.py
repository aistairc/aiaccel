# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import Any

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from fnmatch import fnmatch

from torch import nn, optim

import lightning as lt
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig


@dataclass
class LRSchedulerConfig:
    """
    Configuration for a learning rate scheduler in Lightning.

    Args:
        generator (Callable[..., optim.lr_scheduler.LRScheduler]): A callable that generates the scheduler.
        interval (str | None): Timing to call ``scheduler.step`` (``"step"`` or ``"epoch"``). Defaults to ``"step"``.
        monitor (str | None): Metric to monitor (required for ``ReduceLROnPlateau``). Defaults to ``"validation/loss"``.
        frequency (int | None): How often to call the scheduler. Defaults to None (Lightning default of 1).
        reduce_on_plateau (bool | None): Whether the scheduler is ``ReduceLROnPlateau``. If None, Lightning infers
            it from the scheduler type.
        strict (bool | None): Whether to raise if ``monitor`` is missing. Mirrors Lightning's ``strict`` flag.
        name (str | None): Optional name for logging.
    """

    generator: Callable[..., optim.lr_scheduler.LRScheduler]
    interval: str | None = "step"
    monitor: str | None = "validation/loss"
    frequency: int | None = None
    reduce_on_plateau: bool | None = None
    strict: bool | None = None
    name: str | None = None

    def build(self, optimizer: optim.Optimizer) -> dict[str, Any]:
        """Create a Lightning-compatible scheduler config."""

        if self.interval is not None and self.interval not in {"step", "epoch"}:
            raise ValueError(f"interval must be 'step' or 'epoch', got {self.interval!r}")

        scheduler = self.generator(optimizer=optimizer)

        if self.monitor is None and isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            raise ValueError("monitor must be set when using ReduceLROnPlateau")
        reduce_on_plateau = self.reduce_on_plateau
        if reduce_on_plateau is None and isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            reduce_on_plateau = True

        config: dict[str, Any] = {"scheduler": scheduler}

        if self.interval is not None:
            config["interval"] = self.interval
        if self.frequency is not None:
            config["frequency"] = self.frequency
        if reduce_on_plateau is not None:
            config["reduce_on_plateau"] = reduce_on_plateau
        if self.monitor is not None:
            config["monitor"] = self.monitor
        if self.strict is not None:
            config["strict"] = self.strict
        if self.name is not None:
            config["name"] = self.name

        return config


@dataclass
class OptimizerConfig:
    """
    Configuration for the optimizer and scheduler in a LightningModule.

    Args:
        optimizer_generator (Callable[..., optim.Optimizer]): A callable that generates the optimizer.
        params_transformer (Callable[..., Iterator[tuple[str, Any]]] | None): A callable that transforms the parameters
            into a format suitable for the optimizer. If None, the parameters are used as is. Defaults to None.
        scheduler_generator (Callable[..., optim.lr_scheduler.LRScheduler] | None):
            (Deprecated) A callable that generates a single learning rate scheduler. If None, no scheduler is used.
            ``schedulers`` is preferred when you need multiple schedulers. Defaults to None.
        scheduler_interval (str | None): (Deprecated) The interval at which the single scheduler is called. Defaults
            to ``"step"``.
        scheduler_monitor (str | None): (Deprecated) The metric to monitor for the single scheduler. Defaults to
            ``"validation/loss"``.
        schedulers (list[LRSchedulerConfig]): A list of scheduler configurations, allowing multiple schedulers with
            different intervals or monitors. Defaults to an empty list.
    """

    optimizer_generator: Callable[..., optim.Optimizer]
    params_transformer: Callable[..., Iterator[tuple[str, Any]]] | None = None

    scheduler_generator: Callable[..., optim.lr_scheduler.LRScheduler] | None = None
    scheduler_interval: str | None = "step"
    scheduler_monitor: str | None = "validation/loss"
    schedulers: list[LRSchedulerConfig] = field(default_factory=list)

    def __post_init__(self) -> None:
        schedulers: list[LRSchedulerConfig | Mapping[str, Any]] = list(self.schedulers)

        if self.scheduler_generator is not None:
            if schedulers:
                raise ValueError(
                    "Use either scheduler_generator/scheduler_interval/scheduler_monitor or schedulers, not both."
                )
            if self.scheduler_interval is None or self.scheduler_monitor is None:
                raise ValueError("scheduler_interval and scheduler_monitor must be set when using scheduler_generator")

            schedulers = [
                LRSchedulerConfig(
                    generator=self.scheduler_generator,
                    interval=self.scheduler_interval,
                    monitor=self.scheduler_monitor,
                )
            ]

        normalized: list[LRSchedulerConfig] = []
        for cfg in schedulers:
            if isinstance(cfg, LRSchedulerConfig):
                normalized.append(cfg)
            elif isinstance(cfg, Mapping):
                normalized.append(LRSchedulerConfig(**cfg))
            else:
                raise TypeError(f"Unsupported scheduler config type: {type(cfg)!r}")

        self.schedulers = normalized


def build_param_groups(
    named_params: Iterator[tuple[str, nn.Parameter]],
    groups: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Build parameter groups for the optimizer based on the provided patterns.

    Args:
        named_params (Iterator[tuple[str, nn.Parameter]]): An iterator of named parameters.
        groups (list[dict[str, Any]]): A list of dictionaries where each dictionary contains
            a "pattern" key that specifies the parameter names to match (``fnmatch``), and other optional keys.

    Example:
    In your config file, you might have:

    .. code-block:: yaml

        optimizer_config:
          _target_: aiaccel.torch.lightning.OptimizerConfig
          optimizer_generator:
            _partial_: True
            _target_: torch.optim.AdamW
            weight_decay: 0.01
          params_transformer:
              _partial_: True
              _target_: aiaccel.torch.lightning.build_param_groups
              groups:
                - pattern: "*bias"
                  lr: 0.01
                - pattern: "*weight"
                  lr: 0.001

    This will create two parameter groups: one for biases with a learning rate of 0.01 and another for weights with
    a learning rate of 0.001.
    """
    remaining = dict(named_params)

    param_groups = []
    for spec in groups:
        matched_params = []
        for target in [spec["pattern"]] if isinstance(spec["pattern"], str) else spec["pattern"]:
            matched_params += [remaining.pop(name) for name in list(remaining.keys()) if fnmatch(name, target)]

        assert len(matched_params) > 0

        param_groups.append({"params": matched_params} | {k: v for k, v in spec.items() if k != "pattern"})

    param_groups.append({"params": list(remaining.values())})

    return param_groups


class OptimizerLightningModule(lt.LightningModule):
    """
    LightningModule subclass for models that use custom optimizers and schedulers.

    Args:
        optimizer_config (OptimizerConfig): Configuration object for the optimizer.

    Attributes:
        optcfg (OptimizerConfig): Configuration object for the optimizer.

    Methods:
        configure_optimizers: Configures the optimizer and scheduler for training.
    """

    def __init__(self, optimizer_config: OptimizerConfig):
        super().__init__()

        self._optimizer_config = optimizer_config

    def configure_optimizers(self) -> optim.Optimizer | OptimizerLRSchedulerConfig:
        """
        Configures the optimizer and scheduler for training.

        Returns:
            Union[optim.Optimizer, OptimizerLRSchedulerConfig]: The optimizer and scheduler configuration.
        """

        params: Iterator[tuple[str, Any]] | Iterator[nn.Parameter]
        if self._optimizer_config.params_transformer is None:
            params = self.parameters()  # just because backward compatibility
        else:
            params = self._optimizer_config.params_transformer(self.named_parameters())

        optimizer = self._optimizer_config.optimizer_generator(params=params)

        if len(self._optimizer_config.schedulers) == 0:
            return optimizer

        lr_schedulers = [scheduler_cfg.build(optimizer) for scheduler_cfg in self._optimizer_config.schedulers]

        if len(lr_schedulers) == 1:
            return {"optimizer": optimizer, "lr_scheduler": lr_schedulers[0]}

        return [optimizer], lr_schedulers
