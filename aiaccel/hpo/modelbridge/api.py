"""Facade API for modelbridge runtime entrypoints."""

from __future__ import annotations

from typing import Any, TypeVar

from collections.abc import Callable, Mapping, Sequence
from functools import partial
from pathlib import Path

from omegaconf import OmegaConf

from aiaccel.config import load_config as load_omegaconf_config
from aiaccel.config import resolve_inherit, setup_omegaconf

from .common import CommandFormat, PipelineResult, Role, setup_logging
from .config import BridgeConfig, ExecutionTarget, load_bridge_config
from .execution import emit_commands
from .pipeline import PipelineProfile, run_pipeline

_ResultT = TypeVar("_ResultT")


def load_config(path: Path, overrides: Mapping[str, Any] | None = None) -> BridgeConfig:
    """Load and validate a modelbridge configuration file.

    Args:
        path: Path to modelbridge config YAML.
        overrides: Optional override mapping.

    Returns:
        BridgeConfig: Validated config object.

    Raises:
        ValueError: If loaded payload is not a mapping.
    """
    setup_omegaconf()
    config_path = path.expanduser().resolve()
    parent_ctx = {"config_path": str(config_path), "config_dir": str(config_path.parent)}
    conf = load_omegaconf_config(config_path, parent_ctx)
    conf = resolve_inherit(conf)
    container = OmegaConf.to_container(conf, resolve=True)
    if not isinstance(container, Mapping):
        raise ValueError("Bridge configuration must be a mapping")
    payload = dict(container)
    payload.pop("config_path", None)
    payload.pop("config_dir", None)
    return load_bridge_config(payload, overrides=overrides)


def run(
    config: BridgeConfig,
    steps: Sequence[str] | None = None,
    *,
    profile: PipelineProfile | None = None,
    train_db_paths: Sequence[Path] | None = None,
    eval_db_paths: Sequence[Path] | None = None,
    train_db_pairs: Sequence[tuple[Path, Path]] | None = None,
    eval_db_pairs: Sequence[tuple[Path, Path]] | None = None,
    enable_logging: bool = True,
) -> PipelineResult:
    """Run modelbridge steps or one profile.

    Args:
        config: Validated modelbridge config.
        steps: Optional explicit step list.
        profile: Optional profile name.
        train_db_paths: Optional train DB path override.
        eval_db_paths: Optional eval DB path override.
        train_db_pairs: Optional train DB pair override.
        eval_db_pairs: Optional eval DB pair override.
        enable_logging: Whether to initialize logging.

    Returns:
        PipelineResult: Aggregated step results.
    """
    return _run_with_optional_logging(
        config,
        enable_logging=enable_logging,
        action=partial(
            run_pipeline,
            config,
            steps=steps,
            profile=profile,
            train_db_paths=train_db_paths,
            eval_db_paths=eval_db_paths,
            train_db_pairs=train_db_pairs,
            eval_db_pairs=eval_db_pairs,
        ),
    )


def emit_commands_step(
    config: BridgeConfig,
    *,
    role: Role,
    fmt: CommandFormat,
    execution_target: ExecutionTarget | None = None,
    enable_logging: bool = True,
) -> Path:
    """Emit commands for one role plan.

    Args:
        config: Validated modelbridge config.
        role: Target role.
        fmt: Output format.
        execution_target: Optional execution target override.
        enable_logging: Whether to initialize logging.

    Returns:
        Path: Written command artifact path.
    """
    return _run_with_optional_logging(
        config,
        enable_logging=enable_logging,
        action=partial(emit_commands, config, role=role, fmt=fmt, execution_target=execution_target),
    )


def _run_with_optional_logging(
    config: BridgeConfig,
    *,
    enable_logging: bool,
    action: Callable[[], _ResultT],
) -> _ResultT:
    """Run an action with optional modelbridge logging initialization."""
    if enable_logging:
        setup_logging(config.bridge.log_level, config.bridge.output_dir, json_logs=config.bridge.json_log)
    return action()
