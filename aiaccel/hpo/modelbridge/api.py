"""High-level API helpers for modelbridge."""

from __future__ import annotations

from typing import Any

from collections.abc import Mapping, Sequence
from pathlib import Path

from omegaconf import OmegaConf

from aiaccel.config import load_config as load_omegaconf_config
from aiaccel.config import resolve_inherit, setup_omegaconf

from .analyze import evaluate_model, fit_regression
from .collect import collect_eval, collect_train
from .config import BridgeConfig, load_bridge_config
from .execution import CommandFormat, emit_commands
from .layout import Role
from .pipeline import PipelineProfile, run_pipeline
from .prepare import prepare_eval, prepare_train
from .publish import publish_summary
from .toolkit.logging import setup_logging
from .toolkit.results import PipelineResult, StepResult


def load_config(path: Path, overrides: Mapping[str, Any] | None = None) -> BridgeConfig:
    """Load and validate a modelbridge configuration file.

    Args:
        path: Path to the modelbridge configuration file.
        overrides: Optional override mapping merged after loading.

    Returns:
        BridgeConfig: Validated modelbridge configuration.

    Raises:
        ValueError: If the loaded payload is not a mapping.
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
    """Run modelbridge steps or profile via pipeline facade.

    Args:
        config: Parsed modelbridge configuration.
        steps: Optional explicit step list.
        profile: Optional named profile.
        train_db_paths: Optional explicit train DB paths for collect.
        eval_db_paths: Optional explicit eval DB paths for collect.
        train_db_pairs: Optional explicit train DB pairs for collect.
        eval_db_pairs: Optional explicit eval DB pairs for collect.
        enable_logging: Whether to configure logging before running.

    Returns:
        PipelineResult: Aggregated pipeline execution result.
    """
    if enable_logging:
        setup_logging(
            config.bridge.log_level,
            config.bridge.output_dir,
            json_logs=config.bridge.json_log,
        )
    return run_pipeline(
        config,
        steps=steps,
        profile=profile,
        train_db_paths=train_db_paths,
        eval_db_paths=eval_db_paths,
        train_db_pairs=train_db_pairs,
        eval_db_pairs=eval_db_pairs,
    )


def prepare_train_step(config: BridgeConfig, *, enable_logging: bool = True) -> StepResult:
    """Run the `prepare_train` step directly.

    Args:
        config: Parsed modelbridge configuration.
        enable_logging: Whether to configure logging before running.

    Returns:
        StepResult: Step execution result.
    """
    if enable_logging:
        setup_logging(config.bridge.log_level, config.bridge.output_dir, json_logs=config.bridge.json_log)
    return prepare_train(config)


def prepare_eval_step(config: BridgeConfig, *, enable_logging: bool = True) -> StepResult:
    """Run the `prepare_eval` step directly.

    Args:
        config: Parsed modelbridge configuration.
        enable_logging: Whether to configure logging before running.

    Returns:
        StepResult: Step execution result.
    """
    if enable_logging:
        setup_logging(config.bridge.log_level, config.bridge.output_dir, json_logs=config.bridge.json_log)
    return prepare_eval(config)


def collect_train_step(
    config: BridgeConfig,
    *,
    db_paths: Sequence[Path] | None = None,
    db_pairs: Sequence[tuple[Path, Path]] | None = None,
    enable_logging: bool = True,
) -> StepResult:
    """Run the `collect_train` step directly.

    Args:
        config: Parsed modelbridge configuration.
        db_paths: Optional explicit DB path list.
        db_pairs: Optional explicit `(macro_db, micro_db)` pairs.
        enable_logging: Whether to configure logging before running.

    Returns:
        StepResult: Step execution result.
    """
    if enable_logging:
        setup_logging(config.bridge.log_level, config.bridge.output_dir, json_logs=config.bridge.json_log)
    return collect_train(config, db_paths=db_paths, db_pairs=db_pairs)


def collect_eval_step(
    config: BridgeConfig,
    *,
    db_paths: Sequence[Path] | None = None,
    db_pairs: Sequence[tuple[Path, Path]] | None = None,
    enable_logging: bool = True,
) -> StepResult:
    """Run the `collect_eval` step directly.

    Args:
        config: Parsed modelbridge configuration.
        db_paths: Optional explicit DB path list.
        db_pairs: Optional explicit `(macro_db, micro_db)` pairs.
        enable_logging: Whether to configure logging before running.

    Returns:
        StepResult: Step execution result.
    """
    if enable_logging:
        setup_logging(config.bridge.log_level, config.bridge.output_dir, json_logs=config.bridge.json_log)
    return collect_eval(config, db_paths=db_paths, db_pairs=db_pairs)


def fit_regression_step(config: BridgeConfig, *, enable_logging: bool = True) -> StepResult:
    """Run the `fit_regression` step directly.

    Args:
        config: Parsed modelbridge configuration.
        enable_logging: Whether to configure logging before running.

    Returns:
        StepResult: Step execution result.
    """
    if enable_logging:
        setup_logging(config.bridge.log_level, config.bridge.output_dir, json_logs=config.bridge.json_log)
    return fit_regression(config)


def evaluate_model_step(config: BridgeConfig, *, enable_logging: bool = True) -> StepResult:
    """Run the `evaluate_model` step directly.

    Args:
        config: Parsed modelbridge configuration.
        enable_logging: Whether to configure logging before running.

    Returns:
        StepResult: Step execution result.
    """
    if enable_logging:
        setup_logging(config.bridge.log_level, config.bridge.output_dir, json_logs=config.bridge.json_log)
    return evaluate_model(config)


def publish_summary_step(config: BridgeConfig, *, enable_logging: bool = True) -> StepResult:
    """Run the `publish_summary` step directly.

    Args:
        config: Parsed modelbridge configuration.
        enable_logging: Whether to configure logging before running.

    Returns:
        StepResult: Step execution result.
    """
    if enable_logging:
        setup_logging(config.bridge.log_level, config.bridge.output_dir, json_logs=config.bridge.json_log)
    return publish_summary(config)


def emit_commands_step(
    config: BridgeConfig,
    *,
    role: Role,
    fmt: CommandFormat,
    enable_logging: bool = True,
) -> Path:
    """Run the `emit_commands` step directly.

    Args:
        config: Parsed modelbridge configuration.
        role: Target role (`train` or `eval`).
        fmt: Output command format (`shell` or `json`).
        enable_logging: Whether to configure logging before running.

    Returns:
        Path: Path to emitted command artifact.
    """
    if enable_logging:
        setup_logging(config.bridge.log_level, config.bridge.output_dir, json_logs=config.bridge.json_log)
    return emit_commands(config, role=role, fmt=fmt)
