"""Facade API for modelbridge step execution."""

from __future__ import annotations

from typing import Any, TypeVar

from collections.abc import Callable, Mapping, Sequence
from functools import partial
from pathlib import Path

from omegaconf import OmegaConf

from aiaccel.config import load_config as load_omegaconf_config
from aiaccel.config import resolve_inherit, setup_omegaconf

from .analyze import evaluate_model, fit_regression
from .collect import collect_eval, collect_train
from .common import StepResult, setup_logging
from .config import BridgeConfig, load_bridge_config
from .hpo_runner import hpo_eval, hpo_train
from .prepare import prepare_eval, prepare_train
from .publish import publish_summary

_ResultT = TypeVar("_ResultT")


def load_config(path: Path, overrides: Mapping[str, Any] | None = None) -> BridgeConfig:
    """Load and validate a modelbridge configuration file.

    This function resolves OmegaConf inheritance first, then validates the final
    payload with the modelbridge Pydantic schema.

    Args:
        path: Path to the modelbridge YAML file.
        overrides: Optional nested override mapping merged after config resolution.

    Returns:
        BridgeConfig: Validated modelbridge runtime configuration.

    Raises:
        ValueError: If the resolved config payload is not a mapping or fails validation.
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


def prepare_train_step(config: BridgeConfig, *, enable_logging: bool = True) -> StepResult:
    """Run the ``prepare_train`` step.

    This step creates run-level optimize configs and plan artifacts for train
    role entries.

    Args:
        config: Validated modelbridge configuration.
        enable_logging: Whether to initialize modelbridge logging before execution.

    Returns:
        StepResult: Persistable step execution result for ``prepare_train``.

    Raises:
        ValueError: If input configuration contains invalid step settings.
        RuntimeError: If step execution fails in strict-mode finalization.
    """
    return _run_with_optional_logging(config, enable_logging=enable_logging, action=partial(prepare_train, config))


def prepare_eval_step(config: BridgeConfig, *, enable_logging: bool = True) -> StepResult:
    """Run the ``prepare_eval`` step.

    This step creates run-level optimize configs and plan artifacts for eval
    role entries.

    Args:
        config: Validated modelbridge configuration.
        enable_logging: Whether to initialize modelbridge logging before execution.

    Returns:
        StepResult: Persistable step execution result for ``prepare_eval``.

    Raises:
        ValueError: If input configuration contains invalid step settings.
        RuntimeError: If step execution fails in strict-mode finalization.
    """
    return _run_with_optional_logging(config, enable_logging=enable_logging, action=partial(prepare_eval, config))


def hpo_train_step(config: BridgeConfig, *, enable_logging: bool = True) -> StepResult:
    """Run the ``hpo_train`` step.

    This step emits role command artifacts, generates per-run scripts, and
    executes train-role optimize commands.

    Args:
        config: Validated modelbridge configuration.
        enable_logging: Whether to initialize modelbridge logging before execution.

    Returns:
        StepResult: Persistable step execution result for ``hpo_train``.

    Raises:
        ValueError: If generated plan/command artifacts are malformed.
        RuntimeError: If optimize command execution fails.
    """
    return _run_with_optional_logging(config, enable_logging=enable_logging, action=partial(hpo_train, config))


def hpo_eval_step(config: BridgeConfig, *, enable_logging: bool = True) -> StepResult:
    """Run the ``hpo_eval`` step.

    This step emits role command artifacts, generates per-run scripts, and
    executes eval-role optimize commands.

    Args:
        config: Validated modelbridge configuration.
        enable_logging: Whether to initialize modelbridge logging before execution.

    Returns:
        StepResult: Persistable step execution result for ``hpo_eval``.

    Raises:
        ValueError: If generated plan/command artifacts are malformed.
        RuntimeError: If optimize command execution fails.
    """
    return _run_with_optional_logging(config, enable_logging=enable_logging, action=partial(hpo_eval, config))


def collect_train_step(
    config: BridgeConfig,
    *,
    db_paths: Sequence[Path] | None = None,
    db_pairs: Sequence[tuple[Path, Path]] | None = None,
    enable_logging: bool = True,
) -> StepResult:
    """Run the ``collect_train`` step.

    This step collects macro/micro best-parameter pairs for train role from
    explicit DB inputs, plan files, or layout scan fallback.

    Args:
        config: Validated modelbridge configuration.
        db_paths: Optional DB path candidates.
        db_pairs: Optional explicit macro/micro DB path tuples.
        enable_logging: Whether to initialize modelbridge logging before execution.

    Returns:
        StepResult: Persistable step execution result for ``collect_train``.

    Raises:
        ValueError: If provided inputs or discovered artifacts are malformed.
        RuntimeError: If strict mode escalates scenario collection issues.
    """
    return _run_with_optional_logging(
        config,
        enable_logging=enable_logging,
        action=partial(collect_train, config, db_paths=db_paths, db_pairs=db_pairs),
    )


def collect_eval_step(
    config: BridgeConfig,
    *,
    db_paths: Sequence[Path] | None = None,
    db_pairs: Sequence[tuple[Path, Path]] | None = None,
    enable_logging: bool = True,
) -> StepResult:
    """Run the ``collect_eval`` step.

    This step collects macro/micro best-parameter pairs for eval role from
    explicit DB inputs, plan files, or layout scan fallback.

    Args:
        config: Validated modelbridge configuration.
        db_paths: Optional DB path candidates.
        db_pairs: Optional explicit macro/micro DB path tuples.
        enable_logging: Whether to initialize modelbridge logging before execution.

    Returns:
        StepResult: Persistable step execution result for ``collect_eval``.

    Raises:
        ValueError: If provided inputs or discovered artifacts are malformed.
        RuntimeError: If strict mode escalates scenario collection issues.
    """
    return _run_with_optional_logging(
        config,
        enable_logging=enable_logging,
        action=partial(collect_eval, config, db_paths=db_paths, db_pairs=db_pairs),
    )


def fit_regression_step(config: BridgeConfig, *, enable_logging: bool = True) -> StepResult:
    """Run the ``fit_regression`` step.

    This step trains per-scenario regression models from collected train pair
    CSV files.

    Args:
        config: Validated modelbridge configuration.
        enable_logging: Whether to initialize modelbridge logging before execution.

    Returns:
        StepResult: Persistable step execution result for ``fit_regression``.

    Raises:
        ValueError: If training inputs are malformed.
        RuntimeError: If strict mode escalates scenario fit failures.
    """
    return _run_with_optional_logging(config, enable_logging=enable_logging, action=partial(fit_regression, config))


def evaluate_model_step(config: BridgeConfig, *, enable_logging: bool = True) -> StepResult:
    """Run the ``evaluate_model`` step.

    This step loads fitted models and evaluates prediction quality for eval
    pair CSV files.

    Args:
        config: Validated modelbridge configuration.
        enable_logging: Whether to initialize modelbridge logging before execution.

    Returns:
        StepResult: Persistable step execution result for ``evaluate_model``.

    Raises:
        ValueError: If model or evaluation inputs are malformed.
        RuntimeError: If strict mode escalates scenario evaluation failures.
    """
    return _run_with_optional_logging(config, enable_logging=enable_logging, action=partial(evaluate_model, config))


def publish_summary_step(config: BridgeConfig, *, enable_logging: bool = True) -> StepResult:
    """Run the ``publish_summary`` step.

    This step aggregates step states and scenario artifacts into summary and
    manifest JSON outputs.

    Args:
        config: Validated modelbridge configuration.
        enable_logging: Whether to initialize modelbridge logging before execution.

    Returns:
        StepResult: Persistable step execution result for ``publish_summary``.

    Raises:
        OSError: If output files cannot be written.
        ValueError: If generated artifact payloads are invalid.
    """
    return _run_with_optional_logging(config, enable_logging=enable_logging, action=partial(publish_summary, config))


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
