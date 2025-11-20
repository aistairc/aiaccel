"""Objective evaluation strategies."""

from __future__ import annotations

from typing import Any, cast

from collections.abc import Callable, Mapping
from functools import partial
import importlib
import inspect
import json
import os
import subprocess

from .config import ObjectiveConfig
from .types import EvaluationResult, TrialContext


def build_evaluator(
    config: ObjectiveConfig,
    base_env: Mapping[str, str] | None = None,
) -> Callable[[TrialContext], EvaluationResult]:
    """Build an evaluator callable based on ``ObjectiveConfig``."""

    func = _import_callable(config.target)
    if func is command_objective:
        if not config.command:
            raise ValueError("command_objective requires a command list")
        return cast(
            Callable[[TrialContext], EvaluationResult],
            partial(
                command_objective,
                command=config.command,
                timeout=config.timeout,
                base_env=base_env,
            ),
        )

    signature = inspect.signature(func)
    accepts_base_env = "base_env" in signature.parameters

    def evaluator(context: TrialContext) -> Any:
        if accepts_base_env:
            return func(context, base_env=base_env)
        return func(context)

    return evaluator


def command_objective(
    context: TrialContext,
    *,
    command: list[str],
    timeout: float | None,
    base_env: Mapping[str, str] | None,
) -> EvaluationResult:
    """Execute an external command and parse its JSON output."""

    env = os.environ.copy()
    if base_env:
        env.update(base_env)
    env.update(
        {
            "AIACCEL_SCENARIO": context.scenario,
            "AIACCEL_PHASE": context.phase,
            "AIACCEL_TRIAL_INDEX": str(context.trial_index),
        }
    )
    for key, value in context.params.items():
        env[f"AIACCEL_PARAM_{key.upper()}"] = str(value)

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.CalledProcessError as exc:  # noqa: PERF203
        raise RuntimeError(f"Command failed with exit status {exc.returncode}") from exc
    except subprocess.TimeoutExpired as exc:  # noqa: PERF203
        raise RuntimeError("Command timed out") from exc

    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError("Command did not return valid JSON") from exc

    objective = float(payload.get("objective"))
    metrics = {str(k): float(v) for k, v in payload.get("metrics", {}).items()}
    extra = {str(k): v for k, v in payload.get("payload", {}).items()}
    return EvaluationResult(objective=objective, metrics=metrics, payload=extra)


def _import_callable(path: str) -> Any:
    module_name, _, attr_name = path.rpartition(".")
    if not module_name:
        raise ValueError(f"Invalid target path '{path}'")
    module = importlib.import_module(module_name)
    try:
        func = getattr(module, attr_name)
    except AttributeError as exc:  # noqa: PERF203
        raise ValueError(f"Target '{path}' not found") from exc
    if not callable(func):
        raise ValueError(f"Target '{path}' is not callable")
    return func


__all__ = ["build_evaluator", "command_objective"]
