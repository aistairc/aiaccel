"""Objective evaluation strategies."""

from __future__ import annotations

from typing import Any, cast

from collections.abc import Callable, Mapping
from functools import partial
import importlib
import json
import os
import subprocess
import urllib.request

from .config import ObjectiveConfig
from .types import EvaluationResult, TrialContext


def build_evaluator(
    config: ObjectiveConfig,
    base_env: Mapping[str, str] | None = None,
) -> Callable[[TrialContext], EvaluationResult]:
    """Build an evaluator callable based on ``ObjectiveConfig``."""

    env_payload = dict(base_env or {})
    if config.target.startswith(("http://", "https://")):
        return cast(
            Callable[[TrialContext], EvaluationResult],
            partial(
                http_objective,
                endpoint=config.target,
                timeout=config.timeout,
                base_env=env_payload,
            ),
        )

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
                base_env=env_payload,
            ),
        )

    def evaluator(context: TrialContext) -> Any:
        env = _build_env(context, env_payload)
        return func(context, env)

    return evaluator


def command_objective(
    context: TrialContext,
    *,
    command: list[str],
    timeout: float | None,
    base_env: Mapping[str, str] | None,
) -> EvaluationResult:
    """Execute an external command and parse its JSON output."""

    env = _build_env(context, os.environ | dict(base_env or {}))
    env["AIACCEL_TRIAL_INDEX"] = env["AIACCEL_TRIAL"]

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


def http_objective(
    context: TrialContext,
    *,
    endpoint: str,
    timeout: float | None,
    base_env: Mapping[str, str] | None,
) -> EvaluationResult:
    """POST context/env as JSON to an HTTP endpoint and parse response."""

    env = _build_env(context, base_env or {})
    request_payload = {
        "scenario": context.scenario,
        "phase": context.phase,
        "trial_index": context.trial_index,
        "params": context.params,
        "env": env,
    }
    data = json.dumps(request_payload).encode("utf-8")
    req = urllib.request.Request(endpoint, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp_body = resp.read().decode("utf-8")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"HTTP objective failed: {exc}") from exc

    try:
        payload = json.loads(resp_body or "{}")
    except json.JSONDecodeError as exc:  # noqa: PERF203
        raise RuntimeError("HTTP objective did not return valid JSON") from exc

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


def _build_env(context: TrialContext, base_env: Mapping[str, str]) -> dict[str, str]:
    env = dict(base_env)
    env.update(
        {
            "AIACCEL_SCENARIO": context.scenario,
            "AIACCEL_PHASE": context.phase,
            "AIACCEL_TRIAL": str(context.trial_index),
        }
    )
    env.update({f"AIACCEL_PARAM_{key.upper()}": str(value) for key, value in context.params.items()})
    return env


__all__ = ["build_evaluator", "command_objective"]
