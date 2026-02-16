"""Command emission helpers for external optimize execution.

This module renders deterministic command artifacts from prepared role plans.
"""

from __future__ import annotations

from typing import Any, cast

from pathlib import Path
import shlex

from .common import (
    CommandFormat,
    Role,
    Target,
    command_path,
    optimize_log_path,
    plan_path,
    read_plan,
    write_json,
)
from .config import BridgeConfig, ExecutionTarget, ExecutionTargetConfig


def emit_commands(
    config: BridgeConfig,
    role: Role,
    fmt: CommandFormat,
    execution_target: ExecutionTarget | None = None,
) -> Path:
    """Emit deterministic optimize commands from one role plan.

    Args:
        config: Validated modelbridge configuration.
        role: Target role.
        fmt: Command output format.
        execution_target: Optional execution target override.

    Returns:
        Path: Written command artifact path.

    Raises:
        FileNotFoundError: If role plan file does not exist.
        ValueError: If plan payload is malformed.
    """
    output_dir = config.bridge.output_dir
    effective_execution = resolve_execution_config(config.bridge.execution, execution_target)

    source_plan = plan_path(output_dir, role)
    if not source_plan.exists():
        raise FileNotFoundError(f"Plan file not found: {source_plan}")

    plan_role, entries = read_plan(source_plan)
    if plan_role != role:
        raise ValueError(f"Plan role mismatch: expected {role}, got {plan_role}")
    if not entries:
        raise ValueError(f"No plan entries in {source_plan}")

    optimize_log_dir = effective_execution.job_log_dir or (output_dir / "workspace" / "logs" / "optimize")
    if effective_execution.target == "abci":
        optimize_log_dir.mkdir(parents=True, exist_ok=True)

    commands = _build_command_entries(entries, effective_execution=effective_execution, output_dir=output_dir)
    sorted_commands = _sort_command_entries(commands)

    destination = command_path(output_dir, role, fmt)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "json":
        write_json(destination, {"role": role, "commands": sorted_commands})
        return destination

    script_commands = [cast(list[str], entry["command"]) for entry in sorted_commands]
    script = _render_shell_script(script_commands)
    destination.write_text(script, encoding="utf-8")
    destination.chmod(0o755)
    return destination


def resolve_execution_config(
    base_execution: ExecutionTargetConfig,
    execution_target: ExecutionTarget | None,
) -> ExecutionTargetConfig:
    """Resolve effective execution target settings.

    Args:
        base_execution: Base execution settings from config.
        execution_target: Optional execution target override.

    Returns:
        ExecutionTargetConfig: Effective execution settings.

    Raises:
        ValueError: If target value is unsupported.
    """
    resolved_target = execution_target or base_execution.target
    if resolved_target not in ("local", "abci"):
        raise ValueError(f"Unsupported execution target: {resolved_target}")
    if resolved_target == base_execution.target:
        return base_execution

    payload = base_execution.model_dump(mode="python")
    payload["target"] = resolved_target

    if base_execution.target == "local" and base_execution.job_profile == "local" and resolved_target == "abci":
        payload["job_profile"] = None
    if base_execution.target == "abci" and base_execution.job_profile == "sge" and resolved_target == "local":
        payload["job_profile"] = None

    return ExecutionTargetConfig.model_validate(payload)


def _build_command_entries(
    entries: list[dict[str, Any]],
    *,
    effective_execution: ExecutionTargetConfig,
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Build command entries from validated plan entries.

    Args:
        entries: Validated plan entry mappings.
        effective_execution: Effective execution settings.
        output_dir: Root modelbridge output directory.

    Returns:
        list[dict[str, object]]: Command entry payloads.
    """
    command_entries: list[dict[str, Any]] = []
    optimize_log_dir = effective_execution.job_log_dir or (output_dir / "workspace" / "logs" / "optimize")

    for entry in entries:
        role_value = cast(Role, entry["role"])
        target_value = cast(Target, entry["target"])
        run_id = cast(int, entry["run_id"])
        command = ["aiaccel-hpo", "optimize", "--config", str(entry["config_path"])]
        if effective_execution.target == "abci":
            log_file = optimize_log_path(
                output_dir,
                role=role_value,
                scenario=str(entry["scenario"]),
                run_id=run_id,
                target=target_value,
            )
            wrapped_log_path = optimize_log_dir / log_file.name
            command = _wrap_with_aiaccel_job(command, effective_execution, wrapped_log_path)

        command_entries.append(
            {
                "scenario": str(entry["scenario"]),
                "role": role_value,
                "run_id": run_id,
                "target": target_value,
                "config_path": str(entry["config_path"]),
                "execution_target": effective_execution.target,
                "command": command,
            }
        )

    return command_entries


def _sort_command_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort command entries deterministically by scenario, role, run, and target."""

    def _run_id(entry: dict[str, Any]) -> int:
        value = entry.get("run_id")
        return value if isinstance(value, int) else -1

    return sorted(
        [dict(entry) for entry in entries],
        key=lambda item: (
            str(item.get("scenario", "")),
            str(item.get("role", "")),
            _run_id(item),
            str(item.get("target", "")),
            str(item.get("config_path", "")),
        ),
    )


def _render_shell_script(commands: list[list[str]]) -> str:
    """Render command list into an executable shell script text."""
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    lines.extend(" ".join(shlex.quote(token) for token in command) for command in commands)
    return "\n".join(lines) + "\n"


def _wrap_with_aiaccel_job(command: list[str], execution: ExecutionTargetConfig, log_path: Path) -> list[str]:
    """Wrap an optimize command with ``aiaccel-job`` arguments."""
    profile = execution.job_profile or ("sge" if execution.target == "abci" else "local")
    wrapped = ["aiaccel-job", profile, execution.job_mode]
    if execution.job_walltime:
        wrapped.extend(["--walltime", execution.job_walltime])
    wrapped.extend(list(execution.job_extra_args))
    wrapped.append(str(log_path))
    wrapped.append("--")
    wrapped.extend(command)
    return wrapped
