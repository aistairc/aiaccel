"""Command emission utilities for external HPO execution."""

from __future__ import annotations

from typing import Any, Literal, cast

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .config import BridgeConfig, ExecutionTarget, ExecutionTargetConfig
from .layout import Role, Target, command_path, commands_dir, optimize_log_path, optimize_logs_dir, plan_path
from .toolkit.command_render import render_json_commands, render_shell_commands, sort_command_entries
from .toolkit.io import read_json, write_json
from .toolkit.job_command import wrap_with_aiaccel_job

CommandFormat = Literal["shell", "json"]


@dataclass(frozen=True)
class PlanCommandEntry:
    """Normalized command source parsed from plan payload."""

    scenario: str
    role: Role
    run_id: int
    target: Target
    config_path: str

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> PlanCommandEntry:
        """Validate one plan entry payload."""
        return cls(
            scenario=_require_string(payload.get("scenario"), field_name="scenario"),
            role=_require_role(payload.get("role")),
            run_id=_require_run_id(payload.get("run_id")),
            target=_require_target(payload.get("target")),
            config_path=_require_string(payload.get("config_path"), field_name="config_path"),
        )


@dataclass(frozen=True)
class CommandEntry:
    """Typed command entry for rendered outputs."""

    scenario: str
    role: Role
    run_id: int
    target: Target
    config_path: str
    execution_target: ExecutionTarget
    command: tuple[str, ...]

    def to_payload(self) -> dict[str, Any]:
        """Convert to JSON-serializable mapping."""
        return {
            "scenario": self.scenario,
            "role": self.role,
            "run_id": self.run_id,
            "target": self.target,
            "config_path": self.config_path,
            "execution_target": self.execution_target,
            "command": list(self.command),
        }


def emit_commands(
    config: BridgeConfig,
    role: Role,
    fmt: CommandFormat,
    execution_target: ExecutionTarget | None = None,
) -> Path:
    """Emit deterministic optimize commands from a role plan.

    Args:
        config: Parsed modelbridge configuration.
        role: Target role (`train` or `eval`).
        fmt: Output format (`shell` or `json`).
        execution_target: Optional execution target override (`local` or `abci`).

    Returns:
        Path: Path to emitted command artifact.

    Raises:
        FileNotFoundError: If the role plan file does not exist.
        ValueError: If the plan payload is malformed.
    """
    output_dir = config.bridge.output_dir
    effective_execution = resolve_execution_config(config.bridge.execution, execution_target)
    resolved_target = effective_execution.target

    source_plan = plan_path(output_dir, role)
    if not source_plan.exists():
        raise FileNotFoundError(f"Plan file not found: {source_plan}")

    payload = read_json(source_plan)
    entries_payload = payload.get("entries") if isinstance(payload, dict) else None
    entries = _parse_plan_entries(entries_payload, role=role)
    if not entries:
        raise ValueError(f"No plan entries in {source_plan}")

    optimize_log_dir = effective_execution.job_log_dir
    if optimize_log_dir is None:
        optimize_log_dir = optimize_logs_dir(output_dir)
    if resolved_target == "abci":
        optimize_log_dir.mkdir(parents=True, exist_ok=True)

    commands = _build_command_entries(
        entries,
        resolved_target=resolved_target,
        execution=effective_execution,
        output_dir=output_dir,
        optimize_log_dir=optimize_log_dir,
    )

    rendered_commands = sort_command_entries([item.to_payload() for item in commands])
    destination = command_path(output_dir, role, fmt)
    commands_dir(output_dir).mkdir(parents=True, exist_ok=True)

    if fmt == "json":
        write_json(destination, render_json_commands(role, rendered_commands))
        return destination

    shell_commands: list[list[str]] = []
    for item in rendered_commands:
        command_tokens = item.get("command")
        if not isinstance(command_tokens, list) or any(not isinstance(token, str) for token in command_tokens):
            raise ValueError("Generated command payload is malformed.")
        shell_commands.append(command_tokens)

    script = render_shell_commands(shell_commands)
    destination.write_text(script, encoding="utf-8")
    destination.chmod(0o755)
    return destination


def resolve_execution_config(
    base_execution: ExecutionTargetConfig,
    execution_target: ExecutionTarget | None,
) -> ExecutionTargetConfig:
    """Resolve effective execution settings for command emission.

    Args:
        base_execution: Execution settings loaded from modelbridge config.
        execution_target: Optional execution target override.

    Returns:
        ExecutionTargetConfig: Effective settings for command rendering.

    Raises:
        ValueError: If the execution target is unsupported.
    """
    resolved_target = execution_target or base_execution.target
    if resolved_target not in ("local", "abci"):
        raise ValueError(f"Unsupported execution target: {resolved_target}")
    if resolved_target == base_execution.target:
        return base_execution

    payload = base_execution.model_dump(mode="python")
    payload["target"] = resolved_target

    # If the source target defaulted profile by target, reset it to allow
    # target-specific defaults to be re-applied for the override target.
    if base_execution.target == "local" and base_execution.job_profile == "local" and resolved_target == "abci":
        payload["job_profile"] = None
    if base_execution.target == "abci" and base_execution.job_profile == "sge" and resolved_target == "local":
        payload["job_profile"] = None

    return ExecutionTargetConfig.model_validate(payload)


def _parse_plan_entries(entries: object, *, role: Role) -> list[PlanCommandEntry]:
    """Validate and normalize role plan entries."""
    if entries is None:
        return []
    if not isinstance(entries, list):
        raise ValueError("Plan payload has malformed entries.")

    parsed_entries: list[PlanCommandEntry] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("Plan contains malformed entry.")
        parsed = PlanCommandEntry.from_payload(entry)
        if parsed.role != role:
            raise ValueError(f"Plan entry role mismatch: expected {role}, got {parsed.role}")
        parsed_entries.append(parsed)
    return parsed_entries


def _build_command_entries(
    entries: Sequence[PlanCommandEntry],
    *,
    resolved_target: ExecutionTarget,
    execution: ExecutionTargetConfig,
    output_dir: Path,
    optimize_log_dir: Path,
) -> list[CommandEntry]:
    """Build command entry payloads from plan entries."""
    payloads: list[CommandEntry] = []
    for entry in entries:
        payloads.append(
            _build_command_entry(
                entry,
                resolved_target=resolved_target,
                execution=execution,
                output_dir=output_dir,
                optimize_log_dir=optimize_log_dir,
            )
        )
    return payloads


def _build_command_entry(
    entry: PlanCommandEntry,
    *,
    resolved_target: ExecutionTarget,
    execution: ExecutionTargetConfig,
    output_dir: Path,
    optimize_log_dir: Path,
) -> CommandEntry:
    """Build one command entry payload."""
    command = ["aiaccel-hpo", "optimize", "--config", entry.config_path]
    if resolved_target == "abci":
        default_log_path = optimize_log_path(output_dir, entry.role, entry.scenario, entry.run_id, entry.target)
        log_path = optimize_log_dir / default_log_path.name
        command = wrap_with_aiaccel_job(command, execution, log_path)
    return CommandEntry(
        scenario=entry.scenario,
        role=entry.role,
        run_id=entry.run_id,
        target=entry.target,
        config_path=entry.config_path,
        execution_target=resolved_target,
        command=tuple(command),
    )


def _require_string(value: object, *, field_name: str) -> str:
    """Validate non-empty string field from one plan entry."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"Plan entry missing {field_name}.")
    return value


def _require_role(value: object) -> Role:
    """Validate role value from one plan entry."""
    if value not in ("train", "eval"):
        raise ValueError("Plan entry missing role.")
    return cast(Role, value)


def _require_run_id(value: object) -> int:
    """Validate run id value from one plan entry."""
    if isinstance(value, bool):
        raise ValueError("Plan entry has invalid run_id.")
    if not isinstance(value, int):
        raise ValueError("Plan entry missing run_id.")
    return value


def _require_target(value: object) -> Target:
    """Validate target value from one plan entry."""
    if value not in ("macro", "micro"):
        raise ValueError("Plan entry missing target.")
    return cast(Target, value)
