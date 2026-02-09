"""Command emission utilities for external HPO execution."""

from __future__ import annotations

from typing import Any, Literal

from pathlib import Path

from .config import BridgeConfig
from .layout import Role, command_path, commands_dir, plan_path
from .toolkit.command_render import render_json_commands, render_shell_commands
from .toolkit.io import read_json, write_json

CommandFormat = Literal["shell", "json"]


def emit_commands(config: BridgeConfig, role: Role, fmt: CommandFormat) -> Path:
    """Emit deterministic optimize commands from a role plan.

    Args:
        config: Parsed modelbridge configuration.
        role: Target role (`train` or `eval`).
        fmt: Output format (`shell` or `json`).

    Returns:
        Path: Path to emitted command artifact.

    Raises:
        FileNotFoundError: If the role plan file does not exist.
        ValueError: If the plan payload is malformed.
    """
    output_dir = config.bridge.output_dir
    source_plan = plan_path(output_dir, role)
    if not source_plan.exists():
        raise FileNotFoundError(f"Plan file not found: {source_plan}")

    payload = read_json(source_plan)
    entries = payload.get("entries", []) if isinstance(payload, dict) else []
    if not entries:
        raise ValueError(f"No plan entries in {source_plan}")

    commands: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Plan contains malformed entry.")
        config_path = entry.get("config_path")
        if not isinstance(config_path, str) or not config_path:
            raise ValueError("Plan entry missing config_path.")
        command = ["aiaccel-hpo", "optimize", "--config", config_path]
        commands.append(
            {
                "scenario": entry.get("scenario"),
                "role": entry.get("role"),
                "run_id": entry.get("run_id"),
                "target": entry.get("target"),
                "config_path": config_path,
                "command": command,
            }
        )

    commands.sort(key=_command_sort_key)
    destination = command_path(output_dir, role, fmt)
    commands_dir(output_dir).mkdir(parents=True, exist_ok=True)

    if fmt == "json":
        write_json(destination, render_json_commands(role, commands))
        return destination

    shell_commands: list[list[str]] = []
    for item in commands:
        command_tokens = item.get("command")
        if not isinstance(command_tokens, list) or any(not isinstance(token, str) for token in command_tokens):
            raise ValueError("Generated command payload is malformed.")
        shell_commands.append(command_tokens)

    script = render_shell_commands(shell_commands)
    destination.write_text(script, encoding="utf-8")
    destination.chmod(0o755)
    return destination


def _command_sort_key(command: dict[str, Any]) -> tuple[str, int, str]:
    """Return deterministic sort key for emitted commands."""
    scenario = str(command.get("scenario", ""))
    run_id = int(command.get("run_id", 0))
    target = str(command.get("target", ""))
    return (scenario, run_id, target)
