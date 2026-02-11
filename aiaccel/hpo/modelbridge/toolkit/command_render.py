"""Renderer helpers for command payloads."""

from __future__ import annotations

from typing import Any

from collections.abc import Mapping, Sequence
import shlex


def sort_command_entries(entries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return deterministically sorted command entries.

    Args:
        entries: Command entry payloads.

    Returns:
        list[dict[str, Any]]: Entries sorted by scenario/run_id/target.
    """

    def _key(item: Mapping[str, Any]) -> tuple[str, int, str]:
        scenario = str(item.get("scenario", ""))
        run_raw = item.get("run_id", 0)
        run_id = int(run_raw) if isinstance(run_raw, int) and not isinstance(run_raw, bool) else 0
        target = str(item.get("target", ""))
        return (scenario, run_id, target)

    return [dict(item) for item in sorted(entries, key=_key)]


def render_json_commands(role: str, commands: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Build JSON payload for emitted commands.

    Args:
        role: Command role (`train` or `eval`).
        commands: Command entries with metadata.

    Returns:
        dict[str, Any]: JSON payload for command artifact.
    """
    return {"role": role, "commands": [dict(item) for item in commands]}


def render_shell_commands(commands: Sequence[Sequence[str]]) -> str:
    """Render shell script content for command execution.

    Args:
        commands: Command token sequences.

    Returns:
        str: Shell script text.
    """
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for command in commands:
        lines.append(shlex.join([str(token) for token in command]))
    lines.append("")
    return "\n".join(lines)
