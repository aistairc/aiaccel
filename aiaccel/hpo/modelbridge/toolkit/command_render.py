"""Renderer helpers for command payloads."""

from __future__ import annotations

from typing import Any

from collections.abc import Mapping, Sequence
import shlex


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
