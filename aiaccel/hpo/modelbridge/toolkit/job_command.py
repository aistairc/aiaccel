"""Helpers to wrap commands with aiaccel-job."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from ..config import ExecutionTargetConfig


def wrap_with_aiaccel_job(
    command_tokens: Sequence[str],
    execution: ExecutionTargetConfig,
    log_path: Path,
) -> list[str]:
    """Wrap command tokens with aiaccel-job invocation.

    Args:
        command_tokens: Inner command token sequence.
        execution: Execution target configuration.
        log_path: Log file path for aiaccel-job.

    Returns:
        list[str]: Wrapped command token sequence.
    """
    profile = execution.job_profile or "local"
    wrapped = ["aiaccel-job", profile, execution.job_mode]
    if execution.job_walltime:
        wrapped.extend(["--walltime", execution.job_walltime])
    wrapped.extend(execution.job_extra_args)
    wrapped.extend([str(log_path), "--"])
    wrapped.extend([str(token) for token in command_tokens])
    return wrapped
