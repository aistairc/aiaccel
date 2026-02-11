"""Result models and state writers for modelbridge steps."""

from __future__ import annotations

from typing import Any, Literal

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ..layout import state_path
from .io import write_json

StepStatus = Literal["success", "skipped", "failed", "partial"]


@dataclass(frozen=True)
class StepResult:
    """A normalized result payload for every modelbridge step."""

    step: str
    status: StepStatus
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    reason: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        """Validate step result invariants."""
        if self.status in {"failed", "skipped"} and (self.reason is None or not self.reason.strip()):
            raise ValueError("reason is required when status is failed or skipped")

    def to_state(self) -> dict[str, Any]:
        """Convert step result into state payload.

        Returns:
            dict[str, Any]: JSON-serializable state payload.
        """
        payload: dict[str, Any] = {
            "status": self.status,
            "timestamp": self.timestamp,
            "inputs": self.inputs,
            "outputs": self.outputs,
        }
        if self.reason is not None:
            payload["reason"] = self.reason
        return payload


@dataclass(frozen=True)
class PipelineResult:
    """Result of one pipeline invocation."""

    results: list[StepResult]
    summary_path: Path | None = None
    manifest_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert pipeline result into serializable payload.

        Returns:
            dict[str, Any]: JSON-serializable payload.
        """
        return {
            "results": [result.to_state() | {"step": result.step} for result in self.results],
            "summary_path": str(self.summary_path) if self.summary_path else None,
            "manifest_path": str(self.manifest_path) if self.manifest_path else None,
        }


def finalize_scenario_step(
    *,
    output_dir: Path,
    step: str,
    strict_mode: bool,
    scenario_outputs: Mapping[str, Mapping[str, Any]],
    issues: Sequence[str],
    inputs: Mapping[str, Any] | None = None,
    extra_outputs: Mapping[str, Any] | None = None,
) -> StepResult:
    """Build, persist, and return one step result from per-scenario outputs.

    Args:
        output_dir: Root output directory.
        step: Step name.
        strict_mode: Whether strict failure mode is enabled.
        scenario_outputs: Scenario-level output payloads.
        issues: Collected non-fatal/fatal issue messages.
        inputs: Optional input payload.
        extra_outputs: Optional additional output payload merged into outputs.

    Returns:
        StepResult: Finalized step result.

    Raises:
        RuntimeError: If strict mode is enabled and issues were collected.
    """
    normalized_scenarios = {name: dict(payload) for name, payload in scenario_outputs.items()}
    outputs: dict[str, Any] = {"scenarios": normalized_scenarios}
    if extra_outputs:
        outputs.update(dict(extra_outputs))

    issue_list = list(issues)
    success_count = sum(1 for payload in normalized_scenarios.values() if payload.get("status") == "success")
    total = len(normalized_scenarios)

    if issue_list and strict_mode:
        failure_reason = "; ".join(issue_list)
        result = StepResult(
            step=step,
            status="failed",
            inputs=dict(inputs or {}),
            outputs=outputs,
            reason=failure_reason,
        )
        write_step_state(output_dir, result)
        raise RuntimeError(failure_reason)

    status: StepStatus
    reason: str | None
    if success_count == total:
        status = "success"
        reason = None
    elif success_count == 0:
        status = "skipped"
        reason = "; ".join(issue_list) if issue_list else f"{step} skipped"
    else:
        status = "partial"
        reason = "; ".join(issue_list)

    result = StepResult(
        step=step,
        status=status,
        inputs=dict(inputs or {}),
        outputs=outputs,
        reason=reason,
    )
    write_step_state(output_dir, result)
    return result


def write_step_state(output_dir: Path, result: StepResult) -> Path:
    """Persist step result as workspace state JSON.

    Args:
        output_dir: Root output directory.
        result: Step execution result.

    Returns:
        Path: Written state file path.
    """
    return write_json(state_path(output_dir, result.step), result.to_state())
