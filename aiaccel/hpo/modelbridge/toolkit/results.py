"""Result models and state writers for modelbridge steps."""

from __future__ import annotations

from typing import Any, Literal

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


def write_step_state(output_dir: Path, result: StepResult) -> Path:
    """Persist step result as workspace state JSON.

    Args:
        output_dir: Root output directory.
        result: Step execution result.

    Returns:
        Path: Written state file path.
    """
    return write_json(state_path(output_dir, result.step), result.to_state())
