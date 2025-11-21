"""Shared data structures for the modelbridge pipeline."""

from __future__ import annotations

from typing import Any, Callable

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class TrialContext:
    """Context passed to objective callables."""

    scenario: str
    phase: str
    trial_index: int
    params: dict[str, float]
    seed: int
    output_dir: Path


@dataclass(slots=True)
class EvaluationResult:
    """Outcome returned by objective callables."""

    objective: float
    metrics: dict[str, float] = field(default_factory=dict)
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TrialResult:
    """Captured Optuna trial output."""

    context: TrialContext
    evaluation: EvaluationResult
    state: str


@dataclass(slots=True)
class RegressionSample:
    """Sample collected for regression training."""

    features: dict[str, float]
    target: dict[str, float]


@dataclass(slots=True)
class PhaseContext:
    """Execution unit used by the modelbridge pipeline."""

    scenario: str
    phase: str
    role: str | None = None
    target: str | None = None
    run_id: int | None = None
    seed: int | None = None
    output_dir: Path | None = None
    working_directory: Path | None = None
    runner: Any | None = field(default=None, repr=False, compare=False)

    def serializable(self) -> dict[str, object]:
        """Return a JSON-safe dict view without the runner callable."""

        return {
            "scenario": self.scenario,
            "phase": self.phase,
            "role": self.role,
            "target": self.target,
            "run_id": self.run_id,
            "seed": self.seed,
            "output_dir": str(self.output_dir) if self.output_dir else None,
            "working_directory": str(self.working_directory) if self.working_directory else None,
        }


RunnerFn = Callable[[PhaseContext, Any, Any, Any], Any]
EvaluatorFn = Callable[[TrialContext], EvaluationResult]


__all__ = [
    "TrialContext",
    "EvaluationResult",
    "TrialResult",
    "RegressionSample",
    "PhaseContext",
    "RunnerFn",
    "EvaluatorFn",
]
