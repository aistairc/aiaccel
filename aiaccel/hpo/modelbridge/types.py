"""Shared data structures for the modelbridge pipeline."""

from __future__ import annotations

from typing import Any

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


__all__ = [
    "TrialContext",
    "EvaluationResult",
    "TrialResult",
    "RegressionSample",
]
