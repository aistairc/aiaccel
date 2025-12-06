"""Utilities for aggregating pipeline results."""

from __future__ import annotations

from typing import Any

from dataclasses import dataclass, field
from pathlib import Path

from .io import write_json


@dataclass
class ScenarioSummary:
    """Aggregated results for a finished scenario."""

    train_pairs: int
    eval_pairs: int
    train_macro_best: list[dict[str, float]]
    train_micro_best: list[dict[str, float]]
    eval_macro_best: list[dict[str, float]]
    eval_micro_best: list[dict[str, float]]
    train_metrics: dict[str, float]
    eval_metrics: dict[str, float]


@dataclass
class SummaryBuilder:
    output_dir: Path
    scenarios: dict[str, ScenarioSummary] = field(default_factory=dict)

    def add(self, name: str, summary: ScenarioSummary) -> None:
        """Record ``summary`` under ``name``."""

        self.scenarios[name] = summary

    def finalize(self) -> dict[str, Any]:
        """Persist the collected summaries and return the payload."""

        payload = {name: summary.__dict__ for name, summary in self.scenarios.items()}
        write_json(self.output_dir / "summary.json", payload)
        return payload


__all__ = ["ScenarioSummary", "SummaryBuilder"]
