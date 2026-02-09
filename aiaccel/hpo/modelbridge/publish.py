"""Publish step for modelbridge."""

from __future__ import annotations

from typing import Any

from pathlib import Path

from .config import BridgeConfig
from .layout import scenario_dir, state_dir
from .toolkit.io import hash_file, read_json, write_json
from .toolkit.results import StepResult, write_step_state


def publish_summary(config: BridgeConfig) -> StepResult:
    """Aggregate summary and manifest from scenario artifacts and step states.

    Args:
        config: Parsed modelbridge configuration.

    Returns:
        StepResult: Step execution result for `publish_summary`.
    """
    output_dir = config.bridge.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    states = _load_states(output_dir)
    scenario_payload: dict[str, dict[str, Any]] = {}

    for scenario in config.bridge.scenarios:
        scenario_path = scenario_dir(output_dir, scenario.name)
        train_pairs_path = scenario_path / "train_pairs.csv"
        eval_pairs_path = scenario_path / "test_pairs.csv"
        train_metrics_path = scenario_path / "metrics" / "train_metrics.json"
        eval_metrics_path = scenario_path / "metrics" / "eval_metrics.json"

        scenario_payload[scenario.name] = {
            "train_pairs": _count_csv_rows(train_pairs_path),
            "eval_pairs": _count_csv_rows(eval_pairs_path),
            "train_metrics": read_json(train_metrics_path) if train_metrics_path.exists() else {},
            "eval_metrics": read_json(eval_metrics_path) if eval_metrics_path.exists() else {},
        }

    summary_path = write_json(output_dir / "summary.json", scenario_payload)
    manifest_payload = {
        "states": states,
        "scenarios": scenario_payload,
        "artifacts": _collect_artifacts(output_dir),
        "config": config.model_dump(mode="json"),
    }
    manifest_path = write_json(output_dir / "manifest.json", manifest_payload)

    result = StepResult(
        step="publish_summary",
        status="success",
        inputs={"states": list(states.keys())},
        outputs={
            "summary_path": str(summary_path),
            "manifest_path": str(manifest_path),
        },
    )
    write_step_state(output_dir, result)
    return result


def _load_states(output_dir: Path) -> dict[str, Any]:
    """Load all step state files from workspace."""
    root = state_dir(output_dir)
    if not root.exists():
        return {}

    states: dict[str, Any] = {}
    for file_path in sorted(root.glob("*.json")):
        states[file_path.stem] = read_json(file_path)
    return states


def _count_csv_rows(path: Path) -> int:
    """Count data rows in a CSV file excluding the header."""
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def _collect_artifacts(output_dir: Path) -> list[dict[str, Any]]:
    """Collect artifact metadata under output directory except log files."""
    artifacts: list[dict[str, Any]] = []
    for file_path in sorted(output_dir.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.name.endswith(".log"):
            continue
        artifacts.append(
            {
                "path": str(file_path),
                "sha256": hash_file(file_path),
                "size": file_path.stat().st_size,
            }
        )
    return artifacts
