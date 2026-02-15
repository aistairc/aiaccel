"""Publish step for summary and manifest artifact generation."""

from __future__ import annotations

from typing import Any

from pathlib import Path

from .common import StepResult, hash_file, read_json, scenario_path, write_json, write_step_state
from .config import BridgeConfig


def publish_summary(config: BridgeConfig) -> StepResult:
    """Aggregate scenario artifacts and emit summary/manifest JSON."""
    output_dir = config.bridge.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    states = _load_states(output_dir)
    scenarios: dict[str, dict[str, Any]] = {}
    for scenario in config.bridge.scenarios:
        scenario_output_path = scenario_path(output_dir, scenario.name)
        train_metrics = scenario_output_path / "metrics" / "train_metrics.json"
        eval_metrics = scenario_output_path / "metrics" / "eval_metrics.json"
        scenarios[scenario.name] = {
            "train_pairs": _count_csv_rows(scenario_output_path / "train_pairs.csv"),
            "eval_pairs": _count_csv_rows(scenario_output_path / "test_pairs.csv"),
            "train_metrics": read_json(train_metrics) if train_metrics.exists() else {},
            "eval_metrics": read_json(eval_metrics) if eval_metrics.exists() else {},
        }

    summary_path = write_json(output_dir / "summary.json", scenarios)
    manifest_path = write_json(
        output_dir / "manifest.json",
        {
            "states": states,
            "scenarios": scenarios,
            "artifacts": _collect_artifacts(output_dir),
            "config": config.model_dump(mode="json"),
        },
    )

    result = StepResult(
        step="publish_summary",
        status="success",
        inputs={"states": sorted(states.keys())},
        outputs={"summary_path": str(summary_path), "manifest_path": str(manifest_path)},
    )
    write_step_state(output_dir, result)
    return result


def _load_states(output_dir: Path) -> dict[str, Any]:
    """Load previously written step state JSON files."""
    state_dir = output_dir / "workspace" / "state"
    if not state_dir.exists():
        return {}
    return {path.stem: read_json(path) for path in sorted(state_dir.glob("*.json"))}


def _count_csv_rows(path: Path) -> int:
    """Count CSV data rows excluding header line."""
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def _collect_artifacts(output_dir: Path) -> list[dict[str, Any]]:
    """Collect hash and size metadata for non-log output files."""
    artifacts: list[dict[str, Any]] = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.suffix == ".log":
            continue
        artifacts.append({"path": str(path), "sha256": hash_file(path), "size": path.stat().st_size})
    return artifacts
