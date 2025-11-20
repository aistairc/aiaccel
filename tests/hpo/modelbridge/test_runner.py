from __future__ import annotations

from typing import Any

import json
from pathlib import Path

from aiaccel.hpo.modelbridge.config import load_bridge_config
from aiaccel.hpo.modelbridge.runner import run_pipeline


def _config(tmp_path: Path) -> dict[str, object]:
    return {
        "hpo": {"optimizer": "optuna", "sampler": "tpe"},
        "bridge": {
            "output_dir": str(tmp_path / "outputs"),
            "seed": 7,
            "train_runs": 1,
            "eval_runs": 1,
            "scenarios": [
                {
                    "name": "demo",
                    "macro_trials": 3,
                    "micro_trials": 3,
                    "objective": {
                        "target": "tests.hpo.modelbridge.sample_objective.objective",
                    },
                    "params": {
                        "macro": {"x": {"low": 0.0, "high": 1.0}},
                        "micro": {"y": {"low": 0.0, "high": 1.0}},
                    },
                    "regression": {"degree": 1},
                }
            ],
        },
    }


def test_run_pipeline(tmp_path: Path) -> None:
    bridge_config = load_bridge_config(_config(tmp_path))
    summary = run_pipeline(bridge_config)

    summary_path = tmp_path / "outputs" / "summary.json"
    assert summary_path.exists()
    with summary_path.open("r", encoding="utf-8") as handle:
        summary_file: dict[str, Any] = json.load(handle)
    assert summary_file == summary

    scenario_summary = summary["demo"]
    assert "train_pairs" in scenario_summary
    scenario_dir = tmp_path / "outputs" / "scenarios" / "demo"
    regression_json = scenario_dir / "regression_train.json"
    metrics_json = scenario_dir / "regression_test_metrics.json"
    assert regression_json.exists()
    assert metrics_json.exists()
    metrics_payload = json.loads(metrics_json.read_text(encoding="utf-8"))
    assert "mae" in metrics_payload
    runs_root = tmp_path / "outputs" / "runs" / "demo"
    assert (runs_root / "train" / "macro" / "000" / "optuna.db").exists()
    assert (runs_root / "eval" / "micro" / "000" / "optuna.db").exists()


def test_run_pipeline_partial_phases(tmp_path: Path) -> None:
    bridge_config = load_bridge_config(_config(tmp_path))
    scenario_dir = tmp_path / "outputs" / "scenarios" / "demo"
    runs_root = tmp_path / "outputs" / "runs" / "demo"

    partial = run_pipeline(bridge_config, phases=("hpo",))
    assert partial["phases"] == ["hpo"]
    assert (runs_root / "train" / "macro" / "000" / "optuna.db").exists()
    assert not (scenario_dir / "regression_train.json").exists()

    run_pipeline(bridge_config, phases=("regress",))
    regression_path = scenario_dir / "regression_train.json"
    assert regression_path.exists()

    summary = run_pipeline(bridge_config, phases=("summary",))
    assert "demo" in summary and "train_pairs" in summary["demo"]


def test_run_pipeline_custom_metrics(tmp_path: Path) -> None:
    data = _config(tmp_path)
    data["bridge"]["scenarios"][0]["metrics"] = ["mae"]
    bridge_config = load_bridge_config(data)
    run_pipeline(bridge_config)
    metrics_path = tmp_path / "outputs" / "scenarios" / "demo" / "regression_test_metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert list(payload.keys()) == ["mae"]
